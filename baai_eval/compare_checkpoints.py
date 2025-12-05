#!/usr/bin/env python3
"""
RDT-1B 多Checkpoint对比评估脚本

功能：
1. 从指定episode中随机选取一帧
2. 使用多个checkpoint (如3k, 6k, 9k, 12k, 14k) 对同一帧进行评估
3. 对比不同checkpoint的预测结果，分析训练进度
4. 生成对比图像和统计报告

Author: AI Assistant
"""

import os
import sys
import json
import yaml
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.rdt_runner import RDTRunner


# ============================================================================
# 配置和常量 (从 open_loop_eval.py 复制)
# ============================================================================

BAAI_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"right_hand_joint_{i}_pos"] for i in range(12)
] + [
    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"left_hand_joint_{i}_pos"] for i in range(12)
]

JOINT_NAMES = [
    "R_Arm_J0", "R_Arm_J1", "R_Arm_J2", "R_Arm_J3", "R_Arm_J4", "R_Arm_J5",
    "R_Hand_J0", "R_Hand_J1", "R_Hand_J2", "R_Hand_J3", "R_Hand_J4", "R_Hand_J5",
    "R_Hand_J6", "R_Hand_J7", "R_Hand_J8", "R_Hand_J9", "R_Hand_J10", "R_Hand_J11",
    "L_Arm_J0", "L_Arm_J1", "L_Arm_J2", "L_Arm_J3", "L_Arm_J4", "L_Arm_J5",
    "L_Hand_J0", "L_Hand_J1", "L_Hand_J2", "L_Hand_J3", "L_Hand_J4", "L_Hand_J5",
    "L_Hand_J6", "L_Hand_J7", "L_Hand_J8", "L_Hand_J9", "L_Hand_J10", "L_Hand_J11",
]

JOINT_GROUPS = {
    "right_arm": list(range(0, 6)),
    "right_hand": list(range(6, 18)),
    "left_arm": list(range(18, 24)),
    "left_hand": list(range(24, 36)),
}

JOINT_GROUP_NAMES_ZH = {
    "right_arm": "Right Arm",
    "right_hand": "Right Hand",
    "left_arm": "Left Arm",
    "left_hand": "Left Hand",
}

JOINT_NAMES_DETAILED = {
    "right_arm": ["R_Arm_J0", "R_Arm_J1", "R_Arm_J2", "R_Arm_J3", "R_Arm_J4", "R_Arm_J5"],
    "right_hand": ["R_Hand_J0", "R_Hand_J1", "R_Hand_J2", "R_Hand_J3",
                   "R_Hand_J4", "R_Hand_J5", "R_Hand_J6", "R_Hand_J7",
                   "R_Hand_J8", "R_Hand_J9", "R_Hand_J10", "R_Hand_J11"],
    "left_arm": ["L_Arm_J0", "L_Arm_J1", "L_Arm_J2", "L_Arm_J3", "L_Arm_J4", "L_Arm_J5"],
    "left_hand": ["L_Hand_J0", "L_Hand_J1", "L_Hand_J2", "L_Hand_J3",
                  "L_Hand_J4", "L_Hand_J5", "L_Hand_J6", "L_Hand_J7",
                  "L_Hand_J8", "L_Hand_J9", "L_Hand_J10", "L_Hand_J11"],
}


# ============================================================================
# 模型加载 (简化版，支持快速切换checkpoint)
# ============================================================================

class MultiCheckpointEvaluator:
    """支持多checkpoint评估的模型类"""
    
    def __init__(
        self,
        config_path: str = "configs/base.yaml",
        vision_encoder_path: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        control_frequency: int = 20,
    ):
        self.device = device
        self.dtype = dtype
        self.control_frequency = control_frequency
        
        # 加载配置
        config_file = project_root / config_path
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"📂 Config: {config_file}")
        
        # 加载视觉编码器 (只加载一次)
        print("🔄 加载视觉编码器...")
        self.vision_encoder = SiglipVisionTower(
            vision_tower=vision_encoder_path, 
            args=None
        )
        self.image_processor = self.vision_encoder.image_processor
        self.vision_encoder = self.vision_encoder.to(device, dtype=dtype)
        self.vision_encoder.eval()
        print(f"   ✅ SigLIP已加载, num_patches={self.vision_encoder.num_patches}")
        
        # 计算图像条件长度
        self.img_cond_len = (
            self.config["common"]["img_history_size"] 
            * self.config["common"]["num_cameras"] 
            * self.vision_encoder.num_patches
        )
        
        self.policy = None
        self.current_checkpoint = None
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载指定的checkpoint"""
        if self.current_checkpoint == checkpoint_path:
            return  # 已经加载了相同的checkpoint
        
        print(f"\n🔄 加载checkpoint: {checkpoint_path}")
        
        # 创建新的RDT模型
        self.policy = RDTRunner(
            action_dim=self.config["common"]["state_dim"],
            pred_horizon=self.config["common"]["action_chunk_size"],
            config=self.config["model"],
            lang_token_dim=self.config["model"]["lang_token_dim"],
            img_token_dim=self.config["model"]["img_token_dim"],
            state_token_dim=self.config["model"]["state_token_dim"],
            max_lang_cond_len=self.config["dataset"]["tokenizer_max_length"],
            img_cond_len=self.img_cond_len,
            img_pos_embed_config=[
                ("image", (
                    self.config["common"]["img_history_size"],
                    self.config["common"]["num_cameras"],
                    -self.vision_encoder.num_patches
                )),
            ],
            lang_pos_embed_config=[
                ("lang", -self.config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=self.dtype,
        )
        
        # 加载权重
        checkpoint_file = Path(checkpoint_path) / "pytorch_model.bin"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"未找到checkpoint文件: {checkpoint_file}")
        
        state_dict = torch.load(checkpoint_file, map_location='cpu')
        
        if "module" in state_dict:
            state_dict = state_dict["module"]
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.policy.load_state_dict(new_state_dict, strict=False)
        self.policy = self.policy.to(self.device, dtype=self.dtype)
        self.policy.eval()
        
        self.current_checkpoint = checkpoint_path
        print(f"   ✅ 权重加载成功!")
    
    def _format_state_to_unified(self, state_36: np.ndarray) -> np.ndarray:
        if state_36.ndim == 1:
            state_36 = state_36[np.newaxis, :]
        B, D = state_36.shape
        state_128 = np.zeros((B, self.config["common"]["state_dim"]), dtype=np.float32)
        state_128[:, BAAI_STATE_INDICES] = state_36.astype(np.float32)
        return state_128
    
    def _unformat_unified_to_state(self, action_128: np.ndarray) -> np.ndarray:
        if action_128.ndim == 2:
            return action_128[:, BAAI_STATE_INDICES]
        elif action_128.ndim == 3:
            return action_128[:, :, BAAI_STATE_INDICES]
        return action_128[BAAI_STATE_INDICES]
    
    def preprocess_images(self, images: list) -> torch.Tensor:
        background_color = np.array([
            int(x * 255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.image_processor.size["height"],
            self.image_processor.size["width"], 3
        ), dtype=np.uint8) * background_color
        
        image_tensors = []
        for img in images:
            if img is None:
                img = Image.fromarray(background_image)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            width, height = img.size
            if width != height:
                size = max(width, height)
                new_img = Image.new(
                    img.mode, (size, size),
                    tuple(int(x * 255) for x in self.image_processor.image_mean)
                )
                new_img.paste(img, ((size - width) // 2, (size - height) // 2))
                img = new_img
            
            processed = self.image_processor.preprocess(img, return_tensors='pt')
            image_tensors.append(processed['pixel_values'][0])
        
        image_tensor = torch.stack(image_tensors, dim=0).to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            image_embeds = self.vision_encoder(image_tensor)
            image_embeds = image_embeds.reshape(-1, self.vision_encoder.hidden_size)
            image_embeds = image_embeds.unsqueeze(0)
        
        return image_embeds
    
    @torch.no_grad()
    def predict(self, state_36: np.ndarray, images: list, lang_embeds: torch.Tensor) -> np.ndarray:
        state_128 = self._format_state_to_unified(state_36).astype(np.float32)
        state_tensor = torch.from_numpy(state_128).to(device=self.device, dtype=self.dtype)
        state_tensor = state_tensor.unsqueeze(1)
        
        state_mask = np.zeros(self.config["common"]["state_dim"], dtype=np.float32)
        state_mask[BAAI_STATE_INDICES] = 1
        state_mask_tensor = torch.from_numpy(state_mask).to(device=self.device, dtype=self.dtype)
        state_mask_tensor = state_mask_tensor.unsqueeze(0).unsqueeze(0)
        
        image_embeds = self.preprocess_images(images)
        
        if lang_embeds.ndim == 2:
            lang_embeds = lang_embeds.unsqueeze(0)
        lang_embeds = lang_embeds.to(device=self.device, dtype=self.dtype)
        lang_attn_mask = torch.ones(lang_embeds.shape[:2], dtype=torch.bool, device=self.device)
        
        ctrl_freqs = torch.tensor([self.control_frequency], device=self.device)
        
        predicted_actions = self.policy.predict_action(
            lang_tokens=lang_embeds,
            lang_attn_mask=lang_attn_mask,
            img_tokens=image_embeds,
            state_tokens=state_tensor,
            action_mask=state_mask_tensor,
            ctrl_freqs=ctrl_freqs,
        )
        
        predicted_actions = predicted_actions.squeeze(0).float().cpu().numpy()
        predicted_actions_36 = self._unformat_unified_to_state(predicted_actions)
        
        return predicted_actions_36


# ============================================================================
# 数据加载
# ============================================================================

def load_episode_cache(cache_dir: str, episode_idx: int) -> dict:
    cache_file = Path(cache_dir) / f"episode_{episode_idx:06d}.pt"
    if not cache_file.exists():
        raise FileNotFoundError(f"未找到episode缓存: {cache_file}")
    
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.numeric'] = getattr(np.core, 'numeric', np.core)
    
    return torch.load(cache_file, map_location='cpu', weights_only=False)


def get_sample_from_episode(episode_cache: dict, episode_idx: int, step_idx: int, chunk_size: int = 64) -> dict:
    qpos = episode_cache["state"]
    actions = episode_cache["action"]
    num_steps = episode_cache["frame_num"]
    images_info = episode_cache.get("images_info", {})
    
    state = np.asarray(qpos[step_idx], dtype=np.float32)
    action_gt = np.asarray(actions[step_idx:step_idx + chunk_size], dtype=np.float32)
    
    if len(action_gt) < chunk_size:
        pad_len = chunk_size - len(action_gt)
        action_gt = np.concatenate([action_gt, np.tile(action_gt[-1:], (pad_len, 1))], axis=0)
    
    def load_image(cam_key, frame_idx):
        if cam_key not in images_info:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        cam_data = images_info[cam_key]
        if isinstance(cam_data, np.ndarray):
            if frame_idx < len(cam_data):
                img = cam_data[frame_idx]
                return img.astype(np.uint8) if img.dtype != np.uint8 else img
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    images = []
    for t_offset in [-1, 0]:
        t = max(0, step_idx + t_offset)
        images.append(load_image('camera_head', t))
        images.append(load_image('camera_right_wrist', t))
        images.append(load_image('camera_left_wrist', t))
    
    return {
        "state": state,
        "action_gt": action_gt,
        "images": images,
        "meta": {"episode_idx": episode_idx, "step_idx": step_idx, "num_steps": num_steps}
    }


# ============================================================================
# 评估指标
# ============================================================================

def compute_metrics(action_gt: np.ndarray, action_pred: np.ndarray) -> Dict[str, float]:
    mse = np.mean((action_gt - action_pred) ** 2)
    mae = np.mean(np.abs(action_gt - action_pred))
    rmse = np.sqrt(mse)
    
    mse_per_joint = np.mean((action_gt - action_pred) ** 2, axis=0)
    mae_per_joint = np.mean(np.abs(action_gt - action_pred), axis=0)
    
    metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "mse_per_joint": mse_per_joint.tolist(),
        "mae_per_joint": mae_per_joint.tolist(),
    }
    
    for group_name, indices in JOINT_GROUPS.items():
        metrics[f"mse_{group_name}"] = float(np.mean(mse_per_joint[indices]))
        metrics[f"mae_{group_name}"] = float(np.mean(mae_per_joint[indices]))
    
    return metrics


# ============================================================================
# 可视化
# ============================================================================

def plot_checkpoint_comparison_by_group(
    action_gt: np.ndarray,
    predictions: Dict[str, np.ndarray],  # {ckpt_name: action_pred}
    save_dir: str,
    episode_idx: int,
    step_idx: int,
):
    """
    为每个部位生成checkpoint对比图
    每个关节一个子图，展示GT和不同checkpoint的预测
    """
    chunk_size = action_gt.shape[0]
    timesteps = np.arange(chunk_size)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 颜色方案：GT用黑色，不同checkpoint用不同颜色
    ckpt_names = list(predictions.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ckpt_names)))
    
    for group_name, joint_indices in JOINT_GROUPS.items():
        num_joints = len(joint_indices)
        
        if num_joints == 6:
            nrows, ncols = 2, 3
            figsize = (18, 10)
        else:
            nrows, ncols = 3, 4
            figsize = (22, 14)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=100)
        axes = axes.flatten()
        
        joint_names = JOINT_NAMES_DETAILED[group_name]
        
        for i, (joint_idx, joint_name) in enumerate(zip(joint_indices, joint_names)):
            ax = axes[i]
            
            # 绘制GT
            ax.plot(timesteps, action_gt[:, joint_idx], 'k-', linewidth=2.5, 
                   label='GT', alpha=0.9)
            
            # 绘制各个checkpoint的预测
            for ckpt_idx, (ckpt_name, action_pred) in enumerate(predictions.items()):
                # 缩短legend名称: checkpoint-3000 -> ckpt-3k
                short_label = ckpt_name.replace("checkpoint-", "ckpt-").replace("000", "k")
                ax.plot(timesteps, action_pred[:, joint_idx], 
                       color=colors[ckpt_idx], linestyle='--', linewidth=1.5,
                       label=short_label, alpha=0.8)
            
            # 计算各checkpoint的MSE，使用缩短的名称
            mse_list = []
            for name, pred in predictions.items():
                mse_val = np.mean((action_gt[:, joint_idx] - pred[:, joint_idx])**2)
                # 缩短checkpoint名称: checkpoint-3000 -> 3k
                short_name = name.replace("checkpoint-", "").replace("000", "k")
                # 对于很小的MSE使用科学计数法
                if mse_val < 0.0001:
                    mse_list.append(f"{short_name}:{mse_val:.1e}")
                else:
                    mse_list.append(f"{short_name}:{mse_val:.4f}")
            
            # 每2-3个checkpoint换行，避免太长
            if len(mse_list) <= 3:
                mse_str = "  ".join(mse_list)
            else:
                mid = (len(mse_list) + 1) // 2
                mse_str = "  ".join(mse_list[:mid]) + "\n" + "  ".join(mse_list[mid:])
            
            ax.set_title(f"{joint_name}\n{mse_str}", fontsize=8)
            ax.set_xlabel('Time Step', fontsize=8)
            ax.set_ylabel('Angle (rad)', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=7, ncol=2)
            ax.tick_params(labelsize=7)
        
        for i in range(num_joints, len(axes)):
            axes[i].set_visible(False)
        
        title = f"{JOINT_GROUP_NAMES_ZH[group_name]} - Episode {episode_idx}, Step {step_idx}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = save_dir / f"ep{episode_idx:04d}_step{step_idx:04d}_{group_name}_compare.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_mse_trend(
    checkpoint_metrics: Dict[str, Dict],  # {ckpt_name: metrics}
    save_path: str,
    episode_idx: int,
    step_idx: int,
):
    """
    绘制不同checkpoint的MSE变化趋势
    """
    ckpt_names = list(checkpoint_metrics.keys())
    ckpt_steps = []
    for name in ckpt_names:
        # 从checkpoint名称提取步数，如 "ckpt-3000" -> 3000
        try:
            step = int(name.split('-')[-1].replace('k', '000'))
        except:
            step = 0
        ckpt_steps.append(step)
    
    # 排序
    sorted_indices = np.argsort(ckpt_steps)
    ckpt_names = [ckpt_names[i] for i in sorted_indices]
    ckpt_steps = [ckpt_steps[i] for i in sorted_indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    
    # 1. 总体MSE趋势
    ax = axes[0, 0]
    mse_values = [checkpoint_metrics[name]["mse"] for name in ckpt_names]
    ax.plot(ckpt_steps, mse_values, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Total MSE vs Training Steps', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for x, y, name in zip(ckpt_steps, mse_values, ckpt_names):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    # 2. 分部位MSE趋势
    ax = axes[0, 1]
    group_colors = {'right_arm': '#3498db', 'right_hand': '#2ecc71', 
                    'left_arm': '#e74c3c', 'left_hand': '#f39c12'}
    
    for group_name in JOINT_GROUPS.keys():
        mse_values = [checkpoint_metrics[name][f"mse_{group_name}"] for name in ckpt_names]
        ax.plot(ckpt_steps, mse_values, 'o-', linewidth=2, markersize=6,
               color=group_colors[group_name], label=JOINT_GROUP_NAMES_ZH[group_name])
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE by Joint Group vs Training Steps', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. MSE改进百分比
    ax = axes[1, 0]
    if len(ckpt_names) > 1:
        first_mse = checkpoint_metrics[ckpt_names[0]]["mse"]
        improvement = [(1 - checkpoint_metrics[name]["mse"] / first_mse) * 100 
                      for name in ckpt_names]
        bars = ax.bar(range(len(ckpt_names)), improvement, 
                     color=['#3498db' if v >= 0 else '#e74c3c' for v in improvement])
        ax.set_xticks(range(len(ckpt_names)))
        ax.set_xticklabels([f"{s//1000}k" for s in ckpt_steps], fontsize=10)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Improvement %', fontsize=12)
        ax.set_title(f'MSE Improvement vs {ckpt_names[0]}', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, improvement):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # 4. 分部位MSE条形图对比
    ax = axes[1, 1]
    x = np.arange(len(JOINT_GROUPS))
    width = 0.8 / len(ckpt_names)
    
    for i, name in enumerate(ckpt_names):
        mse_values = [checkpoint_metrics[name][f"mse_{g}"] for g in JOINT_GROUPS.keys()]
        offset = (i - len(ckpt_names)/2 + 0.5) * width
        ax.bar(x + offset, mse_values, width, label=f"{ckpt_steps[i]//1000}k", alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([JOINT_GROUP_NAMES_ZH[g] for g in JOINT_GROUPS.keys()], fontsize=10)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE by Joint Group (All Checkpoints)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, title='Steps')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Checkpoint Comparison - Episode {episode_idx}, Step {step_idx}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_comparison_json(
    action_gt: np.ndarray,
    predictions: Dict[str, np.ndarray],
    checkpoint_metrics: Dict[str, Dict],
    save_path: str,
    episode_idx: int,
    step_idx: int,
):
    """保存对比结果到JSON"""
    data = {
        "meta": {
            "episode_idx": int(episode_idx),
            "step_idx": int(step_idx),
            "chunk_size": int(action_gt.shape[0]),
            "num_checkpoints": len(predictions),
        },
        "checkpoints": {},
        "summary": {},
    }
    
    # 每个checkpoint的详细数据
    for ckpt_name, action_pred in predictions.items():
        metrics = checkpoint_metrics[ckpt_name]
        data["checkpoints"][ckpt_name] = {
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mse_right_arm": metrics["mse_right_arm"],
            "mse_right_hand": metrics["mse_right_hand"],
            "mse_left_arm": metrics["mse_left_arm"],
            "mse_left_hand": metrics["mse_left_hand"],
            "mse_per_joint": metrics["mse_per_joint"],
        }
    
    # 汇总对比
    ckpt_names = list(predictions.keys())
    mse_values = [checkpoint_metrics[name]["mse"] for name in ckpt_names]
    best_idx = np.argmin(mse_values)
    worst_idx = np.argmax(mse_values)
    
    data["summary"] = {
        "best_checkpoint": ckpt_names[best_idx],
        "best_mse": mse_values[best_idx],
        "worst_checkpoint": ckpt_names[worst_idx],
        "worst_mse": mse_values[worst_idx],
        "improvement_ratio": mse_values[worst_idx] / mse_values[best_idx] if mse_values[best_idx] > 0 else 0,
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RDT-1B 多Checkpoint对比评估")
    parser.add_argument(
        "--checkpoint_base", type=str,
        default="./checkpoints/rdt1b-full-action176-20251202_000048",
        help="Checkpoint基础路径"
    )
    parser.add_argument(
        "--checkpoints", type=str,
        default="checkpoint-3000,checkpoint-6000,checkpoint-9000,checkpoint-12000,checkpoint-14000",
        help="要对比的checkpoint列表，逗号分隔"
    )
    parser.add_argument(
        "--dataset", type=str,
        default="./data/baai/data/lerobot_baai",
        help="数据集路径"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/base.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--vision_encoder", type=str,
        default="google/siglip-so400m-patch14-384",
        help="视觉编码器路径"
    )
    parser.add_argument(
        "--episode_idx", type=int, default=None,
        help="指定episode索引，默认随机选择"
    )
    parser.add_argument(
        "--step_idx", type=int, default=None,
        help="指定step索引，默认随机选择"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results/checkpoint_compare",
        help="输出目录"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 解析checkpoint列表
    checkpoint_names = [c.strip() for c in args.checkpoints.split(',')]
    checkpoint_paths = [str(Path(args.checkpoint_base) / name) for name in checkpoint_names]
    
    # 验证checkpoint存在
    valid_checkpoints = []
    for name, path in zip(checkpoint_names, checkpoint_paths):
        ckpt_file = Path(path) / "pytorch_model.bin"
        if ckpt_file.exists():
            valid_checkpoints.append((name, path))
        else:
            print(f"⚠️  Checkpoint不存在，跳过: {path}")
    
    if len(valid_checkpoints) == 0:
        print("❌ 没有找到有效的checkpoint!")
        return
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"compare_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🎯 RDT-1B 多Checkpoint对比评估")
    print("=" * 70)
    print(f"📂 Checkpoint Base: {args.checkpoint_base}")
    print(f"📂 Checkpoints: {[name for name, _ in valid_checkpoints]}")
    print(f"📂 Dataset: {args.dataset}")
    print(f"📂 Output: {output_dir}")
    
    # 检查CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    
    # 初始化评估器
    print("\n" + "=" * 70)
    print("🚀 初始化评估器")
    print("=" * 70)
    
    evaluator = MultiCheckpointEvaluator(
        config_path=args.config,
        vision_encoder_path=args.vision_encoder,
        device=args.device,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        control_frequency=20,
    )
    
    # 加载语言嵌入
    dataset_path = Path(args.dataset)
    lang_embed_path = dataset_path / "instruction.pt"
    if lang_embed_path.exists():
        lang_embeds = torch.load(lang_embed_path, map_location='cpu')
        print(f"📝 语言嵌入: {lang_embeds.shape}")
    else:
        raise FileNotFoundError(f"未找到语言嵌入文件: {lang_embed_path}")
    
    # 获取episode
    cache_dir = dataset_path / "cache"
    all_episodes = sorted([
        int(f.stem.split('_')[1]) 
        for f in cache_dir.glob("episode_*.pt")
        if f.stem != "episode_metadata"
    ])
    
    # 选择episode
    if args.episode_idx is not None:
        episode_idx = args.episode_idx
    else:
        episode_idx = random.choice(all_episodes)
    
    print(f"\n📋 选择 Episode: {episode_idx}")
    
    # 加载episode数据
    episode_cache = load_episode_cache(str(cache_dir), episode_idx)
    num_steps = episode_cache["frame_num"]
    qpos = episode_cache["state"]
    
    # 找到运动起始点
    EPS = 1e-2
    qpos_delta = np.abs(qpos - qpos[0:1])
    indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
    first_idx = indices[0] if len(indices) > 0 else 1
    max_valid_step = max(first_idx, num_steps - 64 - 1)
    
    # 选择step
    if args.step_idx is not None:
        step_idx = args.step_idx
    else:
        step_idx = random.randint(first_idx, max_valid_step)
    
    print(f"📋 选择 Step: {step_idx} (Episode范围: {first_idx}-{max_valid_step})")
    
    # 获取样本
    sample = get_sample_from_episode(episode_cache, episode_idx, step_idx, chunk_size=64)
    action_gt = sample["action_gt"]
    
    # 对每个checkpoint进行评估
    print("\n" + "=" * 70)
    print("🔄 开始评估各Checkpoint")
    print("=" * 70)
    
    predictions = {}
    checkpoint_metrics = {}
    
    for ckpt_name, ckpt_path in tqdm(valid_checkpoints, desc="Evaluating checkpoints"):
        try:
            evaluator.load_checkpoint(ckpt_path)
            
            with torch.inference_mode():
                action_pred = evaluator.predict(
                    state_36=sample["state"],
                    images=sample["images"],
                    lang_embeds=lang_embeds,
                )
            
            predictions[ckpt_name] = action_pred
            metrics = compute_metrics(action_gt, action_pred)
            checkpoint_metrics[ckpt_name] = metrics
            
            print(f"   {ckpt_name}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
            
        except Exception as e:
            print(f"   ⚠️ {ckpt_name} 评估失败: {e}")
            continue
    
    if len(predictions) == 0:
        print("❌ 没有成功评估的checkpoint!")
        return
    
    # 生成可视化
    print("\n" + "=" * 70)
    print("📊 生成可视化图表")
    print("=" * 70)
    
    # 1. 每个部位的对比图
    plot_checkpoint_comparison_by_group(
        action_gt, predictions, str(output_dir), episode_idx, step_idx
    )
    print("   ✅ 分部位对比图已保存")
    
    # 2. MSE趋势图
    plot_mse_trend(
        checkpoint_metrics, 
        str(output_dir / f"ep{episode_idx:04d}_step{step_idx:04d}_mse_trend.png"),
        episode_idx, step_idx
    )
    print("   ✅ MSE趋势图已保存")
    
    # 3. 保存JSON结果
    save_comparison_json(
        action_gt, predictions, checkpoint_metrics,
        str(output_dir / f"ep{episode_idx:04d}_step{step_idx:04d}_comparison.json"),
        episode_idx, step_idx
    )
    print("   ✅ 对比结果JSON已保存")
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("📊 评估汇总")
    print("=" * 70)
    
    ckpt_names = list(checkpoint_metrics.keys())
    mse_values = [checkpoint_metrics[name]["mse"] for name in ckpt_names]
    best_idx = np.argmin(mse_values)
    
    print(f"\n🏆 最佳Checkpoint: {ckpt_names[best_idx]} (MSE={mse_values[best_idx]:.6f})")
    print(f"\n📈 各Checkpoint MSE对比:")
    for name in ckpt_names:
        m = checkpoint_metrics[name]
        print(f"   {name}:")
        print(f"      总体: MSE={m['mse']:.6f}, MAE={m['mae']:.6f}")
        print(f"      右臂: {m['mse_right_arm']:.6f}, 右手: {m['mse_right_hand']:.6f}")
        print(f"      左臂: {m['mse_left_arm']:.6f}, 左手: {m['mse_left_hand']:.6f}")
    
    print("\n" + "=" * 70)
    print("✅ 对比评估完成!")
    print("=" * 70)
    print(f"\n📂 结果保存在: {output_dir}")


if __name__ == "__main__":
    main()

