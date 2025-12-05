#!/usr/bin/env python3
"""
RDT-1B 开环评估脚本 (Open-Loop Evaluation)

功能：
1. 在训练集的多个episode上进行批量推理
2. 对比预测的action chunk与真实的action chunk
3. 统计MSE、MAE等指标，支持分关节、分部位统计
4. 生成可视化图表（分组对比图、热力图、误差分布图）
5. 保存详细的评估结果供分析

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
from typing import Dict, List, Tuple, Optional

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
# 配置和常量
# ============================================================================

# 状态向量索引映射：36维动作向量 -> 128维统一向量
# 顺序与info.json中的action一致：右臂6 + 右手12 + 左臂6 + 左手12
BAAI_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"right_hand_joint_{i}_pos"] for i in range(12)
] + [
    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"left_hand_joint_{i}_pos"] for i in range(12)
]

# 关节名称（用于绘图和统计）
JOINT_NAMES = [
    # 右臂 (0-5)
    "R_Arm_J0", "R_Arm_J1", "R_Arm_J2", "R_Arm_J3", "R_Arm_J4", "R_Arm_J5",
    # 右手 (6-17)
    "R_Hand_J0", "R_Hand_J1", "R_Hand_J2", "R_Hand_J3", "R_Hand_J4", "R_Hand_J5",
    "R_Hand_J6", "R_Hand_J7", "R_Hand_J8", "R_Hand_J9", "R_Hand_J10", "R_Hand_J11",
    # 左臂 (18-23)
    "L_Arm_J0", "L_Arm_J1", "L_Arm_J2", "L_Arm_J3", "L_Arm_J4", "L_Arm_J5",
    # 左手 (24-35)
    "L_Hand_J0", "L_Hand_J1", "L_Hand_J2", "L_Hand_J3", "L_Hand_J4", "L_Hand_J5",
    "L_Hand_J6", "L_Hand_J7", "L_Hand_J8", "L_Hand_J9", "L_Hand_J10", "L_Hand_J11",
]

# 关节分组
JOINT_GROUPS = {
    "right_arm": list(range(0, 6)),
    "right_hand": list(range(6, 18)),
    "left_arm": list(range(18, 24)),
    "left_hand": list(range(24, 36)),
}

JOINT_GROUP_NAMES_ZH = {
    "right_arm": "右臂 (Right Arm)",
    "right_hand": "右手 (Right Hand)",
    "left_arm": "左臂 (Left Arm)",
    "left_hand": "左手 (Left Hand)",
}


# ============================================================================
# 模型加载
# ============================================================================

class BAAIEvalModel:
    """用于开环评估的RDT模型封装类"""
    
    def __init__(
        self,
        checkpoint_path: str,
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
        
        print(f"📂 Checkpoint: {checkpoint_path}")
        print(f"📂 Config: {config_file}")
        
        # 加载视觉编码器
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
        img_cond_len = (
            self.config["common"]["img_history_size"] 
            * self.config["common"]["num_cameras"] 
            * self.vision_encoder.num_patches
        )
        
        # 创建RDT模型
        print("🔄 创建RDT模型...")
        self.policy = RDTRunner(
            action_dim=self.config["common"]["state_dim"],
            pred_horizon=self.config["common"]["action_chunk_size"],
            config=self.config["model"],
            lang_token_dim=self.config["model"]["lang_token_dim"],
            img_token_dim=self.config["model"]["img_token_dim"],
            state_token_dim=self.config["model"]["state_token_dim"],
            max_lang_cond_len=self.config["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
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
            dtype=dtype,
        )
        
        # 加载checkpoint权重
        print("🔄 加载checkpoint权重...")
        self._load_checkpoint(checkpoint_path)
        
        # 移动到设备并设置为评估模式
        self.policy = self.policy.to(device, dtype=dtype)
        self.policy.eval()
        
        print("✅ 模型初始化完成！")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint权重"""
        checkpoint_file = Path(checkpoint_path) / "pytorch_model.bin"
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"未找到checkpoint文件: {checkpoint_file}")
        
        print(f"   📦 加载权重: {checkpoint_file}")
        state_dict = torch.load(checkpoint_file, map_location='cpu')
        
        # 处理DeepSpeed保存的state_dict格式
        if "module" in state_dict:
            state_dict = state_dict["module"]
        
        # 移除可能的 "module." 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.policy.load_state_dict(new_state_dict, strict=False)
        print(f"   ✅ 权重加载成功!")
    
    def _format_state_to_unified(self, state_36: np.ndarray) -> np.ndarray:
        """将36维状态向量映射到128维统一向量"""
        if state_36.ndim == 1:
            state_36 = state_36[np.newaxis, :]
        
        B, D = state_36.shape
        state_128 = np.zeros((B, self.config["common"]["state_dim"]), dtype=np.float32)
        state_128[:, BAAI_STATE_INDICES] = state_36.astype(np.float32)
        return state_128
    
    def _unformat_unified_to_state(self, action_128: np.ndarray) -> np.ndarray:
        """将128维统一向量映射回36维动作向量"""
        if action_128.ndim == 2:
            return action_128[:, BAAI_STATE_INDICES]
        elif action_128.ndim == 3:
            return action_128[:, :, BAAI_STATE_INDICES]
        return action_128[BAAI_STATE_INDICES]
    
    def preprocess_images(self, images: list) -> torch.Tensor:
        """预处理图像列表"""
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
            
            # Pad to square
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
    def predict(
        self,
        state_36: np.ndarray,
        images: list,
        lang_embeds: torch.Tensor,
    ) -> np.ndarray:
        """执行推理，预测action chunk"""
        # 准备状态 - 先转为float32再转为目标dtype，避免numpy不支持bfloat16的问题
        state_128 = self._format_state_to_unified(state_36).astype(np.float32)
        state_tensor = torch.from_numpy(state_128).to(device=self.device, dtype=self.dtype)
        state_tensor = state_tensor.unsqueeze(1)
        
        # 准备状态mask
        state_mask = np.zeros(self.config["common"]["state_dim"], dtype=np.float32)
        state_mask[BAAI_STATE_INDICES] = 1
        state_mask_tensor = torch.from_numpy(state_mask).to(device=self.device, dtype=self.dtype)
        state_mask_tensor = state_mask_tensor.unsqueeze(0).unsqueeze(0)
        
        # 编码图像
        image_embeds = self.preprocess_images(images)
        
        # 准备语言条件
        if lang_embeds.ndim == 2:
            lang_embeds = lang_embeds.unsqueeze(0)
        lang_embeds = lang_embeds.to(device=self.device, dtype=self.dtype)
        lang_attn_mask = torch.ones(
            lang_embeds.shape[:2], dtype=torch.bool, device=self.device
        )
        
        # 准备控制频率
        ctrl_freqs = torch.tensor([self.control_frequency], device=self.device)
        
        # 执行推理
        predicted_actions = self.policy.predict_action(
            lang_tokens=lang_embeds,
            lang_attn_mask=lang_attn_mask,
            img_tokens=image_embeds,
            state_tokens=state_tensor,
            action_mask=state_mask_tensor,
            ctrl_freqs=ctrl_freqs,
        )
        
        # 转回float32再转numpy，因为numpy不支持bfloat16
        predicted_actions = predicted_actions.squeeze(0).float().cpu().numpy()
        predicted_actions_36 = self._unformat_unified_to_state(predicted_actions)
        
        return predicted_actions_36


# ============================================================================
# 数据加载
# ============================================================================

def load_episode_cache(cache_dir: str, episode_idx: int) -> dict:
    """加载缓存的episode数据"""
    cache_file = Path(cache_dir) / f"episode_{episode_idx:06d}.pt"
    
    if not cache_file.exists():
        raise FileNotFoundError(f"未找到episode缓存: {cache_file}")
    
    # 兼容不同numpy版本
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.numeric'] = getattr(np.core, 'numeric', np.core)
    
    return torch.load(cache_file, map_location='cpu', weights_only=False)


def get_sample_from_episode(
    episode_cache: dict,
    episode_idx: int,
    step_idx: int,
    chunk_size: int = 64,
) -> dict:
    """从episode缓存获取单个样本"""
    qpos = episode_cache["state"]
    actions = episode_cache["action"]
    num_steps = episode_cache["frame_num"]
    images_info = episode_cache.get("images_info", {})
    
    # 获取状态和动作，确保转为float32
    state = np.asarray(qpos[step_idx], dtype=np.float32)
    action_gt = np.asarray(actions[step_idx:step_idx + chunk_size], dtype=np.float32)
    
    # 如果action_gt长度不足，用最后一个填充
    if len(action_gt) < chunk_size:
        pad_len = chunk_size - len(action_gt)
        action_gt = np.concatenate([action_gt, np.tile(action_gt[-1:], (pad_len, 1))], axis=0)
    
    # 加载图像
    def load_image(cam_key, frame_idx):
        if cam_key not in images_info:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        cam_data = images_info[cam_key]
        if isinstance(cam_data, np.ndarray):
            if frame_idx < len(cam_data):
                img = cam_data[frame_idx]
                return img.astype(np.uint8) if img.dtype != np.uint8 else img
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 加载2个时间步 x 3个相机的图像
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
        "meta": {
            "episode_idx": episode_idx,
            "step_idx": step_idx,
            "num_steps": num_steps,
        }
    }


# ============================================================================
# 评估指标
# ============================================================================

def compute_metrics(
    action_gt: np.ndarray,
    action_pred: np.ndarray,
) -> Dict[str, float]:
    """计算评估指标"""
    # 基础指标
    mse = np.mean((action_gt - action_pred) ** 2)
    mae = np.mean(np.abs(action_gt - action_pred))
    rmse = np.sqrt(mse)
    
    # 每个关节的指标
    mse_per_joint = np.mean((action_gt - action_pred) ** 2, axis=0)
    mae_per_joint = np.mean(np.abs(action_gt - action_pred), axis=0)
    
    # 分部位指标
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


def aggregate_metrics(all_metrics: List[Dict]) -> Dict:
    """汇总所有样本的指标"""
    agg = {
        "num_samples": len(all_metrics),
        "mse_mean": np.mean([m["mse"] for m in all_metrics]),
        "mse_std": np.std([m["mse"] for m in all_metrics]),
        "mae_mean": np.mean([m["mae"] for m in all_metrics]),
        "mae_std": np.std([m["mae"] for m in all_metrics]),
        "rmse_mean": np.mean([m["rmse"] for m in all_metrics]),
        "rmse_std": np.std([m["rmse"] for m in all_metrics]),
    }
    
    # 汇总分部位指标
    for group_name in JOINT_GROUPS.keys():
        agg[f"mse_{group_name}_mean"] = np.mean([m[f"mse_{group_name}"] for m in all_metrics])
        agg[f"mse_{group_name}_std"] = np.std([m[f"mse_{group_name}"] for m in all_metrics])
        agg[f"mae_{group_name}_mean"] = np.mean([m[f"mae_{group_name}"] for m in all_metrics])
        agg[f"mae_{group_name}_std"] = np.std([m[f"mae_{group_name}"] for m in all_metrics])
    
    # 汇总每个关节的指标
    mse_per_joint_all = np.array([m["mse_per_joint"] for m in all_metrics])
    mae_per_joint_all = np.array([m["mae_per_joint"] for m in all_metrics])
    
    agg["mse_per_joint_mean"] = np.mean(mse_per_joint_all, axis=0).tolist()
    agg["mse_per_joint_std"] = np.std(mse_per_joint_all, axis=0).tolist()
    agg["mae_per_joint_mean"] = np.mean(mae_per_joint_all, axis=0).tolist()
    agg["mae_per_joint_std"] = np.std(mae_per_joint_all, axis=0).tolist()
    
    return agg


def classify_phase(step_idx: int, num_steps: int) -> str:
    """
    根据step在episode中的位置分类阶段
    
    Args:
        step_idx: 当前步数索引
        num_steps: episode总步数
    
    Returns:
        phase: "early" (前1/3), "mid" (中间1/3), "late" (后1/3)
    """
    ratio = step_idx / num_steps
    if ratio < 0.33:
        return "early"
    elif ratio < 0.67:
        return "mid"
    else:
        return "late"


def aggregate_phase_metrics(phase_metrics: Dict[str, List[Dict]]) -> Dict:
    """
    按阶段汇总指标
    
    Args:
        phase_metrics: {"early": [...], "mid": [...], "late": [...]}
    
    Returns:
        Dict with phase-wise aggregated metrics
    """
    phase_agg = {}
    
    for phase_name in ["early", "mid", "late"]:
        metrics_list = phase_metrics.get(phase_name, [])
        if len(metrics_list) == 0:
            continue
            
        phase_agg[phase_name] = {
            "num_samples": len(metrics_list),
            "mse_mean": float(np.mean([m["mse"] for m in metrics_list])),
            "mse_std": float(np.std([m["mse"] for m in metrics_list])),
            "mae_mean": float(np.mean([m["mae"] for m in metrics_list])),
            "mae_std": float(np.std([m["mae"] for m in metrics_list])),
        }
        
        # 分部位指标
        for group_name in JOINT_GROUPS.keys():
            phase_agg[phase_name][f"mse_{group_name}"] = float(
                np.mean([m[f"mse_{group_name}"] for m in metrics_list])
            )
            phase_agg[phase_name][f"mae_{group_name}"] = float(
                np.mean([m[f"mae_{group_name}"] for m in metrics_list])
            )
    
    return phase_agg


def plot_phase_comparison(
    phase_agg: Dict,
    save_path: str,
):
    """
    绘制不同阶段的误差对比图
    """
    phases = ["early", "mid", "late"]
    phase_labels = ["Early (0-33%)", "Mid (33-67%)", "Late (67-100%)"]
    
    # 检查哪些阶段有数据
    available_phases = [p for p in phases if p in phase_agg]
    if len(available_phases) < 2:
        print("   ⚠️  阶段数据不足，跳过阶段对比图")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    
    # 1. 总体MSE对比
    ax = axes[0, 0]
    mse_values = [phase_agg[p]["mse_mean"] for p in available_phases]
    mse_stds = [phase_agg[p]["mse_std"] for p in available_phases]
    labels = [phase_labels[phases.index(p)] for p in available_phases]
    
    bars = ax.bar(labels, mse_values, yerr=mse_stds, capsize=5,
                  color=['#3498db', '#2ecc71', '#e74c3c'][:len(available_phases)], alpha=0.8)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE by Episode Phase', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 2. 总体MAE对比
    ax = axes[0, 1]
    mae_values = [phase_agg[p]["mae_mean"] for p in available_phases]
    mae_stds = [phase_agg[p]["mae_std"] for p in available_phases]
    
    bars = ax.bar(labels, mae_values, yerr=mae_stds, capsize=5,
                  color=['#3498db', '#2ecc71', '#e74c3c'][:len(available_phases)], alpha=0.8)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE by Episode Phase', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mae_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 分部位MSE对比（分组柱状图）
    ax = axes[1, 0]
    group_names = list(JOINT_GROUPS.keys())
    x = np.arange(len(group_names))
    width = 0.25
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, phase in enumerate(available_phases):
        values = [phase_agg[phase][f"mse_{g}"] for g in group_names]
        offset = (i - len(available_phases)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=phase_labels[phases.index(phase)],
               color=colors[i], alpha=0.8)
    
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE by Joint Group and Phase', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([JOINT_GROUP_NAMES_ZH[g] for g in group_names], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 样本数量统计
    ax = axes[1, 1]
    sample_counts = [phase_agg[p]["num_samples"] for p in available_phases]
    bars = ax.bar(labels, sample_counts,
                  color=['#3498db', '#2ecc71', '#e74c3c'][:len(available_phases)], alpha=0.8)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('Sample Distribution by Phase', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, sample_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=12)
    
    plt.suptitle('Error Analysis by Episode Phase', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ phase_comparison.png")


def plot_step_vs_error(
    all_metrics: List[Dict],
    save_path: str,
):
    """
    绘制step_idx vs MSE的散点图，分析误差与时间步的关系
    """
    step_indices = [m["step_idx"] for m in all_metrics]
    mse_values = [m["mse"] for m in all_metrics]
    
    # 分部位
    mse_right_arm = [m["mse_right_arm"] for m in all_metrics]
    mse_right_hand = [m["mse_right_hand"] for m in all_metrics]
    mse_left_arm = [m["mse_left_arm"] for m in all_metrics]
    mse_left_hand = [m["mse_left_hand"] for m in all_metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    
    # 总体MSE
    ax = axes[0, 0]
    ax.scatter(step_indices, mse_values, alpha=0.6, c='#3498db', s=30)
    # 添加趋势线
    z = np.polyfit(step_indices, mse_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(step_indices), max(step_indices), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.2e})')
    ax.set_xlabel('Step Index', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Total MSE vs Step Index', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 分部位散点图
    parts = [
        ("Right Arm", mse_right_arm, '#3498db'),
        ("Right Hand", mse_right_hand, '#2ecc71'),
        ("Left Arm", mse_left_arm, '#e74c3c'),
    ]
    
    for idx, (name, values, color) in enumerate(parts):
        ax = axes[(idx+1)//2, (idx+1)%2]
        ax.scatter(step_indices, values, alpha=0.6, c=color, s=30)
        z = np.polyfit(step_indices, values, 1)
        p = np.poly1d(z)
        ax.plot(x_line, p(x_line), 'k--', linewidth=2, label=f'Trend (slope={z[0]:.2e})')
        ax.set_xlabel('Step Index', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title(f'{name} MSE vs Step Index', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Error vs Time Step Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ step_vs_error.png")


# ============================================================================
# 可视化
# ============================================================================

# 详细关节名称（用于子图标题）
JOINT_NAMES_DETAILED = {
    "right_arm": ["R_Arm_J0 (肩1)", "R_Arm_J1 (肩2)", "R_Arm_J2 (肩3)", 
                  "R_Arm_J3 (肘)", "R_Arm_J4 (腕1)", "R_Arm_J5 (腕2)"],
    "right_hand": ["R_Hand_J0", "R_Hand_J1", "R_Hand_J2", "R_Hand_J3",
                   "R_Hand_J4", "R_Hand_J5", "R_Hand_J6", "R_Hand_J7",
                   "R_Hand_J8", "R_Hand_J9", "R_Hand_J10", "R_Hand_J11"],
    "left_arm": ["L_Arm_J0 (肩1)", "L_Arm_J1 (肩2)", "L_Arm_J2 (肩3)",
                 "L_Arm_J3 (肘)", "L_Arm_J4 (腕1)", "L_Arm_J5 (腕2)"],
    "left_hand": ["L_Hand_J0", "L_Hand_J1", "L_Hand_J2", "L_Hand_J3",
                  "L_Hand_J4", "L_Hand_J5", "L_Hand_J6", "L_Hand_J7",
                  "L_Hand_J8", "L_Hand_J9", "L_Hand_J10", "L_Hand_J11"],
}


def plot_detailed_joint_subplots(
    action_gt: np.ndarray,
    action_pred: np.ndarray,
    save_dir: str,
    episode_idx: int,
    step_idx: int,
):
    """
    为每个部位生成详细的关节子图
    
    生成4张图片：right_arm, right_hand, left_arm, left_hand
    每张图片里每个关节都有独立的子图
    """
    chunk_size = action_gt.shape[0]
    timesteps = np.arange(chunk_size)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for group_name, joint_indices in JOINT_GROUPS.items():
        num_joints = len(joint_indices)
        
        # 确定子图布局
        if num_joints == 6:  # arm
            nrows, ncols = 2, 3
            figsize = (15, 8)
        else:  # hand (12 joints)
            nrows, ncols = 3, 4
            figsize = (18, 12)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=100)
        axes = axes.flatten()
        
        joint_names = JOINT_NAMES_DETAILED[group_name]
        
        for i, (joint_idx, joint_name) in enumerate(zip(joint_indices, joint_names)):
            ax = axes[i]
            
            # 绘制GT和预测
            ax.plot(timesteps, action_gt[:, joint_idx], 'b-', linewidth=2, 
                   label='GT', alpha=0.8)
            ax.plot(timesteps, action_pred[:, joint_idx], 'r--', linewidth=2,
                   label='Pred', alpha=0.8)
            
            # 填充误差区域
            ax.fill_between(timesteps, action_gt[:, joint_idx], action_pred[:, joint_idx],
                           alpha=0.2, color='gray')
            
            # 计算该关节的误差
            mse = np.mean((action_gt[:, joint_idx] - action_pred[:, joint_idx]) ** 2)
            mae = np.mean(np.abs(action_gt[:, joint_idx] - action_pred[:, joint_idx]))
            
            ax.set_title(f"{joint_name}\nMSE: {mse:.4f}, MAE: {mae:.4f}", fontsize=10)
            ax.set_xlabel('Time Step', fontsize=9)
            ax.set_ylabel('Angle (rad)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            ax.tick_params(labelsize=8)
        
        # 隐藏多余的子图
        for i in range(num_joints, len(axes)):
            axes[i].set_visible(False)
        
        # 设置总标题
        title = f"{JOINT_GROUP_NAMES_ZH[group_name]} - Episode {episode_idx}, Step {step_idx}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = save_dir / f"ep{episode_idx:04d}_step{step_idx:04d}_{group_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def save_joint_angles_json(
    action_gt: np.ndarray,
    action_pred: np.ndarray,
    save_dir: str,
    episode_idx: int,
    step_idx: int,
):
    """
    保存关节角度到JSON文件
    
    包含GT和预测的完整数据，按部位分组
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_size = action_gt.shape[0]
    
    # 构建数据结构
    data = {
        "meta": {
            "episode_idx": int(episode_idx),
            "step_idx": int(step_idx),
            "chunk_size": int(chunk_size),
            "num_joints": 36,
        },
        "timesteps": list(range(chunk_size)),
        "joints": {}
    }
    
    # 按部位和关节保存数据
    for group_name, joint_indices in JOINT_GROUPS.items():
        data["joints"][group_name] = {}
        joint_names = JOINT_NAMES_DETAILED[group_name]
        
        for i, (joint_idx, joint_name) in enumerate(zip(joint_indices, joint_names)):
            gt_values = action_gt[:, joint_idx].tolist()
            pred_values = action_pred[:, joint_idx].tolist()
            error = (action_gt[:, joint_idx] - action_pred[:, joint_idx]).tolist()
            
            data["joints"][group_name][f"joint_{i}"] = {
                "name": joint_name,
                "global_index": int(joint_idx),
                "gt": gt_values,
                "pred": pred_values,
                "error": error,
                "mse": float(np.mean((action_gt[:, joint_idx] - action_pred[:, joint_idx]) ** 2)),
                "mae": float(np.mean(np.abs(action_gt[:, joint_idx] - action_pred[:, joint_idx]))),
            }
    
    # 添加汇总统计
    data["summary"] = {
        "total_mse": float(np.mean((action_gt - action_pred) ** 2)),
        "total_mae": float(np.mean(np.abs(action_gt - action_pred))),
    }
    for group_name, joint_indices in JOINT_GROUPS.items():
        group_gt = action_gt[:, joint_indices]
        group_pred = action_pred[:, joint_indices]
        data["summary"][f"{group_name}_mse"] = float(np.mean((group_gt - group_pred) ** 2))
        data["summary"][f"{group_name}_mae"] = float(np.mean(np.abs(group_gt - group_pred)))
    
    # 保存JSON
    save_path = save_dir / f"ep{episode_idx:04d}_step{step_idx:04d}_joints.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return save_path


def plot_sample_comparison(
    action_gt: np.ndarray,
    action_pred: np.ndarray,
    save_path: str,
    meta: dict = None,
):
    """绘制单个样本的对比图（简化版，所有关节在一张图）"""
    chunk_size, num_joints = action_gt.shape
    timesteps = np.arange(chunk_size)
    
    num_groups = len(JOINT_GROUPS)
    fig, axes = plt.subplots(num_groups, 1, figsize=(14, 3.5 * num_groups), dpi=100)
    
    if num_groups == 1:
        axes = [axes]
    
    colors_gt = plt.cm.Blues(np.linspace(0.4, 0.9, 12))
    colors_pred = plt.cm.Oranges(np.linspace(0.4, 0.9, 12))
    
    for ax_idx, (group_name, joint_indices) in enumerate(JOINT_GROUPS.items()):
        ax = axes[ax_idx]
        
        for i, joint_idx in enumerate(joint_indices):
            color_idx = i % len(colors_gt)
            
            ax.plot(
                timesteps, action_gt[:, joint_idx],
                color=colors_gt[color_idx], linestyle='-', linewidth=1.5,
                alpha=0.8
            )
            ax.plot(
                timesteps, action_pred[:, joint_idx],
                color=colors_pred[color_idx], linestyle='--', linewidth=1.5,
                alpha=0.8
            )
        
        ax.set_title(JOINT_GROUP_NAMES_ZH[group_name], fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Joint Angle (rad)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加图例说明
        ax.plot([], [], 'b-', linewidth=2, label='Ground Truth')
        ax.plot([], [], 'r--', linewidth=2, label='Prediction')
        ax.legend(loc='upper right', fontsize=9)
    
    title = "Action Prediction vs Ground Truth"
    if meta:
        title += f" (Episode {meta.get('episode_idx', 'N/A')}, Step {meta.get('step_idx', 'N/A')})"
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_error_bar(
    agg_metrics: Dict,
    save_path: str,
):
    """绘制分部位误差条形图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    
    groups = list(JOINT_GROUPS.keys())
    group_labels = [JOINT_GROUP_NAMES_ZH[g] for g in groups]
    
    # MSE
    ax = axes[0]
    mse_means = [agg_metrics[f"mse_{g}_mean"] for g in groups]
    mse_stds = [agg_metrics[f"mse_{g}_std"] for g in groups]
    
    bars = ax.bar(group_labels, mse_means, yerr=mse_stds, capsize=5, 
                  color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], alpha=0.8)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Mean Squared Error by Joint Group', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上标注数值
    for bar, mean in zip(bars, mse_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{mean:.4f}', ha='center', va='bottom', fontsize=10)
    
    # MAE
    ax = axes[1]
    mae_means = [agg_metrics[f"mae_{g}_mean"] for g in groups]
    mae_stds = [agg_metrics[f"mae_{g}_std"] for g in groups]
    
    bars = ax.bar(group_labels, mae_means, yerr=mae_stds, capsize=5,
                  color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], alpha=0.8)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Mean Absolute Error by Joint Group', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mean in zip(bars, mae_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{mean:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_joint_error(
    agg_metrics: Dict,
    save_path: str,
):
    """绘制每个关节的误差图"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=100)
    
    x = np.arange(36)
    width = 0.8
    
    # 颜色分组
    colors = []
    for group_name, indices in JOINT_GROUPS.items():
        if 'right_arm' in group_name:
            colors.extend(['#3498db'] * len(indices))
        elif 'right_hand' in group_name:
            colors.extend(['#2ecc71'] * len(indices))
        elif 'left_arm' in group_name:
            colors.extend(['#e74c3c'] * len(indices))
        else:
            colors.extend(['#f39c12'] * len(indices))
    
    # MSE per joint
    ax = axes[0]
    mse_means = agg_metrics["mse_per_joint_mean"]
    mse_stds = agg_metrics["mse_per_joint_std"]
    ax.bar(x, mse_means, width, yerr=mse_stds, capsize=2, color=colors, alpha=0.8)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE per Joint', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # MAE per joint
    ax = axes[1]
    mae_means = agg_metrics["mae_per_joint_mean"]
    mae_stds = agg_metrics["mae_per_joint_std"]
    ax.bar(x, mae_means, width, yerr=mae_stds, capsize=2, color=colors, alpha=0.8)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE per Joint', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#3498db', alpha=0.8, label='右臂 (Right Arm)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#2ecc71', alpha=0.8, label='右手 (Right Hand)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#e74c3c', alpha=0.8, label='左臂 (Left Arm)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#f39c12', alpha=0.8, label='左手 (Left Hand)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10, 
               bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_distribution(
    all_errors: List[np.ndarray],
    save_path: str,
):
    """绘制误差分布直方图"""
    all_errors_flat = np.concatenate([e.flatten() for e in all_errors])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    
    # 整体误差分布
    ax = axes[0]
    ax.hist(all_errors_flat, bins=100, density=True, alpha=0.7, color='#3498db')
    ax.axvline(np.mean(all_errors_flat), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(all_errors_flat):.4f}')
    ax.axvline(np.median(all_errors_flat), color='green', linestyle='--',
               linewidth=2, label=f'Median: {np.median(all_errors_flat):.4f}')
    ax.set_xlabel('Absolute Error', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Error Distribution (All Joints)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 分位数统计
    ax = axes[1]
    percentiles = [50, 75, 90, 95, 99]
    values = [np.percentile(all_errors_flat, p) for p in percentiles]
    bars = ax.bar([f'{p}%' for p in percentiles], values, 
                  color=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6'], alpha=0.8)
    ax.set_xlabel('Percentile', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Error Percentiles', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_episode_summary(
    episode_metrics: Dict[int, List[Dict]],
    save_path: str,
):
    """绘制每个episode的误差汇总图"""
    episode_ids = sorted(episode_metrics.keys())
    mse_per_episode = [np.mean([m["mse"] for m in episode_metrics[ep]]) for ep in episode_ids]
    mae_per_episode = [np.mean([m["mae"] for m in episode_metrics[ep]]) for ep in episode_ids]
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), dpi=100)
    
    # MSE per episode
    ax = axes[0]
    ax.bar(range(len(episode_ids)), mse_per_episode, alpha=0.7, color='#3498db')
    ax.axhline(np.mean(mse_per_episode), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(mse_per_episode):.4f}')
    ax.set_xlabel('Episode Index', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE per Episode', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # MAE per episode
    ax = axes[1]
    ax.bar(range(len(episode_ids)), mae_per_episode, alpha=0.7, color='#2ecc71')
    ax.axhline(np.mean(mae_per_episode), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(mae_per_episode):.4f}')
    ax.set_xlabel('Episode Index', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE per Episode', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RDT-1B 开环评估脚本")
    parser.add_argument(
        "--checkpoint", type=str, 
        default="./checkpoints/rdt1b-full-action176-20251202_000048/checkpoint-14000",
        help="Checkpoint路径"
    )
    parser.add_argument(
        "--dataset", type=str,
        default="./data/baai/data/lerobot_baai",
        help="数据集路径"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/base.yaml",
        help="配置文件路径（相对于项目根目录）"
    )
    parser.add_argument(
        "--vision_encoder", type=str,
        default="google/siglip-so400m-patch14-384",
        help="视觉编码器路径"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10,
        help="评估的episode数量，-1表示全部"
    )
    parser.add_argument(
        "--samples_per_episode", type=int, default=5,
        help="每个episode采样的次数"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=64,
        help="Action chunk大小"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results",
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
    parser.add_argument(
        "--save_samples", action="store_true",
        help="是否保存每个样本的对比图"
    )
    parser.add_argument(
        "--episode_list", type=str, default=None,
        help="指定要评估的episode列表，逗号分隔，例如: 0,5,10,15"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = Path(args.checkpoint).name
    output_dir = Path(args.output_dir) / f"eval_{ckpt_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_samples:
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🎯 RDT-1B 开环评估 (Open-Loop Evaluation)")
    print("=" * 70)
    print(f"📂 Checkpoint: {args.checkpoint}")
    print(f"📂 Dataset: {args.dataset}")
    print(f"📂 Output: {output_dir}")
    print(f"🔢 Episodes: {args.num_episodes} (-1 for all)")
    print(f"🔢 Samples per episode: {args.samples_per_episode}")
    print(f"🔢 Chunk size: {args.chunk_size}")
    print(f"🎲 Seed: {args.seed}")
    
    # 检查CUDA可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    
    # 保存配置
    config_save = vars(args).copy()
    config_save["timestamp"] = timestamp
    config_save["output_dir"] = str(output_dir)
    with open(output_dir / "eval_config.json", 'w') as f:
        json.dump(config_save, f, indent=2)
    
    # 初始化模型
    print("\n" + "=" * 70)
    print("🚀 初始化模型")
    print("=" * 70)
    
    model = BAAIEvalModel(
        checkpoint_path=args.checkpoint,
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
    
    # 获取episode列表
    cache_dir = dataset_path / "cache"
    
    if args.episode_list:
        episode_ids = [int(x.strip()) for x in args.episode_list.split(',')]
    else:
        # 扫描cache目录获取所有episode
        all_episodes = sorted([
            int(f.stem.split('_')[1]) 
            for f in cache_dir.glob("episode_*.pt")
            if f.stem != "episode_metadata"
        ])
        
        if args.num_episodes == -1 or args.num_episodes >= len(all_episodes):
            episode_ids = all_episodes
        else:
            episode_ids = random.sample(all_episodes, args.num_episodes)
            episode_ids.sort()
    
    print(f"\n📋 将评估 {len(episode_ids)} 个episodes: {episode_ids[:10]}{'...' if len(episode_ids) > 10 else ''}")
    
    # 开始评估
    print("\n" + "=" * 70)
    print("🔄 开始评估")
    print("=" * 70)
    
    all_metrics = []
    all_errors = []
    episode_metrics = {}
    phase_metrics = {"early": [], "mid": [], "late": []}  # 按阶段分类的指标
    
    for episode_idx in tqdm(episode_ids, desc="Evaluating episodes"):
        try:
            episode_cache = load_episode_cache(str(cache_dir), episode_idx)
        except Exception as e:
            print(f"\n⚠️  加载episode {episode_idx} 失败: {e}")
            continue
        
        num_steps = episode_cache["frame_num"]
        qpos = episode_cache["state"]
        
        # 找到运动起始点
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        first_idx = indices[0] if len(indices) > 0 else 1
        
        # 确定可采样范围
        max_valid_step = max(first_idx, num_steps - args.chunk_size - 1)
        
        if max_valid_step <= first_idx:
            print(f"\n⚠️  Episode {episode_idx} 步数不足，跳过")
            continue
        
        # 在该episode上采样 - 均匀分布在整个episode上
        sample_steps = np.linspace(first_idx, max_valid_step, args.samples_per_episode, dtype=int)
        sample_steps = np.unique(sample_steps)
        
        episode_metrics[episode_idx] = []
        
        for step_idx in sample_steps:
            try:
                sample = get_sample_from_episode(
                    episode_cache, episode_idx, step_idx, args.chunk_size
                )
                
                # 执行推理
                with torch.inference_mode():
                    action_pred = model.predict(
                        state_36=sample["state"],
                        images=sample["images"],
                        lang_embeds=lang_embeds,
                    )
                
                # 计算指标
                metrics = compute_metrics(sample["action_gt"], action_pred)
                metrics["episode_idx"] = episode_idx
                metrics["step_idx"] = step_idx
                metrics["num_steps"] = num_steps  # 记录episode总步数
                
                # 分类阶段
                phase = classify_phase(step_idx, num_steps)
                metrics["phase"] = phase
                
                all_metrics.append(metrics)
                episode_metrics[episode_idx].append(metrics)
                phase_metrics[phase].append(metrics)  # 按阶段收集
                
                # 收集误差用于分布图
                error = np.abs(sample["action_gt"] - action_pred)
                all_errors.append(error)
                
                # 保存样本对比图和JSON
                if args.save_samples:
                    # 1. 保存4张详细的分部位子图（right_arm, right_hand, left_arm, left_hand）
                    plot_detailed_joint_subplots(
                        sample["action_gt"], action_pred,
                        str(samples_dir),
                        episode_idx, step_idx
                    )
                    
                    # 2. 保存关节角JSON文件
                    save_joint_angles_json(
                        sample["action_gt"], action_pred,
                        str(samples_dir),
                        episode_idx, step_idx
                    )
                    
                    # 3. 保存简化版汇总图（可选，保留兼容性）
                    plot_sample_comparison(
                        sample["action_gt"], action_pred,
                        str(samples_dir / f"ep{episode_idx:04d}_step{step_idx:04d}_overview.png"),
                        meta=sample["meta"]
                    )
                    
            except Exception as e:
                print(f"\n⚠️  Episode {episode_idx} Step {step_idx} 推理失败: {e}")
                continue
    
    # 汇总统计
    print("\n" + "=" * 70)
    print("📊 汇总统计")
    print("=" * 70)
    
    if len(all_metrics) == 0:
        print("❌ 没有成功评估的样本！")
        return
    
    agg_metrics = aggregate_metrics(all_metrics)
    
    print(f"\n📈 总体指标 ({agg_metrics['num_samples']} 个样本):")
    print(f"   MSE:  {agg_metrics['mse_mean']:.6f} ± {agg_metrics['mse_std']:.6f}")
    print(f"   MAE:  {agg_metrics['mae_mean']:.6f} ± {agg_metrics['mae_std']:.6f}")
    print(f"   RMSE: {agg_metrics['rmse_mean']:.6f} ± {agg_metrics['rmse_std']:.6f}")
    
    print(f"\n📈 分部位MSE:")
    for group_name in JOINT_GROUPS.keys():
        mean = agg_metrics[f"mse_{group_name}_mean"]
        std = agg_metrics[f"mse_{group_name}_std"]
        print(f"   {JOINT_GROUP_NAMES_ZH[group_name]}: {mean:.6f} ± {std:.6f}")
    
    # 按阶段汇总
    phase_agg = aggregate_phase_metrics(phase_metrics)
    
    print(f"\n📈 分阶段MSE:")
    phase_names_zh = {"early": "初期 (0-33%)", "mid": "中期 (33-67%)", "late": "末期 (67-100%)"}
    for phase_name in ["early", "mid", "late"]:
        if phase_name in phase_agg:
            p = phase_agg[phase_name]
            print(f"   {phase_names_zh[phase_name]}: MSE={p['mse_mean']:.6f} ± {p['mse_std']:.6f} ({p['num_samples']} samples)")
    
    # 生成可视化图表
    print("\n" + "=" * 70)
    print("📊 生成可视化图表")
    print("=" * 70)
    
    # 分部位误差条形图
    plot_aggregate_error_bar(agg_metrics, str(output_dir / "error_by_group.png"))
    print("   ✅ error_by_group.png")
    
    # 每个关节的误差图
    plot_per_joint_error(agg_metrics, str(output_dir / "error_per_joint.png"))
    print("   ✅ error_per_joint.png")
    
    # 误差分布图
    plot_error_distribution(all_errors, str(output_dir / "error_distribution.png"))
    print("   ✅ error_distribution.png")
    
    # Episode汇总图
    if len(episode_metrics) > 1:
        plot_episode_summary(episode_metrics, str(output_dir / "error_per_episode.png"))
        print("   ✅ error_per_episode.png")
    
    # 阶段对比图
    if len(phase_agg) >= 2:
        plot_phase_comparison(phase_agg, str(output_dir / "phase_comparison.png"))
    
    # Step vs Error散点图
    if len(all_metrics) >= 10:
        plot_step_vs_error(all_metrics, str(output_dir / "step_vs_error.png"))
    
    # 保存详细结果
    results = {
        "config": config_save,
        "aggregate_metrics": agg_metrics,
        "phase_metrics": phase_agg,  # 添加阶段指标
        "all_metrics": all_metrics,
    }
    
    # 自定义JSON编码器处理numpy类型
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    # 保存JSON格式
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n📁 结果已保存: {output_dir / 'results.json'}")
    
    # 保存NPZ格式（包含numpy数组）
    np.savez(
        output_dir / "results.npz",
        aggregate_metrics=agg_metrics,
        mse_per_joint_mean=np.array(agg_metrics["mse_per_joint_mean"]),
        mae_per_joint_mean=np.array(agg_metrics["mae_per_joint_mean"]),
    )
    print(f"📁 结果已保存: {output_dir / 'results.npz'}")
    
    print("\n" + "=" * 70)
    print("✅ 开环评估完成!")
    print("=" * 70)
    print(f"\n📂 所有结果保存在: {output_dir}")


if __name__ == "__main__":
    main()

