#!/usr/bin/env python3
"""
RDT-1B 推理对比脚本

功能：
1. 从lerobot_baai数据集随机选取一个episode的某个frame
2. 使用训练好的checkpoint进行推理
3. 对比预测的action chunk与真实的action chunk
4. 绘制关节角对比图

Author: AI Assistant
"""

import os
import sys
import json
import yaml
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# 关节名称（用于绘图）
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


# ============================================================================
# 模型加载
# ============================================================================

class BAAIInferenceModel:
    """用于推理的RDT模型封装类"""
    
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
        
        print("=" * 70)
        print("🚀 初始化推理模型")
        print("=" * 70)
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"📂 Checkpoint: {checkpoint_path}")
        print(f"📂 Config: {config_path}")
        print(f"📂 Vision Encoder: {vision_encoder_path}")
        
        # 加载视觉编码器
        print("\n🔄 加载视觉编码器...")
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
        print("\n🔄 创建RDT模型...")
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
        print("\n🔄 加载checkpoint权重...")
        self._load_checkpoint(checkpoint_path)
        
        # 移动到设备并设置为评估模式
        self.policy = self.policy.to(device, dtype=dtype)
        self.policy.eval()
        
        print("\n✅ 模型初始化完成！")
        print("=" * 70)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint权重"""
        checkpoint_file = Path(checkpoint_path) / "pytorch_model.bin"
        
        if not checkpoint_file.exists():
            # 尝试加载零碎的DeepSpeed checkpoint
            print(f"   ⚠️  未找到 pytorch_model.bin，尝试从DeepSpeed格式加载...")
            zero_to_fp32_path = Path(checkpoint_path) / "zero_to_fp32.py"
            if zero_to_fp32_path.exists():
                raise NotImplementedError(
                    f"请先运行 python {zero_to_fp32_path} {checkpoint_path} {checkpoint_path}/pytorch_model.bin 来转换权重"
                )
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
        state_128 = np.zeros((B, self.config["common"]["state_dim"]))
        state_128[:, BAAI_STATE_INDICES] = state_36
        return state_128
    
    def _unformat_unified_to_state(self, action_128: np.ndarray) -> np.ndarray:
        """将128维统一向量映射回36维动作向量"""
        if action_128.ndim == 2:
            return action_128[:, BAAI_STATE_INDICES]
        elif action_128.ndim == 3:
            return action_128[:, :, BAAI_STATE_INDICES]
        return action_128[BAAI_STATE_INDICES]
    
    def preprocess_images(self, images: list) -> torch.Tensor:
        """
        预处理图像列表
        
        Args:
            images: 图像列表，顺序为 [head_t-1, right_wrist_t-1, left_wrist_t-1, 
                                      head_t, right_wrist_t, left_wrist_t]
                    共6张图像 (2个时间步 x 3个相机)
        
        Returns:
            torch.Tensor: 编码后的图像特征 (1, num_patches*6, hidden_size)
        """
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
            
            # 使用image_processor处理
            processed = self.image_processor.preprocess(img, return_tensors='pt')
            image_tensors.append(processed['pixel_values'][0])
        
        # Stack and encode
        image_tensor = torch.stack(image_tensors, dim=0).to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            image_embeds = self.vision_encoder(image_tensor)
            image_embeds = image_embeds.reshape(-1, self.vision_encoder.hidden_size)
            image_embeds = image_embeds.unsqueeze(0)  # (1, N, hidden_size)
        
        return image_embeds
    
    @torch.no_grad()
    def predict(
        self,
        state_36: np.ndarray,
        images: list,
        lang_embeds: torch.Tensor,
    ) -> np.ndarray:
        """
        执行推理，预测action chunk
        
        Args:
            state_36: 当前状态 (36,)
            images: 图像列表 [6张图像]
            lang_embeds: 语言嵌入 (seq_len, embed_dim)
        
        Returns:
            predicted_actions: (chunk_size, 36) 预测的动作序列
        """
        # 准备状态
        state_128 = self._format_state_to_unified(state_36)
        state_tensor = torch.from_numpy(state_128).to(self.device, dtype=self.dtype)
        state_tensor = state_tensor.unsqueeze(1)  # (1, 1, 128)
        
        # 准备状态mask
        state_mask = np.zeros(self.config["common"]["state_dim"])
        state_mask[BAAI_STATE_INDICES] = 1
        state_mask_tensor = torch.from_numpy(state_mask).to(self.device, dtype=self.dtype)
        state_mask_tensor = state_mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 128)
        
        # 编码图像
        image_embeds = self.preprocess_images(images)
        
        # 准备语言条件
        if lang_embeds.ndim == 2:
            lang_embeds = lang_embeds.unsqueeze(0)  # (1, seq_len, embed_dim)
        lang_embeds = lang_embeds.to(self.device, dtype=self.dtype)
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
        
        # 转换回numpy并提取36维
        predicted_actions = predicted_actions.squeeze(0).cpu().numpy()  # (chunk_size, 128)
        predicted_actions_36 = self._unformat_unified_to_state(predicted_actions)  # (chunk_size, 36)
        
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
    import sys
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.numeric'] = getattr(np.core, 'numeric', np.core)
    
    return torch.load(cache_file, map_location='cpu', weights_only=False)


def get_sample_from_dataset(
    dataset_path: str,
    episode_idx: int = None,
    step_idx: int = None,
    chunk_size: int = 64,
    img_history_size: int = 2,
) -> dict:
    """
    从数据集获取一个样本用于推理
    
    Args:
        dataset_path: 数据集路径
        episode_idx: Episode索引，None表示随机选择
        step_idx: 步数索引，None表示随机选择
        chunk_size: Action chunk大小
        img_history_size: 图像历史大小
    
    Returns:
        dict: 包含 state, action_gt, images, lang_embeds, meta 的字典
    """
    cache_dir = Path(dataset_path) / "cache"
    
    # 加载元数据
    meta_file = cache_dir / "episode_metadata.pt"
    import sys
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.numeric'] = getattr(np.core, 'numeric', np.core)
    
    cache_data = torch.load(meta_file, map_location='cpu', weights_only=False)
    episode_data = cache_data['episode_data']
    episode_lens = cache_data['episode_lens']
    
    # 选择episode
    if episode_idx is None:
        weights = np.array(episode_lens) / np.sum(episode_lens)
        episode_idx = np.random.choice(len(episode_data), p=weights)
    
    episode_info = episode_data[episode_idx]
    actual_episode_idx = episode_info['episode_idx']
    
    print(f"\n📂 选择 Episode {actual_episode_idx} (内部索引: {episode_idx})")
    
    # 加载episode缓存
    episode_cache = load_episode_cache(str(cache_dir), actual_episode_idx)
    
    qpos = episode_cache["state"]  # (T, 36)
    actions = episode_cache["action"]  # (T, 36)
    num_steps = episode_cache["frame_num"]
    images_info = episode_cache.get("images_info", {})
    
    print(f"   总步数: {num_steps}")
    
    # 找到运动起始点
    EPS = 1e-2
    qpos_delta = np.abs(qpos - qpos[0:1])
    indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
    first_idx = indices[0] if len(indices) > 0 else 1
    
    print(f"   运动起始索引: {first_idx}")
    
    # 选择步数
    max_valid_step = min(num_steps - 1, num_steps - chunk_size)
    if step_idx is None:
        step_idx = random.randint(first_idx, max(first_idx, max_valid_step))
    step_idx = min(step_idx, max_valid_step)
    
    print(f"   选择步数索引: {step_idx}")
    
    # 获取状态和动作
    state = qpos[step_idx]  # (36,)
    action_gt = actions[step_idx:step_idx + chunk_size]  # (chunk_size, 36)
    
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
    # 顺序: [head_t-1, right_t-1, left_t-1, head_t, right_t, left_t]
    images = []
    for t_offset in [-1, 0]:
        t = max(0, step_idx + t_offset)
        images.append(load_image('camera_head', t))
        images.append(load_image('camera_right_wrist', t))
        images.append(load_image('camera_left_wrist', t))
    
    # 加载语言嵌入
    lang_embed_path = Path(dataset_path) / "instruction.pt"
    if lang_embed_path.exists():
        lang_embeds = torch.load(lang_embed_path, map_location='cpu')
        print(f"   语言嵌入: {lang_embeds.shape}")
    else:
        raise FileNotFoundError(f"未找到语言嵌入文件: {lang_embed_path}")
    
    return {
        "state": state,
        "action_gt": action_gt,
        "images": images,
        "lang_embeds": lang_embeds,
        "meta": {
            "episode_idx": actual_episode_idx,
            "step_idx": step_idx,
            "num_steps": num_steps,
        }
    }


# ============================================================================
# 可视化
# ============================================================================

def plot_action_comparison(
    action_gt: np.ndarray,
    action_pred: np.ndarray,
    save_path: str = "action_comparison.png",
    title: str = "Action Prediction vs Ground Truth",
    joint_groups: dict = None,
):
    """
    绘制预测动作和真实动作的对比图
    
    Args:
        action_gt: 真实动作 (chunk_size, 36)
        action_pred: 预测动作 (chunk_size, 36)
        save_path: 保存路径
        title: 图表标题
        joint_groups: 关节分组字典，用于分组显示
    """
    chunk_size, num_joints = action_gt.shape
    timesteps = np.arange(chunk_size)
    
    # 定义关节分组
    if joint_groups is None:
        joint_groups = {
            "右臂关节 (Right Arm)": list(range(0, 6)),
            "右手关节 (Right Hand)": list(range(6, 18)),
            "左臂关节 (Left Arm)": list(range(18, 24)),
            "左手关节 (Left Hand)": list(range(24, 36)),
        }
    
    num_groups = len(joint_groups)
    fig, axes = plt.subplots(num_groups, 1, figsize=(16, 4 * num_groups), dpi=100)
    
    if num_groups == 1:
        axes = [axes]
    
    # 使用更好看的颜色
    colors_gt = plt.cm.Blues(np.linspace(0.4, 0.9, 12))
    colors_pred = plt.cm.Oranges(np.linspace(0.4, 0.9, 12))
    
    for ax_idx, (group_name, joint_indices) in enumerate(joint_groups.items()):
        ax = axes[ax_idx]
        
        for i, joint_idx in enumerate(joint_indices):
            color_idx = i % len(colors_gt)
            
            # 绘制真实值
            ax.plot(
                timesteps, action_gt[:, joint_idx],
                color=colors_gt[color_idx], linestyle='-', linewidth=2,
                label=f"{JOINT_NAMES[joint_idx]} (GT)" if i < 6 else None,
                alpha=0.8
            )
            
            # 绘制预测值
            ax.plot(
                timesteps, action_pred[:, joint_idx],
                color=colors_pred[color_idx], linestyle='--', linewidth=2,
                label=f"{JOINT_NAMES[joint_idx]} (Pred)" if i < 6 else None,
                alpha=0.8
            )
        
        ax.set_title(group_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Joint Angle (rad)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 对比图已保存: {save_path}")


def plot_detailed_comparison(
    action_gt: np.ndarray,
    action_pred: np.ndarray,
    save_path: str = "action_comparison_detailed.png",
    meta: dict = None,
):
    """
    绘制每个关节的详细对比图（子图形式）
    
    Args:
        action_gt: 真实动作 (chunk_size, 36)
        action_pred: 预测动作 (chunk_size, 36)
        save_path: 保存路径
        meta: 元数据信息
    """
    chunk_size, num_joints = action_gt.shape
    timesteps = np.arange(chunk_size)
    
    # 创建6x6的子图网格
    fig, axes = plt.subplots(6, 6, figsize=(24, 20), dpi=100)
    axes = axes.flatten()
    
    # 计算误差统计
    mse = np.mean((action_gt - action_pred) ** 2, axis=0)
    mae = np.mean(np.abs(action_gt - action_pred), axis=0)
    
    for joint_idx in range(num_joints):
        ax = axes[joint_idx]
        
        # 绘制真实值和预测值
        ax.plot(timesteps, action_gt[:, joint_idx], 'b-', linewidth=2, label='GT', alpha=0.8)
        ax.plot(timesteps, action_pred[:, joint_idx], 'r--', linewidth=2, label='Pred', alpha=0.8)
        
        # 填充误差区域
        ax.fill_between(
            timesteps,
            action_gt[:, joint_idx],
            action_pred[:, joint_idx],
            alpha=0.2, color='gray'
        )
        
        ax.set_title(f"{JOINT_NAMES[joint_idx]}\nMSE: {mse[joint_idx]:.4f}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
    
    # 添加总体标题
    title = "Action Prediction vs Ground Truth (All 36 Joints)"
    if meta:
        title += f"\nEpisode: {meta.get('episode_idx', 'N/A')}, Step: {meta.get('step_idx', 'N/A')}"
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 详细对比图已保存: {save_path}")
    
    # 打印误差统计
    print("\n📈 误差统计:")
    print(f"   总体MSE: {np.mean(mse):.6f}")
    print(f"   总体MAE: {np.mean(mae):.6f}")
    print(f"   右臂MSE: {np.mean(mse[:6]):.6f}")
    print(f"   右手MSE: {np.mean(mse[6:18]):.6f}")
    print(f"   左臂MSE: {np.mean(mse[18:24]):.6f}")
    print(f"   左手MSE: {np.mean(mse[24:36]):.6f}")


def plot_error_heatmap(
    action_gt: np.ndarray,
    action_pred: np.ndarray,
    save_path: str = "action_error_heatmap.png",
    meta: dict = None,
):
    """
    绘制误差热力图
    
    Args:
        action_gt: 真实动作 (chunk_size, 36)
        action_pred: 预测动作 (chunk_size, 36)
        save_path: 保存路径
        meta: 元数据信息
    """
    error = np.abs(action_gt - action_pred)
    
    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
    
    im = ax.imshow(error.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Joint Index', fontsize=12)
    
    # 设置y轴标签
    ax.set_yticks(range(len(JOINT_NAMES)))
    ax.set_yticklabels(JOINT_NAMES, fontsize=8)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Absolute Error (rad)', fontsize=12)
    
    title = "Prediction Error Heatmap"
    if meta:
        title += f"\nEpisode: {meta.get('episode_idx', 'N/A')}, Step: {meta.get('step_idx', 'N/A')}"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 误差热力图已保存: {save_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RDT-1B 推理对比脚本")
    parser.add_argument(
        "--checkpoint", type=str, 
        default="./checkpoints/rdt1b-full-action176-20251202_000048/checkpoint-6000",
        help="Checkpoint路径"
    )
    parser.add_argument(
        "--dataset", type=str,
        default="./data/baai/data/lerobot_baai",
        help="数据集路径"
    )
    parser.add_argument(
        "--config", type=str,
        default="./configs/base.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--vision_encoder", type=str,
        default="google/siglip-so400m-patch14-384",
        help="视觉编码器路径"
    )
    parser.add_argument(
        "--episode_idx", type=int, default=None,
        help="Episode索引，默认随机选择"
    )
    parser.add_argument(
        "--step_idx", type=int, default=None,
        help="步数索引，默认随机选择"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./inference_results",
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
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🎯 RDT-1B 推理对比")
    print("=" * 70)
    print(f"📂 Checkpoint: {args.checkpoint}")
    print(f"📂 Dataset: {args.dataset}")
    print(f"📂 Output: {args.output_dir}")
    print(f"🎲 Seed: {args.seed}")
    
    # 检查CUDA可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    
    # 初始化模型
    model = BAAIInferenceModel(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        vision_encoder_path=args.vision_encoder,
        device=args.device,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        control_frequency=20,  # BAAI数据集的控制频率
    )
    
    # 获取样本数据
    print("\n" + "=" * 70)
    print("📦 加载样本数据")
    print("=" * 70)
    
    sample = get_sample_from_dataset(
        dataset_path=args.dataset,
        episode_idx=args.episode_idx,
        step_idx=args.step_idx,
        chunk_size=64,
        img_history_size=2,
    )
    
    state = sample["state"]
    action_gt = sample["action_gt"]
    images = sample["images"]
    lang_embeds = sample["lang_embeds"]
    meta = sample["meta"]
    
    print(f"\n📊 样本信息:")
    print(f"   Episode: {meta['episode_idx']}")
    print(f"   Step: {meta['step_idx']}")
    print(f"   State shape: {state.shape}")
    print(f"   Action GT shape: {action_gt.shape}")
    print(f"   Images count: {len(images)}")
    print(f"   Lang embeds shape: {lang_embeds.shape}")
    
    # 执行推理
    print("\n" + "=" * 70)
    print("🔄 执行推理")
    print("=" * 70)
    
    with torch.inference_mode():
        action_pred = model.predict(
            state_36=state,
            images=images,
            lang_embeds=lang_embeds,
        )
    
    print(f"✅ 推理完成! 预测动作shape: {action_pred.shape}")
    
    # 绘制对比图
    print("\n" + "=" * 70)
    print("📊 生成对比图")
    print("=" * 70)
    
    # 1. 分组对比图
    plot_action_comparison(
        action_gt=action_gt,
        action_pred=action_pred,
        save_path=str(output_dir / f"comparison_ep{meta['episode_idx']}_step{meta['step_idx']}.png"),
        title=f"Episode {meta['episode_idx']}, Step {meta['step_idx']}",
    )
    
    # 2. 详细对比图（所有36个关节）
    plot_detailed_comparison(
        action_gt=action_gt,
        action_pred=action_pred,
        save_path=str(output_dir / f"comparison_detailed_ep{meta['episode_idx']}_step{meta['step_idx']}.png"),
        meta=meta,
    )
    
    # 3. 误差热力图
    plot_error_heatmap(
        action_gt=action_gt,
        action_pred=action_pred,
        save_path=str(output_dir / f"error_heatmap_ep{meta['episode_idx']}_step{meta['step_idx']}.png"),
        meta=meta,
    )
    
    # 保存数值结果
    results = {
        "meta": meta,
        "action_gt": action_gt,
        "action_pred": action_pred,
        "mse_per_joint": np.mean((action_gt - action_pred) ** 2, axis=0).tolist(),
        "mae_per_joint": np.mean(np.abs(action_gt - action_pred), axis=0).tolist(),
        "total_mse": float(np.mean((action_gt - action_pred) ** 2)),
        "total_mae": float(np.mean(np.abs(action_gt - action_pred))),
    }
    
    results_path = output_dir / f"results_ep{meta['episode_idx']}_step{meta['step_idx']}.npz"
    np.savez(results_path, **results)
    print(f"\n📁 数值结果已保存: {results_path}")
    
    print("\n" + "=" * 70)
    print("✅ 推理对比完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

