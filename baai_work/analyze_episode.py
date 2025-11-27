#!/usr/bin/env python3
"""
高级数据集分析 - 深入探索episode的特征和模式
运行: conda activate rdt && python advanced_episode_analysis.py
"""
import os
from bson import decode_all
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
from PIL import Image
import json

episode_dir = "/home/chensiqi/chensiqi/RDT_libero_finetune/data/baai/data/action176/episode_0"

print("=" * 80)
print("Episode数据分析")
print("=" * 80)

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n1. 加载数据...")
with open(os.path.join(episode_dir, "xhand_control_data.bson"), "rb") as f:
    xhand_data = decode_all(f.read())[0]

with open(os.path.join(episode_dir, "episode_0.bson"), "rb") as f:
    episode_data = decode_all(f.read())[0]

frames = xhand_data["frames"]
print(f"   加载了 {len(frames)} 帧数据")

# ============================================================================
# 2. 时序特征分析
# ============================================================================
print("\n" + "=" * 80)
print("2. 时序特征分析")
print("=" * 80)

timestamps = np.array([f['t'] for f in frames])
time_diffs = np.diff(timestamps)

print(f"时间跨度: {timestamps[0]:.4f}s - {timestamps[-1]:.4f}s")
print(f"总时长: {timestamps[-1] - timestamps[0]:.4f}s")

# ============================================================================
# 3. Action空间分析
# ============================================================================
print("\n" + "=" * 80)
print("3. Action空间分析")
print("=" * 80)

left_actions = np.array([f['action']['left_hand'] for f in frames])
right_actions = np.array([f['action']['right_hand'] for f in frames])
all_actions = np.concatenate([left_actions, right_actions], axis=1)

print(f"Action维度: 左手{left_actions.shape[1]}维 + 右手{right_actions.shape[1]}维 = 总共{all_actions.shape[1]}维")
print(f"\n左手Action统计:")
for i in range(left_actions.shape[1]):
    print(f"  维度{i:2d}: [{left_actions[:, i].min():8.4f}, {left_actions[:, i].max():8.4f}] "
          )

print(f"\n右手Action统计:")
for i in range(right_actions.shape[1]):
    print(f"  维度{i:2d}: [{right_actions[:, i].min():8.4f}, {right_actions[:, i].max():8.4f}] "
          )


# ============================================================================
# 4. Observation空间分析
# ============================================================================
print("\n" + "=" * 80)
print("4. Observation空间分析")
print("=" * 80)

left_obs = np.array([f['observation']['left_hand'] for f in frames])
right_obs = np.array([f['observation']['right_hand'] for f in frames])

print(f"Observation维度: 左手{left_obs.shape[1]}维 + 右手{right_obs.shape[1]}维")
print(f"\n左手Observation统计:")
for i in range(left_obs.shape[1]):
    print(f"  维度{i:2d}: [{left_obs[:, i].min():8.4f}, {left_obs[:, i].max():8.4f}] "
          )

print(f"\n右手Observation统计:")
for i in range(right_obs.shape[1]):
    print(f"  维度{i:2d}: [{right_obs[:, i].min():8.4f}, {right_obs[:, i].max():8.4f}] "
          )

print("\n" + "=" * 80)
print("5. 图像数据分析")
print("=" * 80)

import glob
camera_views = ['camera_head', 'camera_left_wrist', 'camera_right_wrist', 'camera_third_view']
image_info = {}

for cam_view in camera_views:
    cam_path = os.path.join(episode_dir, cam_view)
    if os.path.exists(cam_path):
        images = sorted(glob.glob(os.path.join(cam_path, "*.jpg")))
        if len(images) > 0:
            # 读取第一张图像获取尺寸
            img = Image.open(images[0])
            image_info[cam_view] = {
                'count': len(images),
                'resolution': img.size,
                'first_frame': int(images[0].split('_')[-1].split('.')[0]),
                'last_frame': int(images[-1].split('_')[-1].split('.')[0])
            }
            print(f"{cam_view}:")
            print(f"  图像数量: {image_info[cam_view]['count']}")
            print(f"  分辨率: {image_info[cam_view]['resolution']}")
            print(f"  帧范围: {image_info[cam_view]['first_frame']} - {image_info[cam_view]['last_frame']}")

