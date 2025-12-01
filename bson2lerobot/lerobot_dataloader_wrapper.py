#!/usr/bin/env python3
"""LeRobot数据集加载器 - 将LeRobot格式映射到自定义obs/action格式"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lerobot' / 'src'))
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotDatasetWrapper:
    """
    将LeRobot数据集包装成自定义格式
    
    输出格式:
    {
        'obs': {
            # 图像观测
            'camera_head_img': np.ndarray,
            'camera_left_wrist_img': np.ndarray,
            'camera_right_wrist_img': np.ndarray,
            'camera_third_view_img': np.ndarray,
            # 左臂关节状态
            'left_arm_joint_pos': np.ndarray,
            'left_arm_joint_vel': np.ndarray,
            'left_arm_joint_eff': np.ndarray,
            # 右臂关节状态
            'right_arm_joint_pos': np.ndarray,
            'right_arm_joint_vel': np.ndarray,
            'right_arm_joint_eff': np.ndarray,
            # 灵巧手观测
            'left_hand_obs': np.ndarray,
            'right_hand_obs': np.ndarray,
        },
        'action': np.ndarray
    }
    """
    
    def __init__(self, lerobot_dir, repo_id="baai/bimanual_dexhand", image_format='CHW'):
        """
        Args:
            lerobot_dir: LeRobot数据集目录
            repo_id: 数据集ID
            image_format: 图像格式 'CHW' (C, H, W) 或 'HWC' (H, W, C)
        """
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=str(lerobot_dir)
        )
        self.image_format = image_format
        
        # 打印数据集信息
        print(f"加载LeRobot数据集: {lerobot_dir}")
        print(f"  总帧数: {len(self.dataset)}")
        print(f"  Episodes: {self.dataset.num_episodes}")
        print(f"  FPS: {self.dataset.fps}")
        print(f"  图像格式: {image_format}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        获取单帧数据，转换为自定义格式
        
        Args:
            idx: 帧索引
            
        Returns:
            dict: 包含obs和action的字典
        """
        # 从LeRobot获取原始数据
        sample = self.dataset[idx]
        
        # 构建自定义格式
        custom_data = {
            'obs': {
                # 图像观测 - 注意LeRobot可能返回tensor，需要转numpy
                'camera_head_img': self._process_image(sample['observation.images.camera_head']),
                'camera_left_wrist_img': self._process_image(sample['observation.images.camera_left_wrist']),
                'camera_right_wrist_img': self._process_image(sample['observation.images.camera_right_wrist']),
                'camera_third_view_img': self._process_image(sample['observation.images.camera_third_view']),
                
                # 左臂关节状态
                'left_arm_joint_pos': self._to_numpy(sample['observation.state.left_arm_joint_pos']),
                'left_arm_joint_vel': self._to_numpy(sample['observation.state.left_arm_joint_vel']),
                'left_arm_joint_eff': self._to_numpy(sample['observation.state.left_arm_joint_eff']),
                
                # 右臂关节状态
                'right_arm_joint_pos': self._to_numpy(sample['observation.state.right_arm_joint_pos']),
                'right_arm_joint_vel': self._to_numpy(sample['observation.state.right_arm_joint_vel']),
                'right_arm_joint_eff': self._to_numpy(sample['observation.state.right_arm_joint_eff']),
                
                # 灵巧手观测
                'left_hand_obs': self._to_numpy(sample['observation.state.left_hand_obs']),
                'right_hand_obs': self._to_numpy(sample['observation.state.right_hand_obs']),
            },
            'action': self._to_numpy(sample['action'])
        }
        
        # 可选：添加额外信息
        if 'task' in sample:
            custom_data['task'] = sample['task']
        if 'episode_index' in sample:
            custom_data['episode_index'] = sample['episode_index']
        if 'frame_index' in sample:
            custom_data['frame_index'] = sample['frame_index']
        
        return custom_data
    
    def _to_numpy(self, data):
        """将tensor转换为numpy数组"""
        if hasattr(data, 'numpy'):
            return data.numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)
    
    def _process_image(self, img):
        """
        处理图像：转换为numpy并调整格式
        LeRobot默认格式: (C, H, W) 范围[0, 1]
        """
        img = self._to_numpy(img)
        
        # 根据设置转换格式
        if self.image_format == 'HWC' and img.shape[0] == 3:
            # 从 (C, H, W) 转换为 (H, W, C)
            img = np.transpose(img, (1, 2, 0))
        
        return img
    
    def get_episode(self, episode_idx):
        """
        获取整个episode的数据
        
        Args:
            episode_idx: episode索引
            
        Returns:
            list: episode的所有帧数据
        """
        # 通过episode_index筛选帧
        frames = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            if sample['episode_index'].item() == episode_idx:
                frames.append(self[i])
        
        return frames
    
    @property
    def num_episodes(self):
        return self.dataset.num_episodes
    
    @property
    def fps(self):
        return self.dataset.fps


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试LeRobot数据加载器")
    parser.add_argument("--lerobot-dir", default="/home/chensiqi/chensiqi/RDT_baai/data/baai/lerobot_test", 
                       help="LeRobot数据集目录")
    parser.add_argument("--image-format", default="HWC", choices=["CHW", "HWC"],
                       help="图像格式: CHW (C,H,W) 或 HWC (H,W,C)")
    args = parser.parse_args()
    
    # 创建加载器
    print("="*70)
    print(" 测试LeRobot数据加载器")
    print("="*70)
    
    loader = LeRobotDatasetWrapper(args.lerobot_dir, image_format=args.image_format)
    
    # 测试单帧加载
    print("\n测试单帧加载:")
    print("-"*70)
    sample = loader[0]
    
    print("\nobs键:")
    for key, value in sample['obs'].items():
        if isinstance(value, np.ndarray):
            print(f"  {key:30s}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key:30s}: {type(value)}")
    
    print(f"\naction: shape={sample['action'].shape}, dtype={sample['action'].dtype}")
    
    if 'task' in sample:
        print(f"task: {sample['task']}")
    
    # 测试图像可视化
    print("\n" + "="*70)
    print(" 图像数据验证")
    print("="*70)
    
    for cam in ['camera_head_img', 'camera_left_wrist_img', 
                'camera_right_wrist_img', 'camera_third_view_img']:
        img = sample['obs'][cam]
        print(f"\n{cam}:")
        print(f"  形状: {img.shape}")
        print(f"  数值范围: [{img.min()}, {img.max()}]")
        print(f"  数据类型: {img.dtype}")
    
    # 测试episode加载
    print("\n" + "="*70)
    print(" 测试Episode加载")
    print("="*70)
    
    if loader.num_episodes > 0:
        print(f"\n加载第一个episode (共{loader.num_episodes}个)...")
        episode_frames = loader.get_episode(0)
        print(f"Episode帧数: {len(episode_frames)}")
        print(f"第一帧action: {episode_frames[0]['action'][:6]}...")  # 只显示前6个
        print(f"最后一帧action: {episode_frames[-1]['action'][:6]}...")
    
    # 测试数据统计
    print("\n" + "="*70)
    print(" 数据统计")
    print("="*70)
    
    print("\n采样前10帧，统计数值范围:")
    n_samples = min(10, len(loader))
    
    stats = {
        'left_arm_joint_pos': {'min': [], 'max': []},
        'right_arm_joint_pos': {'min': [], 'max': []},
        'left_hand_obs': {'min': [], 'max': []},
        'right_hand_obs': {'min': [], 'max': []},
        'action': {'min': [], 'max': []},
    }
    
    for i in range(n_samples):
        sample = loader[i]
        for key in stats.keys():
            if key in sample['obs']:
                data = sample['obs'][key]
            else:
                data = sample[key]
            stats[key]['min'].append(data.min())
            stats[key]['max'].append(data.max())
    
    for key, ranges in stats.items():
        min_val = min(ranges['min'])
        max_val = max(ranges['max'])
        print(f"  {key:30s}: [{min_val:8.3f}, {max_val:8.3f}]")
    
    print("\n✅ 测试完成!")
    print("="*70)

