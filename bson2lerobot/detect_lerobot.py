#!/usr/bin/env python3
"""LeRobot数据集 - Episode级别加载器"""

import sys
import numpy as np
from pathlib import Path
# 添加lerobot路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lerobot' / 'src'))
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class EpisodeBasedDataset:
    """
    Episode级别的数据集访问
    
    用法：
        dataset[0]  -> 返回Episode 0的所有帧（列表）
        dataset[1]  -> 返回Episode 1的所有帧（列表）
        len(dataset) -> Episode数量（不是帧数！）
    """
    
    def __init__(self, lerobot_dir, repo_id="baai/bimanual_dexhand", image_format='HWC'):
        """
        Args:
            lerobot_dir: LeRobot数据集目录
            image_format: 图像格式 'HWC' 或 'CHW'
        """
        self.lerobot_dataset = LeRobotDataset(
            repo_id=repo_id,
            root=str(lerobot_dir)
        )
        self.image_format = image_format
        
        # 预先构建episode索引
        self._build_episode_index()
        
        print(f"加载LeRobot数据集: {lerobot_dir}")
        print(f"  Episodes数: {len(self.episodes)}")
        print(f"  总帧数: {len(self.lerobot_dataset)}")
        print(f"  图像格式: {image_format}")
        for i, ep_info in enumerate(self.episodes):
            print(f"    Episode {i}: {ep_info['count']} 帧")
    
    def _build_episode_index(self):
        """构建episode索引"""
        self.episodes = []
        episode_frames = {}
        
        # 遍历所有帧，按episode分组
        for i in range(len(self.lerobot_dataset)):
            ep_idx = self.lerobot_dataset[i]['episode_index'].item()
            if ep_idx not in episode_frames:
                episode_frames[ep_idx] = []
            episode_frames[ep_idx].append(i)
        
        # 按episode索引排序
        for ep_idx in sorted(episode_frames.keys()):
            self.episodes.append({
                'episode_idx': ep_idx,
                'frame_indices': episode_frames[ep_idx],
                'count': len(episode_frames[ep_idx])
            })
    
    def __len__(self):
        """返回episode数量（不是帧数！）"""
        return len(self.episodes)
    
    def __getitem__(self, episode_idx):
        """
        获取一个episode的所有帧
        
        Args:
            episode_idx: Episode索引 (0, 1, 2, ...)
            
        Returns:
            dict: {
                'obs': {
                    'camera_head_img': np.ndarray (T, H, W, C),
                    'camera_left_wrist_img': np.ndarray (T, H, W, C),
                    'camera_right_wrist_img': np.ndarray (T, H, W, C),
                    'camera_third_view_img': np.ndarray (T, H, W, C),
                    'left_arm_joint_pos': np.ndarray (T, 6),
                    'left_arm_joint_vel': np.ndarray (T, 6),
                    'left_arm_joint_eff': np.ndarray (T, 6),
                    'right_arm_joint_pos': np.ndarray (T, 6),
                    'right_arm_joint_vel': np.ndarray (T, 6),
                    'right_arm_joint_eff': np.ndarray (T, 6),
                    'left_hand_obs': np.ndarray (T, 12),
                    'right_hand_obs': np.ndarray (T, 12),
                },
                'action': np.ndarray (T, 36),
                'episode_idx': int,
                'num_frames': int,
            }
        """
        if episode_idx >= len(self.episodes):
            raise IndexError(f"Episode index {episode_idx} out of range [0, {len(self.episodes)})")
        
        ep_info = self.episodes[episode_idx]
        frame_indices = ep_info['frame_indices']
        T = len(frame_indices)
        
        # 初始化存储
        episode_data = {
            'obs': {
                'camera_head_img': [],
                'camera_left_wrist_img': [],
                'camera_right_wrist_img': [],
                'camera_third_view_img': [],
                'left_arm_joint_pos': [],
                'left_arm_joint_vel': [],
                'left_arm_joint_eff': [],
                'right_arm_joint_pos': [],
                'right_arm_joint_vel': [],
                'right_arm_joint_eff': [],
                'left_hand_obs': [],
                'right_hand_obs': [],
            },
            'action': [],
        }
        
        # 收集所有帧
        for frame_idx in frame_indices:
            sample = self.lerobot_dataset[frame_idx]
            
            # 图像
            episode_data['obs']['camera_head_img'].append(
                self._process_image(sample['observation.images.camera_head']))
            episode_data['obs']['camera_left_wrist_img'].append(
                self._process_image(sample['observation.images.camera_left_wrist']))
            episode_data['obs']['camera_right_wrist_img'].append(
                self._process_image(sample['observation.images.camera_right_wrist']))
            episode_data['obs']['camera_third_view_img'].append(
                self._process_image(sample['observation.images.camera_third_view']))
            
            # 状态
            episode_data['obs']['left_arm_joint_pos'].append(
                self._to_numpy(sample['observation.state.left_arm_joint_pos']))
            episode_data['obs']['left_arm_joint_vel'].append(
                self._to_numpy(sample['observation.state.left_arm_joint_vel']))
            episode_data['obs']['left_arm_joint_eff'].append(
                self._to_numpy(sample['observation.state.left_arm_joint_eff']))
            episode_data['obs']['right_arm_joint_pos'].append(
                self._to_numpy(sample['observation.state.right_arm_joint_pos']))
            episode_data['obs']['right_arm_joint_vel'].append(
                self._to_numpy(sample['observation.state.right_arm_joint_vel']))
            episode_data['obs']['right_arm_joint_eff'].append(
                self._to_numpy(sample['observation.state.right_arm_joint_eff']))
            episode_data['obs']['left_hand_obs'].append(
                self._to_numpy(sample['observation.state.left_hand_obs']))
            episode_data['obs']['right_hand_obs'].append(
                self._to_numpy(sample['observation.state.right_hand_obs']))
            
            # 动作
            episode_data['action'].append(self._to_numpy(sample['action']))
        
        # 转换为numpy数组
        for key in episode_data['obs'].keys():
            episode_data['obs'][key] = np.array(episode_data['obs'][key])
        episode_data['action'] = np.array(episode_data['action'])
        
        # 添加元信息
        episode_data['episode_idx'] = episode_idx
        episode_data['num_frames'] = T
        
        return episode_data
    
    def _to_numpy(self, data):
        """将tensor转换为numpy数组"""
        if hasattr(data, 'numpy'):
            return data.numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)
    
    def _process_image(self, img):
        """处理图像：转换为numpy并调整格式"""
        img = self._to_numpy(img)
        
        # 根据设置转换格式
        if self.image_format == 'HWC' and img.shape[0] == 3:
            # 从 (C, H, W) 转换为 (H, W, C)
            img = np.transpose(img, (1, 2, 0))
        
        return img
    
    def get_frame_count(self, episode_idx):
        """获取某个episode的帧数"""
        return self.episodes[episode_idx]['count']


# ============ 测试代码 ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-dir", 
                       default="/home/chensiqi/chensiqi/RDT_baai/data/baai/data/lerobot_test")
    parser.add_argument("--image-format", default="HWC", choices=["CHW", "HWC"])
    args = parser.parse_args()
    
    print("="*70)
    print(" Episode级别数据集测试")
    print("="*70)
    
    # 创建数据集
    dataset = EpisodeBasedDataset(args.lerobot_dir, image_format=args.image_format)
    
    print(f"\n数据集长度（Episode数）: {len(dataset)}")
    
    # 测试获取Episode 0
    print("\n" + "="*70)
    print(" 测试获取Episode 0")
    print("="*70)
    
    episode_0 = dataset[0]
    
    print(f"\nEpisode 0信息:")
    print(f"  帧数: {episode_0['num_frames']}")
    print(f"  Episode索引: {episode_0['episode_idx']}")
    
    print(f"\nobs数据形状:")
    for key, value in episode_0['obs'].items():
        print(f"  {key:30s}: {value.shape}")
    
    print(f"\naction形状: {episode_0['action'].shape}")
    
    # 测试访问第一帧和最后一帧
    print(f"\n第一帧action前6维: {episode_0['action'][0, :6]}")
    print(f"最后一帧action前6维: {episode_0['action'][-1, :6]}")
    
    # 测试所有episodes
    print("\n" + "="*70)
    print(" 所有Episodes概览")
    print("="*70)
    
    for i in range(len(dataset)):
        ep = dataset[i]
        print(f"\nEpisode {i}:")
        print(f"  帧数: {ep['num_frames']}")
        print(f"  图像形状: {ep['obs']['camera_head_img'].shape}")
        print(f"  动作形状: {ep['action'].shape}")
    
    print("\n✅ 测试完成!")