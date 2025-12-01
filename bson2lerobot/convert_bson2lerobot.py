#!/usr/bin/env python3
"""BSON到LeRobot格式转换 - 最小实现"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import bson
from tqdm import tqdm

# 添加lerobot路径
sys.path.insert(0, '/home/chensiqi/chensiqi/lerobot/src')
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class BSONToLeRobotConverter:
    def __init__(self, bson_dir="data/baai/data/", output_dir="data/baai/data/lerobot_baai", fps=20, test_mode=False):
        self.bson_dir = Path(bson_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.episode_paths = self._find_episodes()
        
        if test_mode:
            print(f"⚠️  测试模式：只转换前3个episodes")
            self.episode_paths = self.episode_paths[:2]
        
        print(f"✓ 找到 {len(self.episode_paths)} 个episodes")
        
        # 统计每个action
        action_counts = {}
        for ep_path in self.episode_paths:
            action_name = ep_path.parent.name
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        print(f"\n按Action分组:")
        for action, count in sorted(action_counts.items()):
            print(f"  {action}: {count} episodes")
    
    def _find_episodes(self):
        """查找所有有效的episode"""
        episodes = []
        for action_dir in sorted(self.bson_dir.glob("action*")):
            for ep_dir in sorted(action_dir.glob("episode_*")):
                if (ep_dir / "episode_0.bson").exists() and \
                   (ep_dir / "xhand_control_data.bson").exists() and \
                   all((ep_dir / cam).exists() for cam in 
                       ['camera_head', 'camera_left_wrist', 'camera_right_wrist', 'camera_third_view']):
                    episodes.append(ep_dir)
        return episodes
    
    def _load_episode(self, ep_path):
        """加载单个episode的数据"""
        # 读取BSON
        with open(ep_path / "episode_0.bson", 'rb') as f:
            arm_data = bson.decode(f.read())["data"]
        with open(ep_path / "xhand_control_data.bson", 'rb') as f:
            hand_data = bson.decode(f.read())
        
        # 获取帧数
        n_frames = min(
            len(arm_data["/observation/left_arm/joint_state"]),
            len(hand_data['frames'])
        )
        
        # 提取数据
        data = {
            'n_frames': n_frames,
            'left_arm_pos': [],
            'right_arm_pos': [],
            'left_arm_vel': [],
            'right_arm_vel': [],
            'left_arm_eff': [],
            'right_arm_eff': [],
            'left_hand_obs': [],
            'right_hand_obs': [],
            'action': [],
            'images': {cam: [] for cam in ['camera_head', 'camera_left_wrist', 
                                           'camera_right_wrist', 'camera_third_view']}
        }
        
        # 检查是否有action数据
        try:
            arm_data["/action/left_arm/joint_state"][0]["data"]["pos"]
            use_arm_action = True
        except:
            use_arm_action = False
        
        # 逐帧提取
        for i in range(n_frames):
            # 机械臂状态
            data['left_arm_pos'].append(arm_data["/observation/left_arm/joint_state"][i]["data"]["pos"])
            data['right_arm_pos'].append(arm_data["/observation/right_arm/joint_state"][i]["data"]["pos"])
            data['left_arm_vel'].append(arm_data["/observation/left_arm/joint_state"][i]["data"]["vel"])
            data['right_arm_vel'].append(arm_data["/observation/right_arm/joint_state"][i]["data"]["vel"])
            data['left_arm_eff'].append(arm_data["/observation/left_arm/joint_state"][i]["data"]["eff"])
            data['right_arm_eff'].append(arm_data["/observation/right_arm/joint_state"][i]["data"]["eff"])
            
            # 灵巧手状态（度转弧度）
            data['left_hand_obs'].append(np.deg2rad(hand_data['frames'][i]["observation"]["left_hand"]))
            data['right_hand_obs'].append(np.deg2rad(hand_data['frames'][i]["observation"]["right_hand"]))
            
            # 动作
            if use_arm_action:
                left_arm = arm_data["/action/left_arm/joint_state"][i]["data"]["pos"]
                right_arm = arm_data["/action/right_arm/joint_state"][i]["data"]["pos"]
            else:
                left_arm = data['left_arm_pos'][-1]
                right_arm = data['right_arm_pos'][-1]
            
            action = np.concatenate([
                right_arm,
                hand_data['frames'][i]["action"]["right_hand"],
                left_arm,
                hand_data['frames'][i]["action"]["left_hand"]
            ])
            data['action'].append(action)
        
        # 转换为numpy
        for key in ['left_arm_pos', 'right_arm_pos', 'left_arm_vel', 'right_arm_vel', 'left_arm_eff', 'right_arm_eff', 'left_hand_obs', 'right_hand_obs', 'action']:
            data[key] = np.array(data[key], dtype=np.float32)
        
        # 图像文件路径
        for cam in data['images'].keys():
            cam_dir = ep_path / cam
            data['images'][cam] = sorted(cam_dir.glob("*.jpg"))[:n_frames]
        
        return data
    
    def convert(self):
        """执行转换"""
        # 定义features
        features = {
            "observation.images.camera_head": {
                "dtype": "video", "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.camera_left_wrist": {
                "dtype": "video", "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.camera_right_wrist": {
                "dtype": "video", "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.camera_third_view": {
                "dtype": "video", "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state.left_arm_joint_pos": {
                "dtype": "float32", "shape": (6,),
                "names": [f"joint_{i}" for i in range(6)],
            },
            "observation.state.right_arm_joint_pos": {
                "dtype": "float32", "shape": (6,),
                "names": [f"joint_{i}" for i in range(6)],
            },
            "observation.state.left_arm_joint_vel": {
                "dtype": "float32", "shape": (6,),
                "names": [f"joint_{i}" for i in range(6)],
            },
            "observation.state.right_arm_joint_vel": {
                "dtype": "float32", "shape": (6,),
                "names": [f"joint_{i}" for i in range(6)],
            },
            "observation.state.left_arm_joint_eff": {
                "dtype": "float32", "shape": (6,),
                "names": [f"joint_{i}" for i in range(6)],
            },
            "observation.state.right_arm_joint_eff": {
                "dtype": "float32", "shape": (6,),
                "names": [f"joint_{i}" for i in range(6)],
            },
            "observation.state.left_hand_obs": {
                "dtype": "float32", "shape": (12,),
                "names": [f"joint_{i}" for i in range(12)],
            },
            "observation.state.right_hand_obs": {
                "dtype": "float32", "shape": (12,),
                "names": [f"joint_{i}" for i in range(12)],
            },
            "action": {
                "dtype": "float32", "shape": (36,),
                "names": ([f"right_arm_{i}" for i in range(6)] +
                         [f"right_hand_{i}" for i in range(12)] +
                         [f"left_arm_{i}" for i in range(6)] +
                         [f"left_hand_{i}" for i in range(12)]),
            },
        }
        
        # 创建LeRobot数据集
        print(f"\n创建LeRobot数据集: {self.output_dir}")
        dataset = LeRobotDataset.create(
            repo_id="baai/bimanual_dexhand",
            fps=self.fps,
            root=self.output_dir,
            robot_type="bimanual_dexhand",
            features=features,
            use_videos=True,
        )
        
        # 转换所有episodes
        print(f"\n开始转换 {len(self.episode_paths)} 个episodes...")
        for ep_idx, ep_path in enumerate(self.episode_paths):
            print(f"\n[{ep_idx+1}/{len(self.episode_paths)}] {ep_path.name}")
            
            try:
                # 加载episode数据
                ep_data = self._load_episode(ep_path)
                n_frames = ep_data['n_frames']
                
                # 逐帧添加
                for i in tqdm(range(n_frames), desc="  添加帧"):
                    # 加载图像
                    imgs = {}
                    for cam in ['camera_head', 'camera_left_wrist', 'camera_right_wrist', 'camera_third_view']:
                        img = np.array(Image.open(ep_data['images'][cam][i]))
                        if img.ndim == 2:  # 灰度转RGB
                            img = np.stack([img] * 3, axis=-1)
                        imgs[cam] = img
                    
                    # 构建frame
                    frame = {
                        "observation.images.camera_head": imgs['camera_head'],
                        "observation.images.camera_left_wrist": imgs['camera_left_wrist'],
                        "observation.images.camera_right_wrist": imgs['camera_right_wrist'],
                        "observation.images.camera_third_view": imgs['camera_third_view'],
                        "observation.state.left_arm_joint_pos": ep_data['left_arm_pos'][i],
                        "observation.state.right_arm_joint_pos": ep_data['right_arm_pos'][i],
                        "observation.state.left_arm_joint_vel": ep_data['left_arm_vel'][i],
                        "observation.state.right_arm_joint_vel": ep_data['right_arm_vel'][i],
                        "observation.state.left_arm_joint_eff": ep_data['left_arm_eff'][i],
                        "observation.state.right_arm_joint_eff": ep_data['right_arm_eff'][i],
                        "observation.state.left_hand_obs": ep_data['left_hand_obs'][i],
                        "observation.state.right_hand_obs": ep_data['right_hand_obs'][i],
                        "action": ep_data['action'][i],
                        "task": f"Bimanual task: {ep_path.parent.name}",
                    }
                    
                    dataset.add_frame(frame)
                
                # 保存episode
                dataset.save_episode()
                print(f"  ✓ Episode {ep_idx} 完成 ({n_frames} 帧)")
                
            except Exception as e:
                print(f"  ✗ 错误: {e}")
                import traceback
                traceback.print_exc()
        
        # 完成
        print("\n正在finalize...")
        dataset.finalize()
        print(f"\n✅ 转换完成! 输出目录: {self.output_dir}")
        print(f"   Episodes: {len(self.episode_paths)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BSON转LeRobot格式")
    parser.add_argument("--bson-dir", default="data/baai/data/", help="BSON数据目录")
    parser.add_argument("--output-dir", default="data/baai/data/lerobot_baai", help="输出目录")
    parser.add_argument("--fps", type=int, default=20, help="帧率")
    parser.add_argument("--test", action="store_true", help="测试模式：只转换前3个episodes")
    args = parser.parse_args()
    
    converter = BSONToLeRobotConverter(
        args.bson_dir, 
        args.output_dir, 
        args.fps,
        test_mode=args.test
    )
    converter.convert()