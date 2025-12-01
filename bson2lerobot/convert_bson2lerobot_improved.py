#!/usr/bin/env python3
"""BSON到LeRobot格式转换 - 改进版
改进点：
1. 修复测试模式的切片错误
2. 添加失败episode的记录和汇总
3. 添加数据验证
4. 改进日志输出
5. 添加进度保存功能
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import bson
from tqdm import tqdm
import json
import logging
from datetime import datetime

# 添加lerobot路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lerobot' / 'src'))
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'conversion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class BSONToLeRobotConverter:
    def __init__(self, bson_dir="data/baai/data/", output_dir="data/baai/data/lerobot_baai", 
                 fps=20, test_mode=False, resume=False):
        self.bson_dir = Path(bson_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.resume = resume
        self.progress_file = self.output_dir / "conversion_progress.json"
        self.completed_episodes = self._load_progress() if resume else set()
        self.failed_episodes = []
        self.success_count = 0
        
        self.episode_paths = self._find_episodes()
        
        if test_mode:
            logger.warning("⚠️  测试模式：只转换前3个episodes")
            self.episode_paths = self.episode_paths[:3]  # 修复：改为3个
        
        logger.info(f"找到 {len(self.episode_paths)} 个episodes")
        
        # 统计每个action
        action_counts = {}
        for ep_path in self.episode_paths:
            action_name = ep_path.parent.name
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        logger.info("\n按Action分组:")
        for action, count in sorted(action_counts.items()):
            logger.info(f"  {action}: {count} episodes")
    
    def _load_progress(self):
        """加载转换进度"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                completed = set(data.get('completed_episodes', []))
                logger.info(f"恢复进度：已完成 {len(completed)} 个episodes")
                return completed
            except Exception as e:
                logger.warning(f"加载进度文件失败: {e}")
        return set()
    
    def _save_progress(self, episode_name):
        """保存转换进度"""
        self.completed_episodes.add(episode_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump({
                'completed_episodes': list(self.completed_episodes),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def _find_episodes(self):
        """查找所有有效的episode"""
        episodes = []
        for action_dir in sorted(self.bson_dir.glob("action*")):
            for ep_dir in sorted(action_dir.glob("episode_*")):
                # 如果恢复模式且已完成，跳过
                if self.resume and str(ep_dir) in self.completed_episodes:
                    continue
                    
                if (ep_dir / "episode_0.bson").exists() and \
                   (ep_dir / "xhand_control_data.bson").exists() and \
                   all((ep_dir / cam).exists() for cam in 
                       ['camera_head', 'camera_left_wrist', 'camera_right_wrist', 'camera_third_view']):
                    episodes.append(ep_dir)
        return episodes
    
    def _validate_episode_data(self, ep_data, ep_path):
        """验证episode数据的完整性"""
        issues = []
        
        # 检查帧数
        if ep_data['n_frames'] == 0:
            issues.append("帧数为0")
            return issues
        
        # 检查action维度
        if ep_data['action'].shape[1] != 36:
            issues.append(f"action维度错误: {ep_data['action'].shape[1]} != 36")
        
        # 检查NaN和Inf
        for key in ['left_arm_pos', 'right_arm_pos', 'left_arm_vel', 'right_arm_vel', 
                    'left_arm_eff', 'right_arm_eff', 'left_hand_obs', 'right_hand_obs', 'action']:
            if np.any(np.isnan(ep_data[key])) or np.any(np.isinf(ep_data[key])):
                issues.append(f"{key}包含NaN或Inf")
        
        # 检查图像数量
        for cam, img_list in ep_data['images'].items():
            if len(img_list) != ep_data['n_frames']:
                issues.append(f"{cam}图像数量({len(img_list)}) != 帧数({ep_data['n_frames']})")
        
        return issues
    
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
            
            # 注意：灵巧手action单位需要确认，这里假设已经是弧度
            action = np.concatenate([
                right_arm,
                hand_data['frames'][i]["action"]["right_hand"],
                left_arm,
                hand_data['frames'][i]["action"]["left_hand"]
            ])
            data['action'].append(action)
        
        # 转换为numpy
        for key in ['left_arm_pos', 'right_arm_pos', 'left_arm_vel', 'right_arm_vel', 
                    'left_arm_eff', 'right_arm_eff', 'left_hand_obs', 'right_hand_obs', 'action']:
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
        logger.info(f"\n创建LeRobot数据集: {self.output_dir}")
        dataset = LeRobotDataset.create(
            repo_id="baai/bimanual_dexhand",
            fps=self.fps,
            root=self.output_dir,
            robot_type="bimanual_dexhand",
            features=features,
            use_videos=True,
        )
        
        # 转换所有episodes
        logger.info(f"\n开始转换 {len(self.episode_paths)} 个episodes...")
        for ep_idx, ep_path in enumerate(self.episode_paths):
            logger.info(f"\n[{ep_idx+1}/{len(self.episode_paths)}] {ep_path.name}")
            
            try:
                # 加载episode数据
                ep_data = self._load_episode(ep_path)
                n_frames = ep_data['n_frames']
                
                # 验证数据
                issues = self._validate_episode_data(ep_data, ep_path)
                if issues:
                    error_msg = f"数据验证失败: {', '.join(issues)}"
                    logger.error(f"  ✗ {error_msg}")
                    self.failed_episodes.append({
                        'path': str(ep_path),
                        'error': error_msg
                    })
                    continue
                
                # 逐帧添加
                for i in tqdm(range(n_frames), desc="  添加帧", leave=False):
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
                self._save_progress(str(ep_path))
                self.success_count += 1
                logger.info(f"  ✓ Episode {ep_idx} 完成 ({n_frames} 帧)")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"  ✗ 错误: {error_msg}")
                import traceback
                traceback.print_exc()
                self.failed_episodes.append({
                    'path': str(ep_path),
                    'error': error_msg
                })
        
        # 完成
        logger.info("\n正在finalize...")
        dataset.finalize()
        
        # 打印总结
        self._print_summary()
    
    def _print_summary(self):
        """打印转换总结"""
        logger.info("\n" + "="*70)
        logger.info(" 转换总结")
        logger.info("="*70)
        logger.info(f"✅ 成功转换: {self.success_count} 个episodes")
        
        if self.failed_episodes:
            logger.warning(f"❌ 失败: {len(self.failed_episodes)} 个episodes")
            logger.warning("\n失败的episodes:")
            for failed in self.failed_episodes:
                logger.warning(f"  - {failed['path']}")
                logger.warning(f"    错误: {failed['error']}")
            
            # 保存失败记录
            failed_log = self.output_dir / "failed_episodes.json"
            with open(failed_log, 'w') as f:
                json.dump(self.failed_episodes, f, indent=2)
            logger.warning(f"\n失败记录已保存到: {failed_log}")
        
        logger.info(f"\n输出目录: {self.output_dir}")
        logger.info(f"总episodes: {self.success_count + len(self.failed_episodes)}")
        logger.info("="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BSON转LeRobot格式 - 改进版")
    parser.add_argument("--bson-dir", default="data/baai/data/", help="BSON数据目录")
    parser.add_argument("--output-dir", default="data/baai/data/lerobot_baai", help="输出目录")
    parser.add_argument("--fps", type=int, default=20, help="帧率")
    parser.add_argument("--test", action="store_true", help="测试模式：只转换前3个episodes")
    parser.add_argument("--resume", action="store_true", help="恢复模式：跳过已完成的episodes")
    args = parser.parse_args()
    
    converter = BSONToLeRobotConverter(
        args.bson_dir, 
        args.output_dir, 
        args.fps,
        test_mode=args.test,
        resume=args.resume
    )
    converter.convert()

