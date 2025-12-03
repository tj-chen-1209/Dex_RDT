#!/usr/bin/env python3
"""BSON到LeRobot格式转换 - siqi"""
import sys
import yaml
import shutil
import argparse
import os
import bson
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


## step1： 从bson取数据
def extract_data_from_bson(episode_path):
    """从BSON文件中提取数据"""
    arm_bson = os.path.join(episode_path, "episode_0.bson")
    xhand_bson = os.path.join(episode_path, "xhand_control_data.bson")
    print("extracting data from bson files...")
    # 检查文件存在性
    if not (os.path.exists(arm_bson) and os.path.exists(xhand_bson)):
        print(f"bson files do not exist: {arm_bson} or {xhand_bson}")
        return None
    print("bson files exist...")    
    # 读取BSON文件
    try:
        with open(arm_bson, 'rb') as f:
            arm_data = bson.decode(f.read())["data"]
        with open(xhand_bson, 'rb') as f:
            xhand_data = bson.decode(f.read())
    except Exception as e:
        print(f"读取BSON失败: {e}")
        return None
    print("bson files read successfully...")
    # 获取帧数
    arm_frame_num = len(arm_data["/observation/left_arm/joint_state"])
    xhand_frame_num = len(xhand_data['frames'])
    frame_num = min(arm_frame_num, xhand_frame_num)
    print(f"frame number: {frame_num}")    
    
    # 检查动作数据是否可用
    arm_dim, hand_dim = 6, 12
    use_arm_actions = True
    try:
        left_test = arm_data["/action/left_arm/joint_state"][0]["data"]["pos"]
        right_test = arm_data["/action/right_arm/joint_state"][0]["data"]["pos"]
        if len(left_test) != arm_dim or len(right_test) != arm_dim:
            use_arm_actions = False
    except (KeyError, IndexError):
        use_arm_actions = False
        print("⚠️  动作数据不可用，使用观测数据代替")
    print(f"use_arm_actions: {use_arm_actions}")
    
    # 分别存储各个状态
    left_arm_pos = []
    right_arm_pos = []
    left_arm_vel = []
    right_arm_vel = []
    left_arm_eff = []
    right_arm_eff = []
    left_hand_obs = []
    right_hand_obs = []
    actions = []
    
    print("Extracting state and action data...")
    for i in range(frame_num):
        # 提取机械臂状态（位置、速度、力矩）
        left_arm_pos.append(arm_data["/observation/left_arm/joint_state"][i]["data"]["pos"])
        right_arm_pos.append(arm_data["/observation/right_arm/joint_state"][i]["data"]["pos"])
        left_arm_vel.append(arm_data["/observation/left_arm/joint_state"][i]["data"]["vel"])
        right_arm_vel.append(arm_data["/observation/right_arm/joint_state"][i]["data"]["vel"])
        left_arm_eff.append(arm_data["/observation/left_arm/joint_state"][i]["data"]["eff"])
        right_arm_eff.append(arm_data["/observation/right_arm/joint_state"][i]["data"]["eff"])
        
        # 灵巧手状态（度转弧度）
        left_hand_obs.append(np.deg2rad(xhand_data['frames'][i]["observation"]["left_hand"]))
        right_hand_obs.append(np.deg2rad(xhand_data['frames'][i]["observation"]["right_hand"]))
        
        # 动作
        if use_arm_actions:
            right_arm = arm_data["/action/right_arm/joint_state"][i]["data"]["pos"]
            left_arm = arm_data["/action/left_arm/joint_state"][i]["data"]["pos"]
        else:
            right_arm = arm_data["/observation/right_arm/joint_state"][i]["data"]["pos"]
            left_arm = arm_data["/observation/left_arm/joint_state"][i]["data"]["pos"]
        
        actions.append(np.concatenate([
            right_arm,
            xhand_data['frames'][i]["action"]["right_hand"],
            left_arm,
            xhand_data['frames'][i]["action"]["left_hand"],
        ]))
    
    print("State and action data extracted...")
    # 获取图像文件列表
    camera_folders = ['camera_head', 'camera_left_wrist', 'camera_right_wrist', 'camera_third_view']
    image_files = {}
    print("Getting image files...")
    for cam in camera_folders:
        cam_path = os.path.join(episode_path, cam)
        if not os.path.exists(cam_path):
            print(f"⚠️  相机文件夹不存在: {cam}")
            return None
        jpg_files = sorted([f for f in os.listdir(cam_path) if f.endswith('.jpg')])
        image_files[cam] = jpg_files[:frame_num]
        if len(jpg_files) < frame_num:
            print(f"⚠️  {cam} 图像数量不足")
            return None
    
    # 返回分离的各个字段
    return {
        'left_arm_pos': np.array(left_arm_pos, dtype=np.float32),
        'right_arm_pos': np.array(right_arm_pos, dtype=np.float32),
        'left_arm_vel': np.array(left_arm_vel, dtype=np.float32),
        'right_arm_vel': np.array(right_arm_vel, dtype=np.float32),
        'left_arm_eff': np.array(left_arm_eff, dtype=np.float32),
        'right_arm_eff': np.array(right_arm_eff, dtype=np.float32),
        'left_hand_obs': np.array(left_hand_obs, dtype=np.float32),
        'right_hand_obs': np.array(right_hand_obs, dtype=np.float32),
        'action': np.array(actions, dtype=np.float32),
        'frame_num': frame_num,
        'image_files': image_files,
    }

def convert_bson_to_lerobot(
    bson_dir="/home/zhukefei/chensiqi/Dex_RDT/data/baai/data",
    output_repo_id="baai/xhand_bimanual_action176",
    output_root="/home/zhukefei/chensiqi/Dex_RDT/data/baai/data/lerobot_baai",
    fps=20,
    robot_type="xhand_bimanual",
    use_videos=True,
    max_episodes=None,  # 新增：限制转换的episode数量，None表示转换全部
):
    """
    将BSON格式数据集转换为LeRobot格式
    
    Args:
        bson_dir: BSON数据集根目录
        output_repo_id: 输出的LeRobot数据集ID
        output_root: 输出根路径
        fps: 帧率（默认20）
        robot_type: 机器人类型
        use_videos: 是否使用视频格式（推荐True）
        max_episodes: 最多转换的episode数量，None表示转换全部
    """
    
    # 1. 定义features（根据你的数据结构）
    features = {
        # 观测：双臂关节 + 双手关节
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
        # 动作：与观测维度相同
        "action": {
            "dtype": "float32",
            "shape": (36,),
            "names": [
                *[f"right_arm_joint_{i}" for i in range(6)],
                *[f"right_hand_joint_{i}" for i in range(12)],
                *[f"left_arm_joint_{i}" for i in range(6)],
                *[f"left_hand_joint_{i}" for i in range(12)],
            ]
        },
        # 相机：3个视角
        "observation.images.camera_head_img": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"]
        },
        "observation.images.camera_left_wrist_img": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"]
        },
        "observation.images.camera_right_wrist_img": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"]
        },
        "observation.images.camera_third_view_img": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"]
        },
    }
    
    print("📦 创建LeRobot数据集...")
    dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=fps,
        features=features,
        robot_type=robot_type,
        root=output_root,
        use_videos=use_videos,
    )
    
    # 2. 遍历所有action目录
    action_dirs = sorted([d for d in os.listdir(bson_dir) if d.startswith('action')])
    
    # 读取任务指令
    task_instructions = {}
    task_description = "Use the left hand to hook the book '皮囊' from the pile of books,then use the right hand to place it on the right bookshelf."
    for action_dir in action_dirs:
        task_instructions[action_dir] = task_description
        
        print(f"📂 找到 {len(action_dirs)} 个action目录")
    
    global_ep_idx = 0
    
    # 显示转换模式
    if max_episodes is not None:
        print(f"🧪 测试模式：仅转换前 {max_episodes} 个episodes")
    else:
        print(f"📦 完整转换模式：转换所有episodes")
    
    for action_dir in tqdm(action_dirs, desc="处理action目录"):
        # 检查是否已达到最大episode数量
        if max_episodes is not None and global_ep_idx >= max_episodes:
            print(f"\n✅ 已达到最大episode数量 ({max_episodes})，停止转换")
            break
            
        action_path = os.path.join(bson_dir, action_dir)
        task = task_instructions.get(action_dir, f"Task {action_dir}")
        
        # 获取所有episode
        episode_dirs = sorted([
            d for d in os.listdir(action_path) 
            if d.startswith('episode') and os.path.isdir(os.path.join(action_path, d))
        ])
        
        for episode_dir in tqdm(episode_dirs, desc=f"  {action_dir}", leave=False):
            # 检查是否已达到最大episode数量
            if max_episodes is not None and global_ep_idx >= max_episodes:
                break
            episode_path = os.path.join(action_path, episode_dir)
            
            # 3. 读取BSON数据
            try:
                episode_data = extract_data_from_bson(episode_path)
                if episode_data is None:
                    print(f"⚠️  跳过无效episode: {episode_path}")
                    continue
            except Exception as e:
                print(f"❌ 读取episode失败 {episode_path}: {e}")
                continue
            
            # 4. 逐帧添加数据（修复：使用正确的字段名，添加所有相机）
            frame_num = episode_data['frame_num']
            for frame_idx in range(frame_num):
                # 准备frame数据（修复：分离各个状态字段）
                frame = {
                    "observation.state.left_arm_joint_pos": episode_data['left_arm_pos'][frame_idx],
                    "observation.state.right_arm_joint_pos": episode_data['right_arm_pos'][frame_idx],
                    "observation.state.left_arm_joint_vel": episode_data['left_arm_vel'][frame_idx],
                    "observation.state.right_arm_joint_vel": episode_data['right_arm_vel'][frame_idx],
                    "observation.state.left_arm_joint_eff": episode_data['left_arm_eff'][frame_idx],
                    "observation.state.right_arm_joint_eff": episode_data['right_arm_eff'][frame_idx],
                    "observation.state.left_hand_obs": episode_data['left_hand_obs'][frame_idx],
                    "observation.state.right_hand_obs": episode_data['right_hand_obs'][frame_idx],
                    "action": episode_data['action'][frame_idx],
                    "task": task,
                }
                
                # 添加所有4个相机图像
                for cam_key, lerobot_key in [
                    ('camera_head', 'observation.images.camera_head_img'),
                    ('camera_left_wrist', 'observation.images.camera_left_wrist_img'),
                    ('camera_right_wrist', 'observation.images.camera_right_wrist_img'),
                    ('camera_third_view', 'observation.images.camera_third_view_img'),
                ]:
                    img_path = os.path.join(
                        episode_path, cam_key, 
                        episode_data['image_files'][cam_key][frame_idx]
                    )
                    img = Image.open(img_path)
                    img_array = np.array(img, dtype=np.uint8)
                    
                    # 处理灰度图
                    if img_array.ndim == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    frame[lerobot_key] = img_array
                
                # 添加帧
                dataset.add_frame(frame)
            
            # 6. 保存episode
            dataset.save_episode()
            global_ep_idx += 1
            
            print(f"✅ Episode {global_ep_idx-1}: {frame_num} frames")
    
    # 7. 完成数据集
    print("\n🎉 完成转换，保存数据集...")
    dataset.finalize()
    print(f"✅ 数据集已保存到: {dataset.root}")
    print(f"📊 总计: {global_ep_idx} episodes, {dataset.meta.total_frames} frames")
    
    return dataset

def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def convert_with_config(config_path: str, override_output: str = None, override_max_episodes: int = None):
    """
    使用配置文件进行转换
    
    Args:
        config_path: 配置文件路径
        override_output: 覆盖输出目录（可选）
        override_max_episodes: 覆盖最大episode数（可选）
    """
    # 加载配置
    config = load_config(config_path)
    
    # 读取配置
    output_root = config['data']['output_root']
    max_episodes = config['conversion']['max_episodes']
    
    # 命令行参数覆盖
    if override_output:
        output_root = override_output
        print(f"📂 使用自定义输出目录: {output_root}")
    
    if override_max_episodes is not None:
        max_episodes = override_max_episodes
        print(f"📊 覆盖最大episodes数: {max_episodes}")
    
    # 显示配置信息
    print("\n" + "="*70)
    print("📋 转换配置")
    print("="*70)
    print(f"输入目录:     {config['data']['bson_dir']}")
    print(f"输出目录:     {output_root}")
    print(f"Repository ID: {config['data']['output_repo_id']}")
    print(f"FPS:          {config['dataset']['fps']}")
    print(f"Robot Type:   {config['dataset']['robot_type']}")
    print(f"Use Videos:   {config['dataset']['use_videos']}")
    print(f"Max Episodes: {max_episodes if max_episodes else '全部'}")
    print(f"Task:         {config['task']['description'][:60]}...")
    print("="*70 + "\n")
    
    # 执行转换
    try:
        dataset = convert_bson_to_lerobot(
            bson_dir=config['data']['bson_dir'],
            output_repo_id=config['data']['output_repo_id'],
            output_root=output_root,
            fps=config['dataset']['fps'],
            robot_type=config['dataset']['robot_type'],
            use_videos=config['dataset']['use_videos'],
            max_episodes=max_episodes,
        )
        
        print("\n" + "="*70)
        print("✅ 转换完成!")
        print("="*70)
        print(f"输出位置:   {dataset.root}")
        print(f"Episodes:   {dataset.meta.total_episodes}")
        print(f"Frames:     {dataset.meta.total_frames}")
        print(f"Tasks:      {dataset.meta.total_tasks}")
        print("="*70)
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ 转换失败: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='BSON 到 LeRobot 格式转换（使用 YAML 配置）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认配置转换
  python convert_bson2lerobot.py
  
  # 使用自定义配置文件
  python convert_bson2lerobot.py -c config/my_config.yml
  
  # 覆盖输出目录
  python convert_bson2lerobot.py -o /path/to/output
  
  # 仅转换前10个episodes（测试用）
  python convert_bson2lerobot.py -n 10
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/convert.yml',
        help='YAML配置文件路径 (默认: config/convert.yml)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='输出目录（覆盖配置文件设置）'
    )
    
    parser.add_argument(
        '--max-episodes', '-n',
        type=int,
        default=None,
        help='最多转换的episode数量（覆盖配置文件设置）'
    )
    
    args = parser.parse_args()
    
    # 解析配置文件路径（相对于脚本目录）
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 执行转换
    success = convert_with_config(
        config_path=str(config_path),
        override_output=args.output_dir,
        override_max_episodes=args.max_episodes
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()