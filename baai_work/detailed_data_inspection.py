#!/usr/bin/env python3
"""
详细输出 action176 数据集的图片和数据维度信息
"""
import os
import bson
import numpy as np
from PIL import Image
import json


def inspect_images_detailed():
    """详细检查图像数据"""
    print("="*80)
    print("图像数据详细信息")
    print("="*80)
    
    episode_dir = "/home/chensiqi/chensiqi/RDT_libero_finetune/data/baai/data/action176/episode_0"
    cameras = ['camera_head', 'camera_left_wrist', 'camera_right_wrist', 'camera_third_view']
    
    for camera_name in cameras:
        camera_dir = os.path.join(episode_dir, camera_name)
        image_files = sorted([f for f in os.listdir(camera_dir) if f.endswith('.jpg')])
        
        print(f"\n{'─'*80}")
        print(f"📷 {camera_name}")
        print(f"{'─'*80}")
        
        # 读取第一张、中间一张、最后一张图像
        sample_indices = [0, len(image_files)//2, -1]
        
        for idx in sample_indices:
            img_path = os.path.join(camera_dir, image_files[idx])
            img = Image.open(img_path)
            img_array = np.array(img)
            
            print(f"\n  文件名: {image_files[idx]}")
            print(f"  文件大小: {os.path.getsize(img_path) / 1024:.2f} KB")
            print(f"  PIL 图像信息:")
            print(f"    • 尺寸 (width, height): {img.size}")
            print(f"    • 模式: {img.mode}")
            print(f"    • 格式: {img.format}")
            
            print(f"  NumPy 数组信息:")
            print(f"    • 形状 (shape): {img_array.shape}")
            print(f"    • 维度解释: (高度={img_array.shape[0]}, 宽度={img_array.shape[1]}, 通道={img_array.shape[2]})")
            print(f"    • 数据类型 (dtype): {img_array.dtype}")
            print(f"    • 内存占用: {img_array.nbytes / 1024:.2f} KB")
            print(f"    • 像素值范围: [{img_array.min()}, {img_array.max()}]")
            print(f"    • 整体均值: {img_array.mean():.2f}")
            print(f"    • 整体标准差: {img_array.std():.2f}")
            
            print(f"  各通道统计:")
            for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
                channel = img_array[:, :, i]
                print(f"    • {channel_name:5s} 通道: min={channel.min():3d}, max={channel.max():3d}, "
                      f"mean={channel.mean():6.2f}, std={channel.std():5.2f}")
            
            if idx == 0:  # 只对第一张图显示更多细节
                print(f"  像素数据样本 (左上角 5×5 区域, R通道):")
                print(f"{img_array[:5, :5, 0]}")
                break  # 只详细显示一张图片
        
        print(f"\n  总计: {len(image_files)} 张图像")


def inspect_episode_bson_detailed():
    """详细检查 episode_0.bson"""
    print("\n\n" + "="*80)
    print("episode_0.bson 详细数据结构")
    print("="*80)
    
    bson_path = "/home/chensiqi/chensiqi/RDT_libero_finetune/data/baai/data/action176/episode_0/episode_0.bson"
    
    with open(bson_path, 'rb') as f:
        data = bson.decode_all(f.read())
    
    doc = data[0]
    
    print(f"\n📋 顶级字段: {list(doc.keys())}")
    
    # 详细分析每个 data 主题
    print(f"\n{'='*80}")
    print("Data 主题详细信息")
    print(f"{'='*80}")
    
    for topic_name, topic_data in doc['data'].items():
        print(f"\n🔹 {topic_name}")
        print(f"{'─'*80}")
        print(f"  数据点数量: {len(topic_data)}")
        
        if len(topic_data) > 0:
            first_point = topic_data[0]
            print(f"  每个数据点的结构: {list(first_point.keys())}")
            
            # 分析 data 字段
            data_content = first_point['data']
            print(f"\n  data 字段的键: {list(data_content.keys())}")
            
            # 详细显示第一个数据点
            print(f"\n  第 1 个数据点 (索引 0):")
            print(f"    时间戳: {first_point['t']}")
            for key, value in data_content.items():
                if isinstance(value, list):
                    value_array = np.array(value)
                    print(f"    {key}:")
                    print(f"      • 类型: 列表/数组")
                    print(f"      • 维度: {len(value)}")
                    print(f"      • 数据类型: {value_array.dtype}")
                    print(f"      • 值: {value}")
                    if len(value) > 0:
                        print(f"      • 范围: [{value_array.min():.4f}, {value_array.max():.4f}]")
                else:
                    print(f"    {key}: {value}")
            
            # 分析中间数据点
            mid_idx = len(topic_data) // 2
            mid_point = topic_data[mid_idx]
            print(f"\n  第 {mid_idx+1} 个数据点 (中间):")
            print(f"    时间戳: {mid_point['t']}")
            for key, value in mid_point['data'].items():
                if isinstance(value, list):
                    print(f"    {key}: {value}")
                else:
                    print(f"    {key}: {value}")
            
            # 统计整个序列
            print(f"\n  整个序列统计 ({len(topic_data)} 个数据点):")
            
            # 收集所有数据点的值
            for key in data_content.keys():
                all_values = []
                for point in topic_data:
                    val = point['data'][key]
                    if val is not None and isinstance(val, list):
                        all_values.append(val)
                
                if all_values:
                    all_values = np.array(all_values)
                    print(f"    {key}:")
                    print(f"      • 数据形状: {all_values.shape}")
                    if all_values.ndim > 1:
                        for dim_idx in range(all_values.shape[1]):
                            dim_data = all_values[:, dim_idx]
                            print(f"      • 维度 {dim_idx}: "
                                  f"min={dim_data.min():8.4f}, "
                                  f"max={dim_data.max():8.4f}, "
                                  f"mean={dim_data.mean():8.4f}, "
                                  f"std={dim_data.std():8.4f}")
                    else:
                        print(f"      • 范围: [{all_values.min():.4f}, {all_values.max():.4f}]")
                        print(f"      • 均值: {all_values.mean():.4f}")
                        print(f"      • 标准差: {all_values.std():.4f}")


def inspect_xhand_bson_detailed():
    """详细检查 xhand_control_data.bson"""
    print("\n\n" + "="*80)
    print("xhand_control_data.bson 详细数据结构")
    print("="*80)
    
    bson_path = "/home/chensiqi/chensiqi/RDT_libero_finetune/data/baai/data/action176/episode_0/xhand_control_data.bson"
    
    with open(bson_path, 'rb') as f:
        data = bson.decode_all(f.read())
    
    doc = data[0]
    frames = doc['frames']
    
    print(f"\n📊 基本信息:")
    print(f"  总帧数: {len(frames)}")
    print(f"  每帧结构: {list(frames[0].keys())}")
    
    # 详细分析第一帧
    print(f"\n{'='*80}")
    print("第 1 帧详细结构 (索引 0)")
    print(f"{'='*80}")
    
    frame = frames[0]
    print(f"\n  时间戳: {frame['t']}")
    
    print(f"\n  🎮 action 字段:")
    print(f"     键: {list(frame['action'].keys())}")
    for hand_name, hand_data in frame['action'].items():
        hand_array = np.array(hand_data)
        print(f"\n     {hand_name}:")
        print(f"       • 数据类型: {type(hand_data).__name__}")
        print(f"       • 维度: {len(hand_data)}")
        print(f"       • NumPy dtype: {hand_array.dtype}")
        print(f"       • 形状: {hand_array.shape}")
        print(f"       • 完整值: {hand_data}")
        print(f"       • 范围: [{hand_array.min():.6f}, {hand_array.max():.6f}]")
        print(f"       • 均值: {hand_array.mean():.6f}")
        print(f"       • 标准差: {hand_array.std():.6f}")
        print(f"       • 各维度值:")
        for i, val in enumerate(hand_data):
            print(f"         [{i:2d}] = {val:.6f}")
    
    print(f"\n  👁️  observation 字段:")
    print(f"     键: {list(frame['observation'].keys())}")
    for hand_name, hand_data in frame['observation'].items():
        hand_array = np.array(hand_data)
        print(f"\n     {hand_name}:")
        print(f"       • 数据类型: {type(hand_data).__name__}")
        print(f"       • 维度: {len(hand_data)}")
        print(f"       • NumPy dtype: {hand_array.dtype}")
        print(f"       • 形状: {hand_array.shape}")
        print(f"       • 完整值: {hand_data}")
        print(f"       • 范围: [{hand_array.min():.2f}, {hand_array.max():.2f}]")
        print(f"       • 均值: {hand_array.mean():.2f}")
        print(f"       • 标准差: {hand_array.std():.2f}")
        print(f"       • 各维度值:")
        for i, val in enumerate(hand_data):
            print(f"         [{i:2d}] = {val:.2f}")
    
    # 分析多帧数据的统计
    print(f"\n{'='*80}")
    print(f"全部 {len(frames)} 帧的统计分析")
    print(f"{'='*80}")
    
    # 收集所有帧的数据
    left_action = np.array([f['action']['left_hand'] for f in frames])
    right_action = np.array([f['action']['right_hand'] for f in frames])
    left_obs = np.array([f['observation']['left_hand'] for f in frames])
    right_obs = np.array([f['observation']['right_hand'] for f in frames])
    
    datasets = {
        'action.left_hand': left_action,
        'action.right_hand': right_action,
        'observation.left_hand': left_obs,
        'observation.right_hand': right_obs
    }
    
    for name, dataset in datasets.items():
        print(f"\n  📊 {name}:")
        print(f"     数据形状: {dataset.shape}  (帧数 × 维度)")
        print(f"     数据类型: {dataset.dtype}")
        print(f"     内存占用: {dataset.nbytes / 1024:.2f} KB")
        print(f"\n     各维度统计:")
        print(f"     {'维度':<6} {'最小值':>12} {'最大值':>12} {'均值':>12} {'标准差':>12} {'中位数':>12}")
        print(f"     {'-'*72}")
        for dim in range(dataset.shape[1]):
            dim_data = dataset[:, dim]
            print(f"     {dim:>4}   {dim_data.min():>12.4f} {dim_data.max():>12.4f} "
                  f"{dim_data.mean():>12.4f} {dim_data.std():>12.4f} {np.median(dim_data):>12.4f}")
    
    # 输出几个特定帧的完整数据
    print(f"\n{'='*80}")
    print("关键帧数据样本")
    print(f"{'='*80}")
    
    key_frames = [0, len(frames)//4, len(frames)//2, len(frames)*3//4, -1]
    for idx in key_frames:
        frame = frames[idx]
        actual_idx = idx if idx >= 0 else len(frames) + idx
        print(f"\n  帧 {actual_idx} (时间戳: {frame['t']:.4f}):")
        print(f"    action.left_hand:  {np.array(frame['action']['left_hand'])}")
        print(f"    action.right_hand: {np.array(frame['action']['right_hand'])}")
        print(f"    obs.left_hand:     {np.array(frame['observation']['left_hand'])}")
        print(f"    obs.right_hand:    {np.array(frame['observation']['right_hand'])}")


def create_summary_json():
    """创建 JSON 格式的详细总结"""
    print("\n\n" + "="*80)
    print("生成 JSON 格式总结")
    print("="*80)
    
    summary = {
        "dataset": "action176",
        "episode": "episode_0",
        "image_data": {
            "cameras": ["camera_head", "camera_left_wrist", "camera_right_wrist", "camera_third_view"],
            "num_cameras": 4,
            "frames_per_camera": 452,
            "total_images": 1808,
            "format": "JPEG",
            "resolution": {"width": 640, "height": 480},
            "color_mode": "RGB",
            "channels": 3,
            "dtype": "uint8",
            "pixel_range": [0, 255],
            "array_shape": [480, 640, 3],
            "shape_description": "height × width × channels",
            "avg_file_size_kb": 84,
            "memory_per_image_kb": 921.6  # 480*640*3 bytes
        },
        "episode_bson": {
            "file": "episode_0.bson",
            "size_mb": 0.59,
            "num_topics": 10,
            "data_points_per_topic": 452,
            "sampling_rate_hz": 19.82,
            "duration_seconds": 22.76,
            "topics": {
                "poses": {
                    "/observation/left_arm/pose": {
                        "translation": {"dims": 3, "description": "[x, y, z]"},
                        "rotation": {"dims": 4, "description": "[qx, qy, qz, qw] quaternion"}
                    },
                    "/observation/right_arm/pose": {
                        "translation": {"dims": 3, "description": "[x, y, z]"},
                        "rotation": {"dims": 4, "description": "[qx, qy, qz, qw] quaternion"}
                    }
                },
                "joint_states": {
                    "left_arm": {"dims": 6, "fields": ["pos", "vel", "eff"]},
                    "right_arm": {"dims": 6, "fields": ["pos", "vel", "eff"]},
                    "head": {"dims": 2, "fields": ["pos", "vel", "eff"]},
                    "spine": {"dims": 1, "fields": ["pos", "vel", "eff"]}
                }
            }
        },
        "xhand_control_bson": {
            "file": "xhand_control_data.bson",
            "size_mb": 0.28,
            "num_frames": 452,
            "sampling_rate_hz": 19.87,
            "duration_seconds": 22.75,
            "data_structure": {
                "action": {
                    "left_hand": {"dims": 12, "dtype": "float64", "description": "12维灵巧手动作指令"},
                    "right_hand": {"dims": 12, "dtype": "float64", "description": "12维灵巧手动作指令"}
                },
                "observation": {
                    "left_hand": {"dims": 12, "dtype": "float64", "description": "12维灵巧手传感器观测"},
                    "right_hand": {"dims": 12, "dtype": "float64", "description": "12维灵巧手传感器观测"}
                }
            },
            "array_shapes": {
                "action.left_hand": [452, 12],
                "action.right_hand": [452, 12],
                "observation.left_hand": [452, 12],
                "observation.right_hand": [452, 12]
            }
        }
    }
    
    output_path = "/home/chensiqi/chensiqi/RDT_libero_finetune/data/baai/data/detailed_data_summary.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✓ JSON 总结已保存至: {output_path}")
    print(f"\n  摘要内容:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    inspect_images_detailed()
    inspect_episode_bson_detailed()
    inspect_xhand_bson_detailed()
    create_summary_json()
    
    print("\n\n" + "="*80)
    print("✓ 详细分析完成！")
    print("="*80)

