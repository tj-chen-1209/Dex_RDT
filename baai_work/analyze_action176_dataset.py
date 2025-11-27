#!/usr/bin/env python3
"""
分析 action176 数据集的格式、属性和 IO 信息
"""
import os
import bson
import json
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
import sys


def analyze_bson_file(bson_path):
    """分析 BSON 文件内容"""
    try:
        with open(bson_path, 'rb') as f:
            data = bson.decode_all(f.read())
        
        info = {
            'file_size_mb': os.path.getsize(bson_path) / (1024 * 1024),
            'num_documents': len(data),
            'documents': []
        }
        
        for i, doc in enumerate(data):
            doc_info = {
                'index': i,
                'keys': list(doc.keys()) if isinstance(doc, dict) else None,
                'sample_data': {}
            }
            
            if isinstance(doc, dict):
                for key, value in doc.items():
                    if isinstance(value, (list, np.ndarray)):
                        doc_info['sample_data'][key] = {
                            'type': str(type(value).__name__),
                            'length': len(value),
                            'shape': np.array(value).shape if hasattr(value, '__len__') else None,
                            'dtype': str(np.array(value).dtype) if hasattr(value, '__len__') else None,
                            'sample_values': str(value[:3]) if len(value) > 0 else None
                        }
                    else:
                        doc_info['sample_data'][key] = {
                            'type': str(type(value).__name__),
                            'value': str(value) if not isinstance(value, bytes) else f'<bytes: {len(value)} bytes>'
                        }
            
            info['documents'].append(doc_info)
        
        return info
    except Exception as e:
        return {'error': str(e)}


def analyze_images_in_camera_dir(camera_dir):
    """分析相机目录中的图像"""
    if not os.path.exists(camera_dir):
        return None
    
    image_files = sorted([f for f in os.listdir(camera_dir) if f.endswith('.jpg')])
    
    if not image_files:
        return {'num_images': 0}
    
    # 分析第一张图像
    first_image_path = os.path.join(camera_dir, image_files[0])
    try:
        img = Image.open(first_image_path)
        img_array = np.array(img)
        
        info = {
            'num_images': len(image_files),
            'image_format': 'JPEG',
            'resolution': img.size,  # (width, height)
            'mode': img.mode,
            'channels': img_array.shape[2] if len(img_array.shape) == 3 else 1,
            'dtype': str(img_array.dtype),
            'first_image': image_files[0],
            'last_image': image_files[-1],
            'avg_file_size_kb': np.mean([
                os.path.getsize(os.path.join(camera_dir, f)) / 1024 
                for f in image_files[:10]  # 只检查前10张以提高速度
            ])
        }
        return info
    except Exception as e:
        return {'error': str(e), 'num_images': len(image_files)}


def analyze_episode(episode_dir):
    """分析单个 episode 目录"""
    print(f"\n分析 episode: {os.path.basename(episode_dir)}")
    
    episode_info = {
        'path': episode_dir,
        'cameras': {},
        'bson_files': {}
    }
    
    # 分析相机目录
    camera_dirs = ['camera_head', 'camera_left_wrist', 'camera_right_wrist', 'camera_third_view']
    for camera in camera_dirs:
        camera_path = os.path.join(episode_dir, camera)
        if os.path.exists(camera_path):
            print(f"  分析 {camera}...")
            episode_info['cameras'][camera] = analyze_images_in_camera_dir(camera_path)
    
    # 分析 BSON 文件
    bson_files = [f for f in os.listdir(episode_dir) if f.endswith('.bson') and 'backup' not in f]
    for bson_file in bson_files:
        bson_path = os.path.join(episode_dir, bson_file)
        print(f"  分析 BSON 文件: {bson_file}...")
        episode_info['bson_files'][bson_file] = analyze_bson_file(bson_path)
    
    return episode_info


def analyze_dataset(dataset_root):
    """分析整个数据集"""
    print(f"开始分析数据集: {dataset_root}\n")
    print("=" * 80)
    
    # 获取所有 episode 目录
    episode_dirs = sorted([
        os.path.join(dataset_root, d) 
        for d in os.listdir(dataset_root) 
        if d.startswith('episode_') and os.path.isdir(os.path.join(dataset_root, d))
    ])
    
    dataset_info = {
        'dataset_root': dataset_root,
        'total_episodes': len(episode_dirs),
        'episodes_analyzed': [],
        'summary': {
            'total_images_per_camera': defaultdict(int),
            'total_bson_size_mb': 0,
            'camera_resolutions': set(),
        }
    }
    
    # 分析前3个 episode 作为样本
    sample_episodes = episode_dirs[:3]
    
    for ep_dir in sample_episodes:
        ep_info = analyze_episode(ep_dir)
        dataset_info['episodes_analyzed'].append(ep_info)
        
        # 更新统计信息
        for camera, cam_info in ep_info['cameras'].items():
            if cam_info and 'num_images' in cam_info:
                dataset_info['summary']['total_images_per_camera'][camera] += cam_info['num_images']
                if 'resolution' in cam_info:
                    dataset_info['summary']['camera_resolutions'].add(tuple(cam_info['resolution']))
        
        for bson_file, bson_info in ep_info['bson_files'].items():
            if 'file_size_mb' in bson_info:
                dataset_info['summary']['total_bson_size_mb'] += bson_info['file_size_mb']
    
    # 转换 set 为 list 以便 JSON 序列化
    dataset_info['summary']['camera_resolutions'] = list(dataset_info['summary']['camera_resolutions'])
    dataset_info['summary']['total_images_per_camera'] = dict(dataset_info['summary']['total_images_per_camera'])
    
    return dataset_info


def print_summary(dataset_info):
    """打印数据集摘要"""
    print("\n" + "=" * 80)
    print("数据集摘要")
    print("=" * 80)
    
    print(f"\n数据集路径: {dataset_info['dataset_root']}")
    print(f"总 episodes 数: {dataset_info['total_episodes']}")
    print(f"已分析 episodes 数: {len(dataset_info['episodes_analyzed'])}")
    
    print("\n" + "-" * 80)
    print("相机配置:")
    print("-" * 80)
    for camera, count in dataset_info['summary']['total_images_per_camera'].items():
        print(f"  {camera}: {count} 张图像 (前3个episodes)")
    
    if dataset_info['summary']['camera_resolutions']:
        print(f"\n图像分辨率: {dataset_info['summary']['camera_resolutions']}")
    
    print("\n" + "-" * 80)
    print("BSON 文件:")
    print("-" * 80)
    print(f"  总大小 (前3个episodes): {dataset_info['summary']['total_bson_size_mb']:.2f} MB")
    
    # 打印第一个 episode 的详细信息
    if dataset_info['episodes_analyzed']:
        first_ep = dataset_info['episodes_analyzed'][0]
        print("\n" + "-" * 80)
        print(f"Episode 0 详细信息:")
        print("-" * 80)
        
        for camera, cam_info in first_ep['cameras'].items():
            if cam_info and 'num_images' in cam_info:
                print(f"\n  {camera}:")
                print(f"    图像数量: {cam_info['num_images']}")
                if 'resolution' in cam_info:
                    print(f"    分辨率: {cam_info['resolution'][0]}x{cam_info['resolution'][1]}")
                    print(f"    颜色模式: {cam_info.get('mode', 'N/A')}")
                    print(f"    通道数: {cam_info.get('channels', 'N/A')}")
                    print(f"    数据类型: {cam_info.get('dtype', 'N/A')}")
                    print(f"    平均文件大小: {cam_info.get('avg_file_size_kb', 0):.2f} KB")
        
        print("\n  BSON 文件:")
        for bson_name, bson_info in first_ep['bson_files'].items():
            print(f"\n    {bson_name}:")
            if 'error' not in bson_info:
                print(f"      文件大小: {bson_info.get('file_size_mb', 0):.2f} MB")
                print(f"      文档数量: {bson_info.get('num_documents', 0)}")
                
                # 打印第一个文档的结构
                if bson_info.get('documents'):
                    first_doc = bson_info['documents'][0]
                    print(f"      文档结构:")
                    if first_doc.get('keys'):
                        print(f"        字段: {', '.join(first_doc['keys'])}")
                    
                    if first_doc.get('sample_data'):
                        print(f"      字段详情:")
                        for key, value_info in first_doc['sample_data'].items():
                            if isinstance(value_info, dict):
                                if 'shape' in value_info:
                                    print(f"        {key}: {value_info['type']}, shape={value_info['shape']}, dtype={value_info['dtype']}")
                                else:
                                    print(f"        {key}: {value_info}")
            else:
                print(f"      错误: {bson_info['error']}")


if __name__ == "__main__":
    dataset_root = "/home/chensiqi/chensiqi/RDT_libero_finetune/data/baai/data/action176"
    
    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
    
    # 分析数据集
    dataset_info = analyze_dataset(dataset_root)
    
    # 打印摘要
    print_summary(dataset_info)
    
    # 保存完整报告到 JSON 文件
    output_file = os.path.join(os.path.dirname(dataset_root), 'action176_dataset_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n完整分析报告已保存至: {output_file}")
    print("=" * 80)

