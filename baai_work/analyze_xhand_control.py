#!/usr/bin/env python3
"""
分析 xhand_control_data.bson 文件的详细结构
"""
import os
from bson import decode_all
import numpy as np
import json
from pprint import pprint

def analyze_bson_structure(data, name="", indent=0, max_items=3):
    """递归分析BSON数据结构"""
    prefix = "  " * indent
    
    if isinstance(data, dict):
        print(f"{prefix}{name}(dict) - Keys: {list(data.keys())}")
        for key, value in data.items():
            analyze_bson_structure(value, name=f"['{key}']", indent=indent+1, max_items=max_items)
    
    elif isinstance(data, list):
        print(f"{prefix}{name}(list) - Length: {len(data)}")
        if len(data) > 0:
            print(f"{prefix}  First item type: {type(data[0])}")
            # 只展示前几个元素
            for i in range(min(max_items, len(data))):
                print(f"{prefix}  Item [{i}]:")
                analyze_bson_structure(data[i], name="", indent=indent+2, max_items=max_items)
            if len(data) > max_items:
                print(f"{prefix}  ... ({len(data) - max_items} more items)")
    
    elif isinstance(data, (bytes, bytearray)):
        print(f"{prefix}{name}(bytes) - Length: {len(data)}")
        # 尝试转换为numpy数组
        try:
            arr = np.frombuffer(data, dtype=np.float64)
            print(f"{prefix}  As float64 array: shape={arr.shape}, values={arr[:5]}...")
        except:
            print(f"{prefix}  Raw bytes (first 20): {data[:20]}")
    
    elif isinstance(data, np.ndarray):
        print(f"{prefix}{name}(numpy.ndarray) - Shape: {data.shape}, dtype: {data.dtype}")
        if data.size <= 10:
            print(f"{prefix}  Values: {data}")
        else:
            print(f"{prefix}  Sample values: {data.flat[:10]}...")
    
    else:
        # 基本类型
        value_str = str(data)
        if len(value_str) > 100:
            value_str = value_str[:100] + "..."
        print(f"{prefix}{name}({type(data).__name__}) = {value_str}")


def main():
    episode_dir = "/home/chensiqi/chensiqi/RDT_libero_finetune/data/baai/data/action176/episode_0"
    xhand_path = os.path.join(episode_dir, "xhand_control_data.bson")
    
    print("=" * 80)
    print("分析 xhand_control_data.bson 文件")
    print("=" * 80)
    
    # 读取BSON文件
    with open(xhand_path, "rb") as f:
        bson_data = f.read()
        docs = decode_all(bson_data)
    
    # 分析每个文档
    for doc_idx, doc in enumerate(docs):
        print(f"\n【dic {doc_idx}】")
        print("-" * 80)
        
        # 顶层结构
        print(f"顶层Keys: {list(docs[0].keys())}")
        print()
        
        # 详细分析每个顶层字段
        for key in doc.keys():
            print(f"\n## 字段: '{key}'")
            print("-" * 40)
            value = doc[key]
            
            if key == "frames" and isinstance(value, list):
                print(f"类型: list")
                print(f"帧数量: {len(value)}")
                
                if len(value) > 0:
                    print(f"\n### 第一帧的结构 (frames[0]):")
                    first_frame = value[0]
                    if isinstance(first_frame, dict):
                        print(f"Keys: {list(first_frame.keys())}")
                        print()
                        
                        for frame_key in first_frame.keys():
                            frame_value = first_frame[frame_key]
                            print(f"  - {frame_key}:")
                            
                            if isinstance(frame_value, (bytes, bytearray)):
                                print(f"      类型: bytes")
                                print(f"      长度: {len(frame_value)} bytes")
                                
                                # 尝试不同的数据类型解析
                                print(f"      尝试解析为numpy数组:")
                                for dtype in [np.float64, np.float32, np.int32, np.int64]:
                                    try:
                                        arr = np.frombuffer(frame_value, dtype=dtype)
                                        print(f"        - {dtype.__name__}: shape={arr.shape}, sample={arr[:5]}")
                                    except Exception as e:
                                        print(f"        - {dtype.__name__}: 解析失败")
                            
                            elif isinstance(frame_value, np.ndarray):
                                print(f"      类型: numpy.ndarray")
                                print(f"      Shape: {frame_value.shape}")
                                print(f"      Dtype: {frame_value.dtype}")
                                print(f"      值: {frame_value}")
                            
                            elif isinstance(frame_value, (int, float)):
                                print(f"      类型: {type(frame_value).__name__}")
                                print(f"      值: {frame_value}")
                            
                            elif isinstance(frame_value, str):
                                print(f"      类型: str")
                                print(f"      值: {frame_value}")
                            
                            elif isinstance(frame_value, dict):
                                print(f"      类型: dict")
                                print(f"      Keys: {list(frame_value.keys())}")
                            
                            else:
                                print(f"      类型: {type(frame_value).__name__}")
                                print(f"      值: {frame_value}")
                    
                    # 展示最后一帧
                    print(f"\n### 最后一帧的结构 (frames[-1]):")
                    last_frame = value[-1]
                    if isinstance(last_frame, dict):
                        print(f"Keys: {list(last_frame.keys())}")
                        for frame_key in last_frame.keys():
                            frame_value = last_frame[frame_key]
                            if isinstance(frame_value, (bytes, bytearray)):
                                try:
                                    arr = np.frombuffer(frame_value, dtype=np.float64)
                                    print(f"  - {frame_key}: bytes -> float64 array, shape={arr.shape}, sample={arr[:3]}")
                                except:
                                    print(f"  - {frame_key}: bytes, length={len(frame_value)}")
                            else:
                                print(f"  - {frame_key}: {type(frame_value).__name__} = {frame_value}")
                    
                    # 统计所有帧的数据
                    print(f"\n### 帧数据统计:")
                    if isinstance(value[0], dict):
                        for frame_key in value[0].keys():
                            values_list = []
                            for frame in value:
                                frame_value = frame.get(frame_key)
                                if isinstance(frame_value, (bytes, bytearray)):
                                    try:
                                        arr = np.frombuffer(frame_value, dtype=np.float64)
                                        values_list.append(arr)
                                    except:
                                        pass
                                elif isinstance(frame_value, (int, float)):
                                    values_list.append(frame_value)
                            
                            if values_list:
                                if isinstance(values_list[0], np.ndarray):
                                    all_values = np.array(values_list)
                                    print(f"  - {frame_key}:")
                                    print(f"      总shape: {all_values.shape}")
                                    print(f"      范围: [{np.min(all_values):.4f}, {np.max(all_values):.4f}]")
                                    print(f"      均值: {np.mean(all_values):.4f}")
                                    print(f"      标准差: {np.std(all_values):.4f}")
                                else:
                                    all_values = np.array(values_list)
                                    print(f"  - {frame_key}:")
                                    print(f"      范围: [{np.min(all_values):.4f}, {np.max(all_values):.4f}]")
            
            else:
                # 非frames字段
                if isinstance(value, dict):
                    print(f"类型: dict")
                    print(f"Keys: {list(value.keys())}")
                    print("内容:")
                    pprint(value, indent=2)
                elif isinstance(value, list):
                    print(f"类型: list")
                    print(f"长度: {len(value)}")
                    if len(value) > 0:
                        print(f"第一个元素: {value[0]}")
                        if len(value) > 1:
                            print(f"最后一个元素: {value[-1]}")
                elif isinstance(value, (bytes, bytearray)):
                    print(f"类型: bytes")
                    print(f"长度: {len(value)}")
                else:
                    print(f"类型: {type(value).__name__}")
                    print(f"值: {value}")


if __name__ == "__main__":
    main()

