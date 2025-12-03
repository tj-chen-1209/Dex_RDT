#!/usr/bin/env python3
"""读取LeRobot数据集的State和Action - siqi"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    USE_LEROBOT_API = True
except ImportError:
    USE_LEROBOT_API = False
    print("⚠️  lerobot未安装，将使用原生方法读取")


class LeRobotDataReader:
    """LeRobot数据集读取器"""
    
    def __init__(self, dataset_root: str):
        """
        初始化数据读取器
        
        Args:
            dataset_root: LeRobot数据集根目录
        """
        self.root = Path(dataset_root)
        self.meta_dir = self.root / "meta"
        self.data_dir = self.root / "data"
        
        # 加载元信息
        self.info = self._load_info()
        self.tasks = self._load_tasks()
        
        print(f"📂 数据集根目录: {self.root}")
        print(f"📊 总Episodes: {self.info['total_episodes']}")
        print(f"📊 总Frames: {self.info['total_frames']}")
        print(f"📋 任务数量: {self.info['total_tasks']}")
        
    def _load_info(self) -> dict:
        """加载info.json"""
        info_path = self.meta_dir / "info.json"
        with open(info_path, 'r') as f:
            return json.load(f)
    
    def _load_tasks(self) -> pd.DataFrame:
        """加载任务列表"""
        tasks_path = self.meta_dir / "tasks.parquet"
        if tasks_path.exists():
            return pd.read_parquet(tasks_path)
        return None
    
    def get_all_data(self) -> pd.DataFrame:
        """
        读取所有数据
        
        Returns:
            包含所有数据的DataFrame
        """
        if USE_LEROBOT_API:
            return self._get_all_data_via_lerobot()
        else:
            return self._get_all_data_native()
    
    def _get_all_data_via_lerobot(self) -> pd.DataFrame:
        """使用LeRobot API读取数据"""
        print("📖 使用LeRobot API读取数据...")
        dataset = LeRobotDataset(str(self.root))
        
        # 将整个数据集转换为DataFrame
        all_frames = []
        for idx in range(len(dataset)):
            frame = dataset[idx]
            # 将tensor转为numpy
            frame_dict = {}
            for key, value in frame.items():
                if hasattr(value, 'numpy'):
                    frame_dict[key] = value.numpy()
                else:
                    frame_dict[key] = value
            all_frames.append(frame_dict)
            
            if (idx + 1) % 500 == 0:
                print(f"  已读取 {idx + 1} / {len(dataset)} 帧")
        
        return pd.DataFrame(all_frames)
    
    def _get_all_data_native(self) -> pd.DataFrame:
        """使用原生方法读取数据"""
        all_data = []
        
        # 遍历所有chunk
        for chunk_dir in sorted(self.data_dir.glob("chunk-*")):
            # 读取chunk中的所有parquet文件
            for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
                print(f"📖 读取: {parquet_file.relative_to(self.root)}")
                df = pd.read_parquet(parquet_file)
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def get_episode_data(self, episode_idx: int) -> pd.DataFrame:
        """
        获取指定episode的数据
        
        Args:
            episode_idx: Episode索引
            
        Returns:
            该episode的所有帧数据
        """
        all_data = self.get_all_data()
        
        # 处理 episode_index 可能是数组的情况
        if 'episode_index' in all_data.columns:
            ep_indices = all_data['episode_index'].values
            if len(ep_indices) > 0 and isinstance(ep_indices[0], np.ndarray):
                # 如果是数组，提取标量值
                mask = [
                    (x.item() if x.size == 1 else x[0]) == episode_idx 
                    for x in ep_indices
                ]
                return all_data[mask]
            else:
                return all_data[all_data['episode_index'] == episode_idx]
        return pd.DataFrame()
    
    def get_state_action_columns(self) -> Dict[str, List[str]]:
        """
        获取state和action相关的列名
        
        Returns:
            包含state和action列名的字典
        """
        # 从features中提取
        features = self.info.get('features', {})
        
        state_cols = [k for k in features.keys() if k.startswith('observation.state')]
        action_cols = [k for k in features.keys() if k == 'action']
        image_cols = [k for k in features.keys() if 'images' in k]
        
        return {
            'state': state_cols,
            'action': action_cols,
            'images': image_cols,
        }
    
    def print_summary(self, df: pd.DataFrame):
        """
        打印数据摘要
        
        Args:
            df: 数据DataFrame
        """
        print("\n" + "="*70)
        print("📊 数据摘要")
        print("="*70)
        print(f"总行数: {len(df)}")
        print(f"列数: {len(df.columns)}")
        print(f"\n列名和数据类型:")
        for i, col in enumerate(df.columns, 1):
            # 获取第一个非空值来判断类型和维度
            sample = None
            for val in df[col]:
                if val is not None and (not isinstance(val, float) or not np.isnan(val)):
                    sample = val
                    break
            
            if sample is not None:
                if isinstance(sample, np.ndarray):
                    print(f"  {i:2d}. {col:50s} | array shape={sample.shape}, dtype={sample.dtype}")
                elif isinstance(sample, (list, tuple)):
                    print(f"  {col:50s} | list/tuple len={len(sample)}")
                else:
                    print(f"  {i:2d}. {col:50s} | scalar type={type(sample).__name__}")
            else:
                print(f"  {i:2d}. {col:50s} | (no data)")
        
        if 'episode_index' in df.columns:
            # 安全处理 episode_index（可能是数组）
            try:
                ep_indices = df['episode_index'].values
                if len(ep_indices) > 0 and isinstance(ep_indices[0], np.ndarray):
                    ep_indices = [x.item() if x.size == 1 else x[0] for x in ep_indices]
                min_ep = min(ep_indices)
                max_ep = max(ep_indices)
                unique_ep = len(set(ep_indices))
                print(f"\nEpisode范围: {min_ep} - {max_ep}")
                print(f"Episode数量: {unique_ep}")
            except Exception as e:
                print(f"\nEpisode信息: 无法解析 ({e})")
        
        if 'task' in df.columns:
            print(f"\n任务列表:")
            try:
                tasks = df['task'].values
                task_counts = {}
                for task in tasks:
                    task_str = task.item() if isinstance(task, np.ndarray) and task.size == 1 else str(task)
                    task_counts[task_str] = task_counts.get(task_str, 0) + 1
                
                for task, count in task_counts.items():
                    print(f"  - {task}: {count} 帧")
            except Exception as e:
                print(f"  无法解析任务列表 ({e})")
    
    def print_data_structure(self, df: pd.DataFrame):
        """
        详细打印数据结构和维度
        
        Args:
            df: 数据DataFrame
        """
        print("\n" + "="*70)
        print("🔍 数据结构详解")
        print("="*70)
        
        if len(df) == 0:
            print("⚠️  数据为空")
            return
        
        # 分类列
        state_cols = [c for c in df.columns if 'observation.state' in c]
        action_cols = [c for c in df.columns if c == 'action']
        image_cols = [c for c in df.columns if 'images' in c]
        meta_cols = [c for c in df.columns if c in ['episode_index', 'frame_index', 'task', 'timestamp', 'index', 'task_index']]
        
        # 显示State结构
        if state_cols:
            print("\n📍 State Fields (观测状态):")
            total_state_dim = 0
            for col in sorted(state_cols):
                sample = df[col].iloc[0]
                if isinstance(sample, np.ndarray):
                    dim = sample.shape[0] if len(sample.shape) > 0 else 1
                    total_state_dim += dim
                    print(f"  {col:50s} | dim={dim:2d}, dtype={sample.dtype}, range=[{sample.min():.3f}, {sample.max():.3f}]")
                else:
                    print(f"  {col:50s} | scalar: {sample}")
            print(f"\n  ✅ Total State Dimension: {total_state_dim}")
        
        # 显示Action结构
        if action_cols:
            print("\n🎯 Action Fields (动作):")
            for col in action_cols:
                sample = df[col].iloc[0]
                if isinstance(sample, np.ndarray):
                    dim = sample.shape[0] if len(sample.shape) > 0 else 1
                    print(f"  {col:50s} | dim={dim:2d}, dtype={sample.dtype}, range=[{sample.min():.3f}, {sample.max():.3f}]")
                else:
                    print(f"  {col:50s} | scalar: {sample}")
        
        # 显示Image结构
        if image_cols:
            print("\n📷 Image Fields (图像):")
            for col in sorted(image_cols):
                sample = df[col].iloc[0]
                if isinstance(sample, np.ndarray):
                    print(f"  {col:50s} | shape={sample.shape}, dtype={sample.dtype}")
                else:
                    print(f"  {col:50s} | type={type(sample)}")
        
        # 显示Meta结构
        if meta_cols:
            print("\n📋 Meta Fields (元数据):")
            for col in sorted(meta_cols):
                sample = df[col].iloc[0]
                if isinstance(sample, np.ndarray):
                    val = sample.item() if sample.size == 1 else sample
                    print(f"  {col:50s} | value={val}")
                else:
                    print(f"  {col:50s} | value={sample}")
    
    def print_first_frame_detail(self, df: pd.DataFrame, num_frames: int = 1):
        """
        详细打印前N帧的所有数据（维度和内容）
        
        Args:
            df: 数据DataFrame
            num_frames: 显示的帧数（默认1）
        """
        if len(df) == 0:
            print("⚠️  数据为空")
            return
        
        num_frames = min(num_frames, len(df))  # 不超过数据总帧数
        
        for frame_idx in range(num_frames):
            print("\n" + "="*70)
            print(f"🔬 第 {frame_idx + 1} 帧完整数据详解")
            print("="*70)
            
            first_frame = df.iloc[frame_idx]
            
            # 分类字段
            state_fields = [k for k in first_frame.index if 'observation.state' in k]
            action_fields = [k for k in first_frame.index if k == 'action']
            image_fields = [k for k in first_frame.index if 'images' in k]
            meta_fields = [k for k in first_frame.index if k in ['episode_index', 'frame_index', 'task', 'timestamp', 'index', 'task_index']]
            
            # 1. 显示元数据
            if meta_fields:
                print("\n📋 元数据 (Metadata):")
                for field in sorted(meta_fields):
                    value = first_frame[field]
                    if isinstance(value, np.ndarray):
                        val = value.item() if value.size == 1 else value
                        print(f"  {field:30s} = {val}")
                    else:
                        print(f"  {field:30s} = {value}")
            
            # 2. 显示状态数据
            if state_fields:
                print("\n📍 观测状态 (Observation State):")
                total_dim = 0
                for field in sorted(state_fields):
                    value = first_frame[field]
                    if isinstance(value, np.ndarray):
                        dim = value.shape[0] if len(value.shape) > 0 else 1
                        total_dim += dim
                        print(f"\n  {field}")
                        print(f"    维度 (shape):  {value.shape}")
                        print(f"    类型 (dtype):  {value.dtype}")
                        print(f"    范围 (range):  [{value.min():.6f}, {value.max():.6f}]")
                        print(f"    内容 (values): {value}")
                    else:
                        print(f"  {field:30s} = {value}")
                print(f"\n  ✅ 总状态维度: {total_dim}")
            
            # 3. 显示动作数据
            if action_fields:
                print("\n🎯 动作 (Action):")
                for field in action_fields:
                    value = first_frame[field]
                    if isinstance(value, np.ndarray):
                        print(f"\n  {field}")
                        print(f"    维度 (shape):  {value.shape}")
                        print(f"    类型 (dtype):  {value.dtype}")
                        print(f"    范围 (range):  [{value.min():.6f}, {value.max():.6f}]")
                        print(f"    内容 (values): {value}")
                    else:
                        print(f"  {field:30s} = {value}")
            
            # 4. 显示图像数据（只显示维度，不显示全部像素）
            if image_fields:
                print("\n📷 图像 (Images):")
                for field in sorted(image_fields):
                    value = first_frame[field]
                    if isinstance(value, np.ndarray):
                        print(f"\n  {field}")
                        print(f"    维度 (shape):     {value.shape}")
                        print(f"    类型 (dtype):     {value.dtype}")
                        print(f"    像素范围 (range): [{value.min():.1f}, {value.max():.1f}]")
                        print(f"    均值 (mean):      {value.mean():.3f}")
                        print(f"    前3x3像素预览:")
                        if len(value.shape) == 3:
                            # CHW 或 HWC 格式
                            if value.shape[0] in [1, 3, 4]:  # CHW
                                print(f"      (注意: 数据格式为 CHW - Channel, Height, Width)")
                                print(f"      第1通道前3x3: \n{value[0, :3, :3]}")
                            else:  # HWC
                                print(f"      (注意: 数据格式为 HWC - Height, Width, Channel)")
                                print(f"      前3x3像素RGB: \n{value[:3, :3, :]}")
                    else:
                        print(f"  {field:30s} type={type(value)}")
            
            print("\n" + "="*70)
    
    def print_state_action_sample(self, df: pd.DataFrame, num_samples: int = 3):
        """
        打印state和action样本
        
        Args:
            df: 数据DataFrame
            num_samples: 打印样本数量
        """
        cols_info = self.get_state_action_columns()
        
        print("\n" + "="*70)
        print("🔍 State & Action 样本数据")
        print("="*70)
        
        for idx in range(min(num_samples, len(df))):
            row = df.iloc[idx]
            print(f"\n【样本 {idx+1}】")
            print(f"Episode: {row.get('episode_index', 'N/A')}, Frame: {row.get('frame_index', 'N/A')}")
            if 'task' in row:
                print(f"Task: {row['task']}")
            
            print("\n  📍 State:")
            for state_col in cols_info['state']:
                if state_col in row:
                    value = row[state_col]
                    if isinstance(value, (list, np.ndarray)):
                        print(f"    {state_col:50s}: shape={np.array(value).shape}, sample={np.array(value)[:3]}...")
                    else:
                        print(f"    {state_col:50s}: {value}")
            
            print("\n  🎯 Action:")
            for action_col in cols_info['action']:
                if action_col in row:
                    value = row[action_col]
                    if isinstance(value, (list, np.ndarray)):
                        print(f"    {action_col:50s}: shape={np.array(value).shape}, sample={np.array(value)[:3]}...")
                    else:
                        print(f"    {action_col:50s}: {value}")
            
            print("-" * 70)
    
    def export_to_numpy(self, episode_idx: Optional[int] = None, 
                       output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        导出为numpy数组
        
        Args:
            episode_idx: Episode索引（None表示所有数据）
            output_dir: 输出目录（None表示不保存文件）
            
        Returns:
            包含state和action数组的字典
        """
        if episode_idx is not None:
            df = self.get_episode_data(episode_idx)
            print(f"📤 导出Episode {episode_idx}")
        else:
            df = self.get_all_data()
            print(f"📤 导出所有数据")
        
        cols_info = self.get_state_action_columns()
        
        # 提取state和action
        result = {}
        
        # 合并所有state列
        state_arrays = []
        for state_col in sorted(cols_info['state']):
            if state_col in df.columns:
                arr = np.stack(df[state_col].values)
                state_arrays.append(arr)
                print(f"  State '{state_col}': {arr.shape}")
        
        if state_arrays:
            result['state'] = np.concatenate(state_arrays, axis=-1)
            print(f"✅ 合并后State shape: {result['state'].shape}")
        
        # 提取action
        for action_col in cols_info['action']:
            if action_col in df.columns:
                result['action'] = np.stack(df[action_col].values)
                print(f"✅ Action shape: {result['action'].shape}")
        
        # 其他元数据
        if 'episode_index' in df.columns:
            result['episode_index'] = df['episode_index'].values
        if 'frame_index' in df.columns:
            result['frame_index'] = df['frame_index'].values
        if 'task' in df.columns:
            result['task'] = df['task'].values
        
        # 保存到文件
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    filename = f"episode_{episode_idx}_{key}.npy" if episode_idx is not None else f"all_{key}.npy"
                    np.save(output_path / filename, value)
                    print(f"💾 已保存: {filename}")
        
        return result


def main():
    """主函数 - 演示如何使用"""
    
    # 数据集路径
    dataset_root = "/home/zhukefei/chensiqi/Dex_RDT/data/baai/data/lerobot_baai"
    
    print("🚀 LeRobot数据集读取器")
    print("="*70)
    
    # 初始化读取器
    reader = LeRobotDataReader(dataset_root)
    
    # 1. 读取所有数据
    print("\n【1】读取所有数据...")
    all_data = reader.get_all_data()
    reader.print_summary(all_data)
    
    # 2. 打印前2帧完整数据
    print("\n【2】打印前2帧完整数据...")
    reader.print_first_frame_detail(all_data, num_frames=2)
    
    # 3. 打印数据结构
    print("\n【3】打印数据结构...")
    reader.print_data_structure(all_data)
    
    # 4. 打印样本
    print("\n【4】打印State和Action样本...")
    reader.print_state_action_sample(all_data, num_samples=2)
    
    # 5. 读取特定episode
    if len(all_data) > 0 and 'episode_index' in all_data.columns:
        ep_idx_sample = all_data['episode_index'].iloc[0]
        if isinstance(ep_idx_sample, np.ndarray):
            first_episode = ep_idx_sample.item() if ep_idx_sample.size == 1 else ep_idx_sample[0]
        else:
            first_episode = ep_idx_sample
            
        print(f"\n【5】读取Episode {first_episode}...")
        episode_data = reader.get_episode_data(first_episode)
        print(f"Episode {first_episode} 包含 {len(episode_data)} 帧")
        
        # 6. 导出为numpy
        print(f"\n【6】导出Episode {first_episode}为numpy数组...")
        numpy_data = reader.export_to_numpy(episode_idx=first_episode)
        
        print("\n📦 导出的数据:")
        for key, value in numpy_data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key:20s}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key:20s}: {type(value)}")
    
    # 7. 显示任务信息
    if reader.tasks is not None and len(reader.tasks) > 0:
        print("\n【7】任务信息:")
        print(reader.tasks)
    
    print("\n" + "="*70)
    print("✅ 完成!")


if __name__ == "__main__":
    main()

