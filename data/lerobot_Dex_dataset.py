import os
import json
import yaml
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from configs.state_vec import STATE_VEC_IDX_MAPPING

class LerobotDexDataset:
    """
    This class is used to sample episodes from the lerobot dataset.
    由于lerobot格式将图片存储在视频中,需要额外的库来解码。
    
    本加载器使用预处理的缓存文件(.pt格式)来避免依赖额外的库。
    如需生成缓存,请在有lerobot库的环境中运行: preprocess_lerobot_cache.py
    """

    def __init__(self, dataset_path="data/baai/data/lerobot_baai", use_cache=True) -> None:
        """
        Initialize the lerobot dataset loader.
        
        Args:
            dataset_path: Path to the lerobot dataset directory
            use_cache: Whether to use cached .pt files (recommended)
        """
        # print("="*70)
        # print("🚀 初始化 LerobotDexDataset")
        # print("="*70)
        
        self.DATASET_PATH = dataset_path
        self.DATASET_NAME = "baai"  # 使用与BsonDexDataset相同的名称，因为是同一数据集
        self.use_cache = use_cache
        
        # print(f"📂 数据集路径: {self.DATASET_PATH}")
        # print(f"💾 使用缓存模式: {use_cache}")
        
        # Load dataset info
        info_path = Path(dataset_path) / "meta" / "info.json"
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        # print(f"📊 总Episodes: {self.info['total_episodes']}")
        # print(f"📊 总Frames: {self.info['total_frames']}")
        # print(f"📊 FPS: {self.info['fps']}")
        
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
        
        # print(f"⚙️  配置: CHUNK_SIZE={self.CHUNK_SIZE}, IMG_HISTORY_SIZE={self.IMG_HISORY_SIZE}, STATE_DIM={self.STATE_DIM}")
        
        # Load instruction embeddings
        instruction_path = Path(dataset_path) / "instruction.pt"
        if instruction_path.exists():
            # print(f"📝 正在加载instruction embeddings...")
            self.instruction_embeddings = torch.load(instruction_path)
            # print(f"   ✅ shape={self.instruction_embeddings.shape}")
        else:
            self.instruction_embeddings = None
            # print(f"⚠️  未找到instruction.pt文件")
        
        if use_cache:
            # Load from cache
            # print("📦 正在加载缓存数据...")
            self._load_from_cache()
        else:
            raise NotImplementedError(
                "不使用缓存模式需要安装lerobot库。\n"
                "请运行 preprocess_lerobot_cache.py 来生成缓存文件，\n"
                "或在有lerobot的环境中初始化数据集。"
            )
        
        # print(f"✅ 数据集初始化完成！共 {len(self.episode_data)} 个episodes")
        # print("="*70)

    def _load_from_cache(self):
        """Load preprocessed cache files."""
        cache_dir = Path(self.DATASET_PATH) / "cache"
        
        if not cache_dir.exists():
            raise FileNotFoundError(
                f"缓存目录不存在: {cache_dir}\n"
                f"请先运行 preprocess_lerobot_cache.py 生成缓存文件"
            )
        
        # Load episode metadata
        episode_meta_path = cache_dir / "episode_metadata.pt"
        if not episode_meta_path.exists():
            raise FileNotFoundError(
                f"未找到episode元数据: {episode_meta_path}\n"
                f"请运行 preprocess_lerobot_cache.py 生成缓存"
            )
        
        # print(f"  读取: {episode_meta_path.relative_to(self.DATASET_PATH)}")
        
        # 修复numpy模块路径兼容性
        import sys
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
        sys.modules['numpy._core.numeric'] = np.core.numeric if hasattr(np.core, 'numeric') else np.core
        
        cache_data = torch.load(episode_meta_path, map_location='cpu')
        self.episode_data = cache_data['episode_data']
        episode_lens = cache_data['episode_lens']
        
        # print(f"  ✅ 加载了 {len(self.episode_data)} 个episode的元数据")
        # print(f"  📊 Episode长度: min={min(episode_lens)}, max={max(episode_lens)}, mean={np.mean(episode_lens):.1f}")
        
        # Calculate sampling weights
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
        
        # Check if cached episode data exists
        self.cache_dir = cache_dir
        sample_ep_file = cache_dir / f"episode_{self.episode_data[0]['episode_idx']:06d}.pt"
        if sample_ep_file.exists():
            # print(f"  ✅ Episode缓存文件已准备就绪")
            pass
        else:
            pass  # print(f"  ⚠️  警告: 未找到episode缓存文件 {sample_ep_file}")

    def _load_episode_cache(self, episode_idx):
        """
        Load cached episode data.
        
        Args:
            episode_idx: Episode index
            
        Returns:
            dict: Cached episode data containing state, action, images_info
        """
        cache_file = self.cache_dir / f"episode_{episode_idx:06d}.pt"
        
        if not cache_file.exists():
            raise FileNotFoundError(f"未找到episode缓存: {cache_file}")
        
        # print(f"  📦 加载缓存: episode_{episode_idx:06d}.pt")
        
        # 兼容不同numpy版本的加载
        import sys
        import pickle
        
        # 修复numpy模块路径兼容性
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
        sys.modules['numpy._core.numeric'] = np.core.numeric if hasattr(np.core, 'numeric') else np.core
        
        try:
            episode_cache = torch.load(cache_file, map_location='cpu')
        except Exception as e:
            # print(f"    ⚠️  标准加载失败，尝试兼容性加载...")
            # 尝试使用weights_only=False
            episode_cache = torch.load(cache_file, map_location='cpu', weights_only=False)
        
        # print(f"    State shape: {episode_cache['state'].shape}")
        # print(f"    Action shape: {episode_cache['action'].shape}")
        # print(f"    帧数: {episode_cache['frame_num']}")
        
        return episode_cache

    def __len__(self):
        return len(self.episode_data)

    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def _load_image_from_cache(self, images_info, camera_key, frame_idx):
        """
        Load image from cached numpy arrays.
        
        Args:
            images_info: Images info dict from cached episode
            camera_key: Camera key like 'camera_head'
            frame_idx: Frame index
            
        Returns:
            np.ndarray: Image array (H, W, 3)
        """
        try:
            if camera_key not in images_info:
                # print(f"      ⚠️  未找到相机: {camera_key}")
                return np.zeros((480, 640, 3), dtype=np.uint8)
            
            cam_data = images_info[camera_key]
            
            if isinstance(cam_data, np.ndarray):
                # All frames stored as array
                if frame_idx < len(cam_data):
                    return cam_data[frame_idx].astype(np.uint8)
            elif isinstance(cam_data, dict) and 'type' in cam_data:
                # File-based storage (lazy loading)
                if cam_data['type'] == 'file_sequence':
                    img_dir = cam_data['path']
                    img_file = cam_data['files'][frame_idx]
                    img_path = os.path.join(img_dir, img_file)
                    
                    with Image.open(img_path) as img:
                        img_array = np.array(img)
                    if img_array.ndim == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    return img_array.astype(np.uint8)
            
            # print(f"      ⚠️  无法加载图像")
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        except Exception as e:
            # print(f"      ⚠️  加载图像失败 {camera_key} frame {frame_idx}: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_item(self, index: int = None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
            sample (dict): a dictionary containing the training sample.
        """
        # print("\n" + "="*70)
        # print("🎲 采样训练数据")
        # print("="*70)
        
        while True:
            if index is None:
                episode_idx = np.random.choice(
                    len(self.episode_data), p=self.episode_sample_weights)
                episode_info = self.episode_data[episode_idx]
                # print(f"🎯 随机选择Episode {episode_info['episode_idx']} (内部索引: {episode_idx})")
            else:
                episode_info = self.episode_data[index]
                # print(f"🎯 使用指定Episode {episode_info['episode_idx']} (内部索引: {index})")
            
            # Parse episode based on state_only flag
            if state_only:
                valid, sample = self.parse_lerobot_episode_state_only(episode_info)
            else:
                valid, sample = self.parse_lerobot_episode(episode_info)
            
            if valid:
                # print("✅ 采样成功！")
                # print("="*70)
                return sample
            else:
                if index is None:
                    # print(f"⚠️  Episode无效，重新采样...")
                    continue
                else:
                    raise RuntimeError(f"Episode at index {index} is invalid")

    def parse_lerobot_episode(self, episode_info):
        """
        Parse a lerobot episode to generate a training sample at a random timestep.

        Args:
            episode_info (dict): Episode information dict
            
        Returns:
            valid (bool): whether the episode is valid
            dict: a dictionary containing the training sample
        """
        # Load episode cache
        episode_idx = episode_info['episode_idx']
        episode_cache = self._load_episode_cache(episode_idx)
        
        qpos = episode_cache["state"]
        num_steps = episode_cache["frame_num"]
        
        # print(f"\n  🔍 处理Episode数据...")
        # print(f"    总步数: {num_steps}")

        # Skip the first few still steps
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            # print(f"  ❌ 未找到运动起始点（所有qpos变化都小于{EPS}）")
            return False, None
        
        # print(f"    运动起始索引: {first_idx}")

        if first_idx >= num_steps:
            # print(f"  ❌ 起始索引超出范围")
            return False, None

        # Randomly sample a timestep
        step_id = np.random.randint(first_idx-1, num_steps)
        # print(f"    随机采样步数: {step_id}")
        
        # Get instruction
        if self.instruction_embeddings is not None and episode_idx < len(self.instruction_embeddings):
            instruction = self.instruction_embeddings[episode_idx]
            # print(f"    Instruction: embedding shape={instruction.shape}")
        else:
            instruction = "Use the left hand to hook the book '皮囊' from the pile of books,then use the right hand to place it on the right bookshelf."
            # print(f"    Instruction: 使用默认文本")

        # Assemble the meta
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": instruction
        }

        def fill_in_state(values):
            """Fill 36-dim state/action into 128-dim unified vector"""
            UNI_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
            ] + [
                STATE_VEC_IDX_MAPPING[f"right_hand_joint_{i}_pos"] for i in range(12)
            ] + [
                STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
            ] + [
                STATE_VEC_IDX_MAPPING[f"left_hand_joint_{i}_pos"] for i in range(12)
            ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec

        target_qpos = episode_cache["action"][step_id:step_id+self.CHUNK_SIZE]
        state = qpos[step_id:step_id+1]
        state_std = np.std(qpos, axis=0)
        state_indicator = np.ones_like(state_std)
        state_mean = np.mean(qpos, axis=0)
        state_norm = np.sqrt(np.mean(qpos**2, axis=0))
        actions = target_qpos
        
        # print(f"    原始state shape: {state.shape}")
        # print(f"    原始action shape: {actions.shape}")
        
        if actions.shape[0] < self.CHUNK_SIZE:
            # Pad the actions using the last action
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:],
                        (self.CHUNK_SIZE-actions.shape[0], 1))
            ], axis=0)
            # print(f"    Action已补齐到 {actions.shape}")

        # Fill the state into the unified vector
        state = fill_in_state(state)
        state_std = fill_in_state(state_std)
        state_mean = fill_in_state(state_mean)
        state_norm = fill_in_state(state_norm)
        actions = fill_in_state(actions)
        state_indicator = fill_in_state(state_indicator)
        
        # print(f"    填充后state shape: {state.shape}")
        # print(f"    填充后action shape: {actions.shape}")

        # Parse images on demand - load only the needed frames
        # print(f"\n  📷 加载图像...")
        images_info = episode_cache.get("images_info", {})
        
        def parse_img(cam_key):
            """Load image sequence for a specific camera"""
            # print(f"    加载 {cam_key}...")
            
            # Load IMG_HISTORY_SIZE frames around step_id
            start_idx = max(step_id - self.IMG_HISORY_SIZE + 1, 0)
            imgs = []
            
            for i in range(start_idx, step_id + 1):
                img = self._load_image_from_cache(images_info, cam_key, i)
                imgs.append(img)
            
            if len(imgs) == 0:
                # print(f"      ⚠️  未能加载任何图像")
                return np.zeros((self.IMG_HISORY_SIZE, 480, 640, 3), dtype=np.uint8)
            
            imgs = np.stack(imgs)
            
            # Pad images if history is not full
            if imgs.shape[0] < self.IMG_HISORY_SIZE:
                pad_width = self.IMG_HISORY_SIZE - imgs.shape[0]
                imgs = np.pad(imgs, ((pad_width, 0), (0,0), (0,0), (0,0)), 'edge')
            
            # print(f"      ✅ shape: {imgs.shape}")
            return imgs
        
        # Load images from 3 cameras (不使用第三人称摄像头)
        cam_high = parse_img('camera_head')
        cam_left_wrist = parse_img('camera_left_wrist')
        cam_right_wrist = parse_img('camera_right_wrist')

        # Create masks
        valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
        cam_mask = np.array(
            [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
        )
        
        # print(f"    图像mask: valid_len={valid_len}, mask={cam_mask}")

        # print(f"\n  📊 最终数据统计:")
        # print(f"    meta: {meta}")
        # print(f"    state: {state.shape}")
        # print(f"    state_std: {state_std.shape}")
        # print(f"    state_mean: {state_mean.shape}")
        # print(f"    state_norm: {state_norm.shape}")
        # print(f"    actions: {actions.shape}")
        # print(f"    state_indicator: {state_indicator.shape}")
        # print(f"    cam_high: {cam_high.shape}")
        # print(f"    cam_high_mask: {cam_mask.shape}")
        # print(f"    cam_left_wrist: {cam_left_wrist.shape}")
        # print(f"    cam_left_wrist_mask: {cam_mask.shape}")
        # print(f"    cam_right_wrist: {cam_right_wrist.shape}")
        # print(f"    cam_right_wrist_mask: {cam_mask.shape}")

        return True, {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": cam_mask,
            "cam_left_wrist": cam_left_wrist,
            "cam_left_wrist_mask": cam_mask.copy(),
            "cam_right_wrist": cam_right_wrist,
            "cam_right_wrist_mask": cam_mask.copy(),
        }

    def parse_lerobot_episode_state_only(self, episode_info):
        """
        Parse a lerobot episode to generate full state and action trajectories.
        用于统计计算，返回完整轨迹而不是单个时间步。
        
        Args:
            episode_info (dict): Episode information dict
            
        Returns:
            valid (bool): whether the episode is valid
            dict: a dictionary containing the full trajectory:
                {
                    "state": ndarray,   # state[:], (T, state_dim)
                    "action": ndarray,  # action[:], (T, action_dim)
                }
        """
        # print(f"\n  🔍 提取完整轨迹（state_only模式）...")
        
        # Load episode cache
        episode_idx = episode_info['episode_idx']
        episode_cache = self._load_episode_cache(episode_idx)
        
        qpos = episode_cache["state"]
        actions = episode_cache["action"]
        num_steps = episode_cache["frame_num"]
        
        # print(f"    总步数: {num_steps}")

        if num_steps < self.CHUNK_SIZE:
            # print(f"  ❌ Episode太短 ({num_steps} < {self.CHUNK_SIZE})")
            return False, None

        # Skip the first few still steps
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        first_idx = indices[0] if len(indices) > 0 else 1
        
        # print(f"    运动起始索引: {first_idx}")
        
        if first_idx >= num_steps:
            # print(f"  ❌ 起始索引超出范围")
            return False, None

        # Return full trajectory from first moving frame
        state_traj = qpos[first_idx-1:]
        action_traj = actions[first_idx-1:]
        
        # print(f"    轨迹长度: {len(state_traj)}")

        def fill_in_state(values):
            """Fill 36-dim state/action into 128-dim unified vector"""
            UNI_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
            ] + [
                STATE_VEC_IDX_MAPPING[f"right_hand_joint_{i}_pos"] for i in range(12)
            ] + [
                STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
            ] + [
                STATE_VEC_IDX_MAPPING[f"left_hand_joint_{i}_pos"] for i in range(12)
            ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec

        # 填充state和action到128维
        state_traj = fill_in_state(state_traj)
        action_traj = fill_in_state(action_traj)
        
        # print(f"    填充后state shape: {state_traj.shape}")
        # print(f"    填充后action shape: {action_traj.shape}")
        # print(f"  ✅ 轨迹提取完成")

        return True, {
            "state": state_traj,
            "action": action_traj
        }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 测试 LerobotDexDataset")
    print("="*70 + "\n")
    
    ds = LerobotDexDataset()
    
    print(f"\n数据集长度: {len(ds)}")
    
    # Test first episode
    print("\n" + "="*70)
    print("测试第一个Episode")
    print("="*70)
    sample = ds.get_item(0)
    
    print("\n" + "="*70)
    print("测试随机Episode")
    print("="*70)
    sample = ds.get_item()
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)
