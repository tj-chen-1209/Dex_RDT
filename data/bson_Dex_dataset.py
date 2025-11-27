import os
import fnmatch
import json

import bson
import yaml
import numpy as np
from PIL import Image
import torch
from configs.state_vec import STATE_VEC_IDX_MAPPING

class BsonDexDataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in Bson.
    """

    def __init__(self) -> None:
        # Path to the Bson dataset directory
        BSON_DIR = "data/baai/data/"
        self.DATASET_NAME = "baai"

        # print(f"Loading Bson dataset from {BSON_DIR}...")
        
        # Find all valid episode directories
        self.episode_paths = []
        
        # Iterate through action directories
        for action_item in os.listdir(BSON_DIR):
            action_path = os.path.join(BSON_DIR, action_item)
            # print(f"Action path: {action_path}")
            if not os.path.isdir(action_path) or not action_item.startswith('action'):
                # print(f"Warning: Invalid action directory: {action_path}")
                continue
        
            # Iterate through episode subdirectories
            for episode_item in os.listdir(action_path):
                episode_path = os.path.join(action_path, episode_item)
                # print(f"Action path: {action_path}, Episode path: {episode_path}")
                if not os.path.isdir(episode_path) or not episode_item.startswith('episode'):
                    print(f"Warning: Invalid episode directory: {episode_path}")
                    continue
                
                # Check if required files exist
                main_bson = os.path.join(episode_path, "episode_0.bson")
                xhand_bson = os.path.join(episode_path, "xhand_control_data.bson")
                if os.path.exists(main_bson) and os.path.exists(xhand_bson):
                    # Check camera folders
                    required_cameras = ['camera_head', 'camera_left_wrist', 
                                       'camera_right_wrist', 'camera_third_view']
                    cameras_exist = all(
                        os.path.exists(os.path.join(episode_path, cam)) 
                        for cam in required_cameras
                    )
                    
                    if cameras_exist:
                        self.episode_paths.append(episode_path)
                    else:
                        print(f"Warning: Missing camera folders in {episode_path}")
                else:
                    print(f"Warning: Missing BSON files in {episode_path}")
        
        # print(f"Found {len(self.episode_paths)} valid episodes")
        
        if len(self.episode_paths) == 0:
            raise ValueError(f"No valid episodes found in {BSON_DIR}")

        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        # Calculate episode lengths and sample weights
        # print("Calculating episode lengths...")
        episode_lens = []
        valid_episodes = []
        
        for episode_path in self.episode_paths:
            try:
                # Quick length check by reading BSON
                main_bson_path = os.path.join(episode_path, "episode_0.bson")
                with open(main_bson_path, 'rb') as f:
                    main_data = bson.decode(f.read())["data"]
                    num_steps = len(main_data["/observation/left_arm/joint_state"])
                
                if num_steps >= self.CHUNK_SIZE:  # Drop too short episodes
                    episode_lens.append(num_steps)
                    valid_episodes.append(episode_path)
                else:
                    print(f"Skipping short episode {episode_path}: {num_steps} steps")
            except Exception as e:
                print(f"Error reading {episode_path}: {e}")
                continue
        
        self.episode_paths = valid_episodes
        # print(f"Using {len(self.episode_paths)} valid episodes after filtering")
        
        if len(episode_lens) == 0:
            raise ValueError("No valid episodes after filtering")
            
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)

    def _extract_data_from_episode(self, episode_path):
        '''
        if the episode is valid, return the data of the episode
        else return None
        Args:
            episode_path (str): the path to the bson episode
        Returns:
            dict: a dictionary containing the data of the episode
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                    "frame_num": int,          # the number of frames in the episode.
                    "images": dict,            # a dictionary containing the images of the episode.(T, H, W, 3)
                } or None if the episode is invalid.
        '''
        # print('Analyzing BSON episode...')
        arm_bson_path = os.path.join(episode_path, "episode_0.bson")
        xhand_bson_path = os.path.join(episode_path, "xhand_control_data.bson")
        try:
            with open(arm_bson_path, 'rb') as f:
                arm_data = bson.decode(f.read())["data"]
        except Exception as e:
            print(f"Error reading main BSON file {arm_bson_path}: {e}")
            return None
        
        try:
            with open(xhand_bson_path, 'rb') as f:
                xhand_data = bson.decode(f.read())
        except Exception as e:
            print(f"Error reading xhand BSON file {xhand_bson_path}: {e}")
            return None

        # Define data keys for arms (same as original)
        arm_dim, eef_dim = 6, 12
        keys = {
            "left_obs_arm": "/observation/left_arm/joint_state",
            "right_obs_arm": "/observation/right_arm/joint_state",
            "left_act_arm": "/action/left_arm/joint_state",
            "right_act_arm": "/action/right_arm/joint_state",
        }

        # Check frame counts
        arm_frame_num = len(arm_data[keys["left_obs_arm"]])
        xhand_frame_num = len(xhand_data['frames'])
        # print(f"Arm frame number: {arm_frame_num}, Xhand frame number: {xhand_frame_num}")

        # Use minimum frame count to ensure alignment
        frame_num = min(arm_frame_num, xhand_frame_num)
        
        # Extract arm data
        state = np.zeros((frame_num, 2 * (arm_dim + eef_dim)), dtype=np.float32)
        action = np.zeros((frame_num, 2 * (arm_dim + eef_dim)), dtype=np.float32)

        # xhand obs from degree to radian !!!!!!!!!!!!!!!!!!
        for i in range(frame_num):
            xhand_data['frames'][i]["observation"]["right_hand"] = np.deg2rad(xhand_data['frames'][i]["observation"]["right_hand"])
            xhand_data['frames'][i]["observation"]["left_hand"] = np.deg2rad(xhand_data['frames'][i]["observation"]["left_hand"])
        
        # Check if action data has correct length (6) for arms
        use_arm_actions = True
        try:
            # Check first frame to determine if action data is valid
            left_arm_action = arm_data[keys["left_act_arm"]][0]["data"]["pos"]
            right_arm_action = arm_data[keys["right_act_arm"]][0]["data"]["pos"]
            if len(left_arm_action) != arm_dim or len(right_arm_action) != arm_dim:
                use_arm_actions = False
                print(f"Warning: Action data has incorrect length in {episode_path}")
        except (KeyError, IndexError):
            use_arm_actions = False
            print(f"Warning: Action data not available in {episode_path}, using observation as action")

        for i in range(frame_num):
            state[i, :] = np.concatenate([
                arm_data[keys["right_obs_arm"]][i]["data"]["pos"],
                xhand_data['frames'][i]["observation"]["right_hand"],
                arm_data[keys["left_obs_arm"]][i]["data"]["pos"],
                xhand_data['frames'][i]["observation"]["left_hand"]

            ])
            
            # Use action data if available and correct, otherwise use observation
            if use_arm_actions:
                left_arm_data = arm_data[keys["left_act_arm"]][i]["data"]["pos"]
                right_arm_data = arm_data[keys["right_act_arm"]][i]["data"]["pos"]
            else:
                left_arm_data = arm_data[keys["left_obs_arm"]][i]["data"]["pos"]
                right_arm_data = arm_data[keys["right_obs_arm"]][i]["data"]["pos"]
            
            action[i, :] = np.concatenate([
                right_arm_data,
                xhand_data['frames'][i]["action"]["right_hand"],
                left_arm_data,
                xhand_data['frames'][i]["action"]["left_hand"]
            ])

        # Prepare image path information (lazy loading - do NOT load actual images)
        # 准备图像路径信息（懒加载 - 不加载实际图像）
        camera_folders = ['camera_head', 'camera_left_wrist', 'camera_right_wrist', 'camera_third_view']
        images_info = {}
        
        for cam_folder in camera_folders:
            cam_path = os.path.join(episode_path, cam_folder)
            if not os.path.exists(cam_path):
                print(f"Warning: {cam_folder} not found in {episode_path}")
                images_info[cam_folder] = None
                return None
            
            # Get sorted list of jpg files - only store file names, not actual images
            jpg_files = sorted([f for f in os.listdir(cam_path) if f.endswith('.jpg')])
            # Store file info aligned with state data
            images_info[cam_folder] = {
                'type': 'file_sequence',
                'path': cam_path,
                'files': jpg_files[:frame_num]  # Only file names, not images!
            }
            
            if len(jpg_files) < frame_num:
                print(f"Warning: {cam_folder} has fewer images ({len(jpg_files)}) than frames ({frame_num})")

        return {
            "state": state,
            "action": action,
            "frame_num": frame_num,
            "images_info": images_info,  # Path info, not actual images
            "episode_path": episode_path,
        }

    def __len__(self):
        return len(self.episode_paths)

    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def _load_file_sequence(self, cam_info: dict, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Load a sequence of image files on demand.
        按需加载图像序列
        
        Args:
            cam_info: Camera information dict with 'path' and 'files'
            start_idx: Start frame index
            end_idx: End frame index (exclusive)
            
        Returns:
            np.ndarray: Stacked images of shape (T, H, W, 3)
        """
        if cam_info is None or cam_info.get('type') != 'file_sequence':
            print(f"Warning: Invalid cam_info in _load_file_sequence: {cam_info}")
            return np.array([])
        
        frames = []
        cam_path = cam_info['path']
        files = cam_info['files']
        
        for i in range(start_idx, min(end_idx, len(files))):
            img_path = os.path.join(cam_path, files[i])
            try:
                with Image.open(img_path) as img:
                    img_array = np.array(img)
                if img_array.ndim == 2:  # Grayscale to RGB
                    img_array = np.stack([img_array] * 3, axis=-1)
                frames.append(img_array)
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}. Error: {e}")
                # Use zero image as placeholder
                if frames:
                    frames.append(np.zeros_like(frames[0]))
                else:
                    frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        if frames:
            return np.stack(frames)
        else:
            print(f"Warning: No frames loaded in _load_file_sequence for path {cam_info.get('path', 'unknown')}")
            return np.array([])

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
        while True:
            if index is None:
                episode_idx = np.random.choice(
                    len(self.episode_paths), p=self.episode_sample_weights)
                episode_path = self.episode_paths[episode_idx]
            else:
                episode_path = self.episode_paths[index]
            
            # Parse episode based on state_only flag
            if state_only:
                valid, sample = self.parse_bson_episode_state_only(episode_path)
            else:
                valid, sample = self.parse_bson_episode(episode_path)
            
            if valid:
                return sample
            else:
                if index is None:
                    print(f"Warning: Invalid sample from {episode_path}, resampling...")
                    continue  # Try another random episode
                else:
                    raise RuntimeError(f"Episode at index {index} is invalid: {episode_path}")

    def parse_bson_episode(self, episode_path):
        """[Modify] Parse a bson episode to generate a training sample at
            a random timestep.

        Args:
            episode_path (str): the path to the bson episode
            demo_key (str): the key of the demo 
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        
        episode_data = self._extract_data_from_episode(episode_path)
        if not episode_data:
            return False, None

        qpos = episode_data["state"]
        num_steps = episode_data["frame_num"]

        # [Optional] We skip the first few still steps TODO
        EPS = 1e-2
        # Get the idx of the first qpos whose delta exceeds the threshold
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            raise ValueError("Found no qpos that exceeds the threshold.")

        if first_idx >= num_steps:  # case where robot doesn't move
            return False, None

        # We randomly sample a timestep TODO
        step_id = np.random.randint(first_idx-1, num_steps)
        action_dir = os.path.dirname(episode_path)
        # TODO: instruction function
        if os.path.exists(os.path.join(action_dir, "instruction.pt")):
            instruction = os.path.join(action_dir, "instruction.pt")
        else:
            instruction = "Use the left hand to hook the book '皮囊' from the pile of books,then use the right hand to place it on the right bookshelf."


        # Assemble the meta
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": instruction
        }

        def fill_in_state(values):
            # Target indices corresponding to your state space
            # In this example: 6 arm_joint_state + 12 hand_state
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

        target_qpos = episode_data["action"][step_id:step_id+self.CHUNK_SIZE]
        state = qpos[step_id:step_id+1]
        state_std = np.std(qpos, axis=0)
        state_indicator = np.ones_like(state_std)
        state_mean = np.mean(qpos, axis=0)
        state_norm = np.sqrt(np.mean(qpos**2, axis=0))
        actions = target_qpos
        if actions.shape[0] < self.CHUNK_SIZE:
            # Pad the actions using the last action
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:],
                        (self.CHUNK_SIZE-actions.shape[0], 1))
            ], axis=0)

        # Fill the state into the unified vector
        state = fill_in_state(state)
        state_std = fill_in_state(state_std)
        state_mean = fill_in_state(state_mean)
        state_norm = fill_in_state(state_norm)
        # Fill the actions into the unified vector
        actions = fill_in_state(actions)
        # Fill the state_indicator into the unified vector
        state_indicator = fill_in_state(state_indicator)

        ### 到这里运动规划结束，接下来是图片解析（懒加载）
        
        # Parse images on demand - load only the needed frames
        # 按需解析图像 - 只加载需要的帧
        images_info = episode_data["images_info"]
        
        def parse_img(cam_key):
            """Load image sequence for a specific camera"""
            img_info = images_info.get(cam_key)
            
            if img_info is None:
                print(f"Warning: {cam_key} not found for episode")
                return np.zeros((self.IMG_HISORY_SIZE, 480, 640, 3), dtype=np.uint8)
            
            # Load only IMG_HISTORY_SIZE frames around step_id
            start_idx = max(step_id - self.IMG_HISORY_SIZE + 1, 0)
            imgs = self._load_file_sequence(img_info, start_idx, step_id + 1)
            
            if imgs.ndim != 4 or imgs.shape[0] == 0:  # If loading failed or empty
                print(f"Warning: Failed to load file sequence for {cam_key}")
                return np.zeros((self.IMG_HISORY_SIZE, 480, 640, 3), dtype=np.uint8)
            
            # Pad images if history is not full
            if imgs.shape[0] < self.IMG_HISORY_SIZE:
                pad_width = self.IMG_HISORY_SIZE - imgs.shape[0]
                imgs = np.pad(imgs, ((pad_width, 0), (0,0), (0,0), (0,0)), 'edge')
            
            return imgs
        
        # Load images from all 4 cameras
        cam_high = parse_img('camera_head')  # Head camera -> cam_high
        # print("cam_high.shape: ", cam_high.shape)
        cam_left_wrist = parse_img('camera_left_wrist')
        cam_right_wrist = parse_img('camera_right_wrist')
        # cam_third_view = parse_img('camera_third_view')

        # Create masks - valid_len indicates how many frames are real (not padded)
        valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
        cam_mask = np.array(
            [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
        )

        # print("meta: ", meta)
        # print("step_id: ", meta["step_id"])
        # print("instruction: ", meta["instruction"])
        # print("num_steps: ", meta["#steps"])
        # print("state: ", state.shape)
        # print("state_std: ", state_std.shape)
        # print("state_mean: ", state_mean.shape)
        # print("state_norm: ", state_norm.shape)
        # print("actions: ", actions.shape)
        # print("state_indicator: ", state_indicator.shape)
        # print("cam_high: ", cam_high.shape)
        # print("cam_high_mask: ", cam_mask.shape)
        # print("cam_left_wrist: ", cam_left_wrist.shape)
        # print("cam_left_wrist_mask: ", cam_mask.shape)
        # print("cam_right_wrist: ", cam_right_wrist.shape)
        # print("cam_right_wrist_mask: ", cam_mask.shape)

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
            # "cam_third_view": cam_third_view,
            # "cam_third_view_mask": cam_mask.copy(),
        }

    def parse_bson_episode_state_only(self, episode_path):
        """
        Parse a bson episode to generate full state and action trajectories.
        用于统计计算，返回完整轨迹而不是单个时间步。
        
        Args:
            episode_path (str): the path to the bson episode
            
        Returns:
            valid (bool): whether the episode is valid
            dict: a dictionary containing the full trajectory:
                {
                    "state": ndarray,   # state[:], (T, state_dim)
                    "action": ndarray,  # action[:], (T, action_dim)
                }
        """
        episode_data = self._extract_data_from_episode(episode_path)
        if not episode_data:
            return False, None

        qpos = episode_data["state"]
        actions = episode_data["action"]
        num_steps = episode_data["frame_num"]

        if num_steps < self.CHUNK_SIZE:  # Drop too-short episodes
            return False, None

        # Skip the first few still steps
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        first_idx = indices[0] if len(indices) > 0 else 1
        
        if first_idx >= num_steps:
            return False, None

        # Return full trajectory from first moving frame
        state_traj = qpos[first_idx-1:]
        action_traj = actions[first_idx-1:]

        # 添加这个函数来填充到128维
        def fill_in_state(values):
            # Target indices corresponding to your state space
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

        return True, {
            "state": state_traj,
            "action": action_traj
        }


if __name__ == "__main__":
    ds = BsonDexDataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
