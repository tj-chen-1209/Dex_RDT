#!/usr/bin/env python3
"""
比较 observation 和 action 中的 right_arm joint_state
"""
import bson
import numpy as np
import matplotlib.pyplot as plt

bson_path = "/home/chensiqi/chensiqi/RDT_libero_finetune/data/baai/data/action176/episode_0/episode_0.bson"

# 读取 BSON 文件
with open(bson_path, 'rb') as f:
    data = bson.decode_all(f.read())[0]

# 获取数据
obs_right = data['data']['/observation/right_arm/joint_state']
act_right = data['data']['/action/right_arm/joint_state']

print(f"\n数据点数量:")
print(f"  observation: {len(obs_right)}")
print(f"  action: {len(act_right)}")

# 提取位置数据
obs_pos = np.array([frame['data']['pos'] for frame in obs_right])
act_pos = np.array([frame['data']['pos'] for frame in act_right])

print(f"\n数据形状:")
print(f"  observation pos: {obs_pos.shape}")
print(f"  action pos: {act_pos.shape}")

# 提取时间戳
obs_timestamps = np.array([frame['t'] for frame in obs_right])
act_timestamps = np.array([frame['t'] for frame in act_right])

print(f"\n时间戳范围:")
print(f"  observation: {obs_timestamps[0]} - {obs_timestamps[-1]}")
print(f"  action: {act_timestamps[0]} - {act_timestamps[-1]}")

for i in range(100,min(150, len(obs_right))):
    print(f"\n帧 {i}:")
    print(f"  observation 时间戳: {obs_right[i]['t']}")
    print(f"  action 时间戳:      {act_right[i]['t']}")
    print(f"  时间差: {act_right[i]['t'] - obs_right[i]['t']} ms")
    
    print(f"\n  observation pos: {obs_pos[i]}")
    print(f"  action pos:      {act_pos[i]}")
    


