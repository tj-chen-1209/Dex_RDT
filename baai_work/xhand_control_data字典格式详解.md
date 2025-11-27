# xhand_control_data.bson 字典格式详解

## 📦 整体文件结构

```
BSON 文件
  └── 文档列表 (list, 长度=1)
       └── 文档 0 (dict)
            └── 'frames' (list, 长度=452)
                 ├── 帧 0 (dict)
                 ├── 帧 1 (dict)
                 ├── ...
                 └── 帧 451 (dict)
```

**文件信息:**
- 文件大小: 288.59 KB
- BSON 文档数: 1 个
- 总帧数: 452 帧

---

## 🔍 完整字典结构

### 层级 1: BSON 文档

```python
# 读取后得到的数据
data = bson.decode_all(file_content)  # 返回 list
type(data)  # <class 'list'>
len(data)   # 1
```

### 层级 2: 顶层字典

```python
doc = data[0]  # 获取第一个文档
type(doc)      # <class 'dict'>
doc.keys()     # dict_keys(['frames'])
```

**顶层字典只有一个键:**
- `'frames'`: 包含所有帧数据的列表

### 层级 3: frames 列表

```python
frames = doc['frames']
type(frames)  # <class 'list'>
len(frames)   # 452
```

**frames 列表属性:**
- 类型: `list`
- 长度: `452` (对应 452 个时间点)
- 元素类型: `dict` (每个元素是一帧的数据字典)

### 层级 4: 单帧字典结构

每一帧是一个包含 3 个键的字典：

```python
frame = frames[0]  # 获取第一帧
type(frame)        # <class 'dict'>
frame.keys()       # dict_keys(['t', 'action', 'observation'])
```

**单帧字典的 3 个键:**

| 键名 | 类型 | 说明 |
|------|------|------|
| `'t'` | `float` | 时间戳(秒) |
| `'action'` | `dict` | 动作指令 |
| `'observation'` | `dict` | 传感器观测 |

---

## 📊 详细字段说明

### 1. 时间戳字段 `'t'`

```python
frame['t']  # 0.2000839050160721
type(frame['t'])  # <class 'float'>
```

**属性:**
- **类型**: `float`
- **单位**: 秒
- **范围**: 0.200 ~ 22.950
- **说明**: 相对时间戳，从开始录制时刻计时

**时间序列示例:**
```
帧 0:   t = 0.2001 秒
帧 100: t = 5.2502 秒
帧 200: t = 10.2502 秒
帧 300: t = 15.3501 秒
帧 451: t = 22.9502 秒
```

### 2. 动作字段 `'action'`

```python
action = frame['action']
type(action)  # <class 'dict'>
action.keys() # dict_keys(['left_hand', 'right_hand'])
```

**结构:**
```python
{
  'left_hand': [float × 12],   # 左手 12 维控制向量
  'right_hand': [float × 12]   # 右手 12 维控制向量
}
```

#### 2.1 `action['left_hand']` - 左手动作指令

```python
left_action = frame['action']['left_hand']
type(left_action)  # <class 'list'>
len(left_action)   # 12
type(left_action[0])  # <class 'float'>

# 转为 NumPy 数组
import numpy as np
left_action_np = np.array(left_action)
left_action_np.shape  # (12,)
left_action_np.dtype  # dtype('float64')
```

**完整数据示例 (帧 0):**
```python
[
  0.3329027209281921,    # 维度 0
  1.3588594453811647,    # 维度 1
  0.005547930195927619,  # 维度 2
  0.05706075113415718,   # 维度 3
  0.013657755986787378,  # 维度 4
  0.004675589229166508,  # 维度 5
  0.0,                   # 维度 6
  0.0,                   # 维度 7
  0.0,                   # 维度 8
  0.0,                   # 维度 9
  0.0,                   # 维度 10
  0.0990701819896698     # 维度 11
]
```

**数据特征:**
- **维度**: 12
- **数据类型**: `float64`
- **数值范围**: 约 0 ~ 1.6
- **特点**: 归一化的控制指令

#### 2.2 `action['right_hand']` - 右手动作指令

结构与左手相同：

```python
right_action = frame['action']['right_hand']  # list[12]
```

**完整数据示例 (帧 0):**
```python
[
  0.5535446681976318,       # 维度 0
  1.4174476760864259,       # 维度 1
  0.021568830478191373,     # 维度 2
  0.0,                      # 维度 3
  0.000766583930142224,     # 维度 4
  0.0006410610467195508,    # 维度 5
  0.0,                      # 维度 6
  0.0,                      # 维度 7
  0.0,                      # 维度 8
  0.0,                      # 维度 9
  0.0,                      # 维度 10
  0.00044186223968863386    # 维度 11
]
```

### 3. 观测字段 `'observation'`

```python
observation = frame['observation']
type(observation)  # <class 'dict'>
observation.keys()  # dict_keys(['left_hand', 'right_hand'])
```

**结构:**
```python
{
  'left_hand': [float × 12],   # 左手 12 维传感器读数
  'right_hand': [float × 12]   # 右手 12 维传感器读数
}
```

#### 3.1 `observation['left_hand']` - 左手观测数据

```python
left_obs = frame['observation']['left_hand']  # list[12]
```

**完整数据示例 (帧 0):**
```python
[
  20.74,   # 维度 0
  75.55,   # 维度 1
  6.41,    # 维度 2
  -0.57,   # 维度 3
  2.17,    # 维度 4
  33.57,   # 维度 5
  0.17,    # 维度 6
  1.25,    # 维度 7
  0.5,     # 维度 8
  0.58,    # 维度 9
  0.67,    # 维度 10
  6.91     # 维度 11
]
```

**数据特征:**
- **维度**: 12
- **数据类型**: `float64`
- **数值范围**: -3 ~ 85 (未归一化)
- **特点**: 传感器原始读数

#### 3.2 `observation['right_hand']` - 右手观测数据

**完整数据示例 (帧 0):**
```python
[
  35.67,   # 维度 0
  77.16,   # 维度 1
  10.75,   # 维度 2
  -0.02,   # 维度 3
  1.42,    # 维度 4
  46.08,   # 维度 5
  2.08,    # 维度 6
  35.25,   # 维度 7
  0.58,    # 维度 8
  11.67,   # 维度 9
  1.08,    # 维度 10
  34.25    # 维度 11
]
```

**数据特征:**
- **数值范围**: -4.5 ~ 112 (部分维度值很大)
- **特点**: 右手传感器读数范围比左手更大

---

## 📐 完整数据结构图

```
xhand_control_data.bson
│
└── [文档列表]
     │
     └── 文档 0 (dict)
          │
          └── 'frames': [帧列表, 长度 452]
               │
               ├── 帧 0 (dict)
               │    ├── 't': 0.2001 (float)
               │    ├── 'action': (dict)
               │    │    ├── 'left_hand': [12 floats]
               │    │    └── 'right_hand': [12 floats]
               │    └── 'observation': (dict)
               │         ├── 'left_hand': [12 floats]
               │         └── 'right_hand': [12 floats]
               │
               ├── 帧 1 (dict)
               │    └── (同上结构)
               │
               ├── ...
               │
               └── 帧 451 (dict)
                    └── (同上结构)
```

---

## 💻 Python 访问示例

### 基础读取

```python
import bson
import numpy as np

# 1. 读取 BSON 文件
with open("xhand_control_data.bson", 'rb') as f:
    data = bson.decode_all(f.read())

# 2. 获取顶层文档
doc = data[0]  # list[0] -> dict

# 3. 获取帧列表
frames = doc['frames']  # dict['frames'] -> list[452]

# 4. 访问第一帧
frame0 = frames[0]  # list[0] -> dict

# 5. 获取各字段
t = frame0['t']  # float
left_action = frame0['action']['left_hand']  # list[12]
right_action = frame0['action']['right_hand']  # list[12]
left_obs = frame0['observation']['left_hand']  # list[12]
right_obs = frame0['observation']['right_hand']  # list[12]

print(f"时间戳: {t}")
print(f"左手动作: {left_action}")
print(f"左手观测: {left_obs}")
```

### 批量处理所有帧

```python
# 提取所有帧的左手动作
all_left_actions = []
for frame in frames:
    all_left_actions.append(frame['action']['left_hand'])

# 转换为 NumPy 数组
all_left_actions = np.array(all_left_actions)
print(f"形状: {all_left_actions.shape}")  # (452, 12)

# 或使用列表推导式
all_left_actions = np.array([
    frame['action']['left_hand'] 
    for frame in frames
])

# 同时提取多个数据
timestamps = np.array([frame['t'] for frame in frames])
left_actions = np.array([frame['action']['left_hand'] for frame in frames])
right_actions = np.array([frame['action']['right_hand'] for frame in frames])
left_obs = np.array([frame['observation']['left_hand'] for frame in frames])
right_obs = np.array([frame['observation']['right_hand'] for frame in frames])

print(f"时间戳: {timestamps.shape}")      # (452,)
print(f"左手动作: {left_actions.shape}")  # (452, 12)
print(f"右手动作: {right_actions.shape}")  # (452, 12)
print(f"左手观测: {left_obs.shape}")      # (452, 12)
print(f"右手观测: {right_obs.shape}")      # (452, 12)
```

### 访问特定维度

```python
# 获取所有帧的左手第一个维度
left_dim0 = all_left_actions[:, 0]  # shape: (452,)

# 获取第 100 帧的所有数据
frame100 = frames[100]
t100 = frame100['t']
action100 = frame100['action']
obs100 = frame100['observation']

# 获取第 100 帧左手的第 5 个维度
left_hand_dim5 = frame100['action']['left_hand'][5]
```

---

## 📊 数据类型总结表

| 访问路径 | Python 类型 | NumPy dtype | 形状/长度 | 说明 |
|---------|------------|-------------|----------|------|
| `data` | `list` | - | 1 | BSON 文档列表 |
| `data[0]` | `dict` | - | 1 键 | 顶层文档 |
| `data[0]['frames']` | `list` | - | 452 | 帧列表 |
| `frames[i]` | `dict` | - | 3 键 | 单帧数据 |
| `frames[i]['t']` | `float` | - | 标量 | 时间戳 |
| `frames[i]['action']` | `dict` | - | 2 键 | 动作字典 |
| `frames[i]['action']['left_hand']` | `list` | `float64` | 12 | 左手动作 |
| `frames[i]['action']['right_hand']` | `list` | `float64` | 12 | 右手动作 |
| `frames[i]['observation']` | `dict` | - | 2 键 | 观测字典 |
| `frames[i]['observation']['left_hand']` | `list` | `float64` | 12 | 左手观测 |
| `frames[i]['observation']['right_hand']` | `list` | `float64` | 12 | 右手观测 |

**转换为 NumPy 数组后:**

| 数据 | 形状 | dtype |
|-----|------|-------|
| 所有时间戳 | `(452,)` | `float64` |
| 所有左手动作 | `(452, 12)` | `float64` |
| 所有右手动作 | `(452, 12)` | `float64` |
| 所有左手观测 | `(452, 12)` | `float64` |
| 所有右手观测 | `(452, 12)` | `float64` |

---

## 🔄 数据对比: action vs observation

### 数值范围对比

**左手 (left_hand):**

| 维度 | action 范围 | observation 范围 |
|-----|-------------|-----------------|
| 0 | 0.02 - 0.64 | 2.17 - 37.23 |
| 1 | 1.23 - 1.51 | 71.55 - 85.04 |
| 2 | 0.00 - 0.02 | -1.08 - 6.41 |
| ... | 归一化 [0-2] | 原始读数 [-3-85] |

**右手 (right_hand):**

| 维度 | action 范围 | observation 范围 |
|-----|-------------|-----------------|
| 0 | 0.00 - 1.32 | 13.83 - 73.00 |
| 1 | 0.68 - 1.57 | 36.08 - 84.33 |
| 7 | 0.00 - 1.94 | -4.50 - 110.33 |
| 9 | 0.00 - 1.92 | -3.75 - 112.41 |

### 关键区别

| 特性 | action | observation |
|-----|--------|-------------|
| **用途** | 控制指令 | 传感器反馈 |
| **数值范围** | 归一化 [0-2] | 未归一化 [-5-112] |
| **单位** | 无量纲 | 可能是度数或编码器值 |
| **变化** | 平滑 | 跳跃性更大 |

---

## 📝 JSON 格式示例

### 单帧完整 JSON

```json
{
  "t": 0.2000839050160721,
  "action": {
    "left_hand": [
      0.3329, 1.3589, 0.0055, 0.0571, 0.0137, 0.0047,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0991
    ],
    "right_hand": [
      0.5535, 1.4174, 0.0216, 0.0, 0.0008, 0.0006,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0004
    ]
  },
  "observation": {
    "left_hand": [
      20.74, 75.55, 6.41, -0.57, 2.17, 33.57,
      0.17, 1.25, 0.5, 0.58, 0.67, 6.91
    ],
    "right_hand": [
      35.67, 77.16, 10.75, -0.02, 1.42, 46.08,
      2.08, 35.25, 0.58, 11.67, 1.08, 34.25
    ]
  }
}
```

---

## 🎯 使用建议

### 1. 数据加载策略

```python
# 一次性加载所有数据（内存足够时）
with open("xhand_control_data.bson", 'rb') as f:
    data = bson.decode_all(f.read())
frames = data[0]['frames']

# 转换为 NumPy 数组便于处理
import numpy as np
data_dict = {
    'timestamps': np.array([f['t'] for f in frames]),
    'action_left': np.array([f['action']['left_hand'] for f in frames]),
    'action_right': np.array([f['action']['right_hand'] for f in frames]),
    'obs_left': np.array([f['observation']['left_hand'] for f in frames]),
    'obs_right': np.array([f['observation']['right_hand'] for f in frames])
}
```

### 2. 数据归一化

```python
# observation 需要归一化才能与 action 匹配
obs_left = data_dict['obs_left']
obs_left_norm = (obs_left - obs_left.mean(axis=0)) / obs_left.std(axis=0)
```

### 3. 时序数据处理

```python
# 获取时间序列
timestamps = data_dict['timestamps']
time_diffs = np.diff(timestamps)
sampling_rate = 1.0 / np.mean(time_diffs)  # ~19.87 Hz

print(f"采样率: {sampling_rate:.2f} Hz")
print(f"平均时间间隔: {np.mean(time_diffs)*1000:.2f} ms")
```

---

生成时间: 2025-11-26  
分析文件: xhand_control_data.bson (episode_0)

