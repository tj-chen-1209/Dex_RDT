#!/usr/bin/env python3
"""代码Review和验证脚本 - 检查BSON数据和LeRobot输出"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import bson
import json

# 添加lerobot路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lerobot' / 'src'))
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class BSONLeRobotVerifier:
    """验证BSON数据和LeRobot转换的完整性"""
    
    def __init__(self, bson_dir="data/baai/data/", lerobot_dir="data/baai/data/lerobot_baai"):
        self.bson_dir = Path(bson_dir)
        self.lerobot_dir = Path(lerobot_dir)
        self.issues = []
        self.warnings = []
        
    def log_issue(self, msg):
        """记录问题"""
        self.issues.append(f"❌ {msg}")
        print(f"❌ {msg}")
    
    def log_warning(self, msg):
        """记录警告"""
        self.warnings.append(f"⚠️  {msg}")
        print(f"⚠️  {msg}")
    
    def log_ok(self, msg):
        """记录通过"""
        print(f"✅ {msg}")
    
    # ============ 步骤1: 验证BSON数据结构 ============
    def step1_verify_bson_structure(self):
        """验证BSON文件的完整性和数据结构"""
        print("\n" + "="*60)
        print("步骤1: 验证BSON数据结构")
        print("="*60)
        
        # 查找一个示例episode
        episodes = list(self.bson_dir.glob("action*/episode_*"))
        if not episodes:
            self.log_issue("未找到任何episode目录")
            return False
        
        ep_path = episodes[0]
        print(f"\n检查示例episode: {ep_path}")
        
        # 1.1 检查必需文件
        print("\n1.1 检查必需文件:")
        required_files = [
            "episode_0.bson",
            "xhand_control_data.bson"
        ]
        required_dirs = [
            "camera_head",
            "camera_left_wrist", 
            "camera_right_wrist",
            "camera_third_view"
        ]
        
        for fname in required_files:
            if (ep_path / fname).exists():
                self.log_ok(f"{fname} 存在")
            else:
                self.log_issue(f"{fname} 缺失")
        
        for dname in required_dirs:
            if (ep_path / dname).exists():
                img_count = len(list((ep_path / dname).glob("*.jpg")))
                self.log_ok(f"{dname}/ 存在 ({img_count} 张图片)")
            else:
                self.log_issue(f"{dname}/ 缺失")
        
        # 1.2 检查BSON数据结构
        print("\n1.2 检查episode_0.bson数据结构:")
        try:
            with open(ep_path / "episode_0.bson", 'rb') as f:
                arm_data = bson.decode(f.read())
            
            # 检查关键字段
            if "data" not in arm_data:
                self.log_issue("episode_0.bson缺少'data'字段")
                return False
            
            arm_data = arm_data["data"]
            
            # 检查observation字段
            obs_keys = [
                "/observation/left_arm/joint_state",
                "/observation/right_arm/joint_state"
            ]
            for key in obs_keys:
                if key in arm_data:
                    count = len(arm_data[key])
                    if count > 0:
                        sample = arm_data[key][0]
                        if "data" in sample:
                            pos_len = len(sample["data"].get("pos", []))
                            vel_len = len(sample["data"].get("vel", []))
                            eff_len = len(sample["data"].get("eff", []))
                            self.log_ok(f"{key}: {count}帧, pos={pos_len}, vel={vel_len}, eff={eff_len}")
                        else:
                            self.log_warning(f"{key}[0]缺少'data'字段")
                    else:
                        self.log_warning(f"{key}为空")
                else:
                    self.log_issue(f"缺少字段: {key}")
            
            # 检查action字段（可能不存在）
            action_keys = [
                "/action/left_arm/joint_state",
                "/action/right_arm/joint_state"
            ]
            has_action = False
            for key in action_keys:
                if key in arm_data and len(arm_data[key]) > 0:
                    has_action = True
                    try:
                        sample = arm_data[key][0]
                        pos_len = len(sample["data"]["pos"])
                        self.log_ok(f"{key}: {len(arm_data[key])}帧, pos={pos_len}")
                    except:
                        self.log_warning(f"{key}存在但格式异常")
            
            if not has_action:
                self.log_warning("episode_0.bson中没有action数据（将使用observation作为action）")
        
        except Exception as e:
            self.log_issue(f"读取episode_0.bson失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 1.3 检查xhand_control_data.bson
        print("\n1.3 检查xhand_control_data.bson数据结构:")
        try:
            with open(ep_path / "xhand_control_data.bson", 'rb') as f:
                hand_data = bson.decode(f.read())
            
            if "frames" not in hand_data:
                self.log_issue("xhand_control_data.bson缺少'frames'字段")
                return False
            
            frame_count = len(hand_data['frames'])
            self.log_ok(f"总帧数: {frame_count}")
            
            if frame_count > 0:
                sample_frame = hand_data['frames'][0]
                
                # 检查observation
                if "observation" in sample_frame:
                    obs = sample_frame["observation"]
                    left_len = len(obs.get("left_hand", []))
                    right_len = len(obs.get("right_hand", []))
                    self.log_ok(f"observation: left_hand={left_len}, right_hand={right_len}")
                else:
                    self.log_issue("frames[0]缺少'observation'字段")
                
                # 检查action
                if "action" in sample_frame:
                    act = sample_frame["action"]
                    left_len = len(act.get("left_hand", []))
                    right_len = len(act.get("right_hand", []))
                    self.log_ok(f"action: left_hand={left_len}, right_hand={right_len}")
                else:
                    self.log_issue("frames[0]缺少'action'字段")
        
        except Exception as e:
            self.log_issue(f"读取xhand_control_data.bson失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 1.4 检查图像
        print("\n1.4 检查图像文件:")
        for cam in required_dirs:
            cam_dir = ep_path / cam
            if cam_dir.exists():
                imgs = sorted(cam_dir.glob("*.jpg"))
                if imgs:
                    # 检查第一张图片
                    try:
                        img = Image.open(imgs[0])
                        self.log_ok(f"{cam}: {len(imgs)}张图片, 尺寸={img.size}, 模式={img.mode}")
                    except Exception as e:
                        self.log_issue(f"{cam}图片读取失败: {e}")
                else:
                    self.log_warning(f"{cam}目录为空")
        
        return True
    
    # ============ 步骤2: 验证数据维度一致性 ============
    def step2_verify_dimensions(self):
        """验证各数据源的维度一致性"""
        print("\n" + "="*60)
        print("步骤2: 验证数据维度一致性")
        print("="*60)
        
        episodes = list(self.bson_dir.glob("action*/episode_*"))[:3]  # 检查前3个
        
        for ep_path in episodes:
            print(f"\n检查: {ep_path.name}")
            
            try:
                # 加载数据
                with open(ep_path / "episode_0.bson", 'rb') as f:
                    arm_data = bson.decode(f.read())["data"]
                with open(ep_path / "xhand_control_data.bson", 'rb') as f:
                    hand_data = bson.decode(f.read())
                
                # 获取各数据源的长度
                arm_obs_len = len(arm_data["/observation/left_arm/joint_state"])
                hand_len = len(hand_data['frames'])
                
                img_lens = {}
                for cam in ['camera_head', 'camera_left_wrist', 'camera_right_wrist', 'camera_third_view']:
                    cam_dir = ep_path / cam
                    if cam_dir.exists():
                        img_lens[cam] = len(list(cam_dir.glob("*.jpg")))
                
                # 检查一致性
                all_lens = [arm_obs_len, hand_len] + list(img_lens.values())
                min_len = min(all_lens)
                max_len = max(all_lens)
                
                print(f"  机械臂observation: {arm_obs_len}帧")
                print(f"  灵巧手数据: {hand_len}帧")
                for cam, img_len in img_lens.items():
                    print(f"  {cam}: {img_len}帧")
                
                if min_len == max_len:
                    self.log_ok(f"所有数据源长度一致: {min_len}帧")
                else:
                    self.log_warning(f"数据源长度不一致: {min_len}~{max_len}帧（代码会使用min={min_len}）")
            
            except Exception as e:
                self.log_issue(f"验证失败: {e}")
    
    # ============ 步骤3: 验证数据范围和类型 ============
    def step3_verify_data_ranges(self):
        """验证数据的数值范围和类型"""
        print("\n" + "="*60)
        print("步骤3: 验证数据范围和类型")
        print("="*60)
        
        episodes = list(self.bson_dir.glob("action*/episode_*"))
        if not episodes:
            return
        
        ep_path = episodes[0]
        print(f"\n分析: {ep_path.name}")
        
        try:
            with open(ep_path / "episode_0.bson", 'rb') as f:
                arm_data = bson.decode(f.read())["data"]
            with open(ep_path / "xhand_control_data.bson", 'rb') as f:
                hand_data = bson.decode(f.read())
            
            # 3.1 机械臂数据
            print("\n3.1 机械臂joint数据范围:")
            for arm in ["left_arm", "right_arm"]:
                obs_key = f"/observation/{arm}/joint_state"
                if obs_key in arm_data and len(arm_data[obs_key]) > 0:
                    # 收集所有pos/vel/eff
                    pos_data = [frame["data"]["pos"] for frame in arm_data[obs_key]]
                    vel_data = [frame["data"]["vel"] for frame in arm_data[obs_key]]
                    eff_data = [frame["data"]["eff"] for frame in arm_data[obs_key]]
                    
                    pos_arr = np.array(pos_data)
                    vel_arr = np.array(vel_data)
                    eff_arr = np.array(eff_data)
                    
                    print(f"\n  {arm}:")
                    print(f"    pos形状: {pos_arr.shape}, 范围: [{pos_arr.min():.3f}, {pos_arr.max():.3f}]")
                    print(f"    vel形状: {vel_arr.shape}, 范围: [{vel_arr.min():.3f}, {vel_arr.max():.3f}]")
                    print(f"    eff形状: {eff_arr.shape}, 范围: [{eff_arr.min():.3f}, {eff_arr.max():.3f}]")
                    
                    # 检查异常值
                    if np.any(np.isnan(pos_arr)) or np.any(np.isinf(pos_arr)):
                        self.log_issue(f"{arm} pos包含NaN或Inf")
                    if np.any(np.isnan(vel_arr)) or np.any(np.isinf(vel_arr)):
                        self.log_issue(f"{arm} vel包含NaN或Inf")
                    if np.any(np.isnan(eff_arr)) or np.any(np.isinf(eff_arr)):
                        self.log_issue(f"{arm} eff包含NaN或Inf")
            
            # 3.2 灵巧手数据
            print("\n3.2 灵巧手数据范围:")
            for hand in ["left_hand", "right_hand"]:
                obs_data = [frame["observation"][hand] for frame in hand_data['frames']]
                act_data = [frame["action"][hand] for frame in hand_data['frames']]
                
                obs_arr = np.array(obs_data)
                act_arr = np.array(act_data)
                
                print(f"\n  {hand}:")
                print(f"    observation形状: {obs_arr.shape}, 范围: [{obs_arr.min():.3f}, {obs_arr.max():.3f}] (度)")
                print(f"    action形状: {act_arr.shape}, 范围: [{act_arr.min():.3f}, {act_arr.max():.3f}]")
                
                # 转换为弧度后的范围
                obs_rad = np.deg2rad(obs_arr)
                print(f"    observation(弧度): [{obs_rad.min():.3f}, {obs_rad.max():.3f}]")
                
                if np.any(np.isnan(obs_arr)) or np.any(np.isinf(obs_arr)):
                    self.log_issue(f"{hand} observation包含NaN或Inf")
                if np.any(np.isnan(act_arr)) or np.any(np.isinf(act_arr)):
                    self.log_issue(f"{hand} action包含NaN或Inf")
            
            # 3.3 拼接后的action维度
            print("\n3.3 拼接后的action:")
            # 检查是否有arm action
            has_arm_action = "/action/left_arm/joint_state" in arm_data and \
                           len(arm_data["/action/left_arm/joint_state"]) > 0
            
            if has_arm_action:
                try:
                    left_arm = arm_data["/action/left_arm/joint_state"][0]["data"]["pos"]
                    right_arm = arm_data["/action/right_arm/joint_state"][0]["data"]["pos"]
                    print(f"  使用arm action数据")
                except:
                    left_arm = arm_data["/observation/left_arm/joint_state"][0]["data"]["pos"]
                    right_arm = arm_data["/observation/right_arm/joint_state"][0]["data"]["pos"]
                    print(f"  arm action数据异常，使用observation")
            else:
                left_arm = arm_data["/observation/left_arm/joint_state"][0]["data"]["pos"]
                right_arm = arm_data["/observation/right_arm/joint_state"][0]["data"]["pos"]
                print(f"  没有arm action，使用observation")
            
            left_hand = hand_data['frames'][0]["action"]["left_hand"]
            right_hand = hand_data['frames'][0]["action"]["right_hand"]
            
            action = np.concatenate([right_arm, right_hand, left_arm, left_hand])
            print(f"  action维度: {action.shape}")
            print(f"    right_arm: {len(right_arm)}")
            print(f"    right_hand: {len(right_hand)}")
            print(f"    left_arm: {len(left_arm)}")
            print(f"    left_hand: {len(left_hand)}")
            print(f"    总计: {len(action)}")
            
            if len(action) != 36:
                self.log_warning(f"action维度 {len(action)} != 36")
        
        except Exception as e:
            self.log_issue(f"数据范围验证失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ============ 步骤4: 代码逻辑Review ============
    def step4_code_review(self):
        """代码逻辑review"""
        print("\n" + "="*60)
        print("步骤4: 代码逻辑Review")
        print("="*60)
        
        print("\n4.1 潜在问题:")
        
        # Issue 1: 测试模式的bug
        print("\n  问题1: 测试模式的不一致")
        print("    代码第25行: self.episode_paths[:2]")
        print("    但注释说'只转换前3个episodes'")
        self.log_warning("测试模式切片[:2]只取2个，但注释说3个")
        
        # Issue 2: 错误处理
        print("\n  问题2: 错误处理")
        print("    第246-249行: try-except捕获了错误但继续执行")
        print("    建议: 记录失败的episode列表，最后汇总报告")
        self.log_warning("失败的episode没有被记录，难以追踪")
        
        # Issue 3: 图像格式假设
        print("\n  问题3: 图像格式处理")
        print("    第218-219行: 假设灰度图要转RGB")
        print("    但实际数据可能都是RGB，这个检查是好的")
        self.log_ok("灰度图转RGB的处理是合理的")
        
        # Issue 4: 数据类型转换
        print("\n  问题4: 度转弧度")
        print("    第100-101行: 灵巧手observation从度转弧度")
        print("    但action没有转换")
        self.log_warning("需要确认灵巧手action的单位是什么")
        
        print("\n4.2 优化建议:")
        
        suggestions = [
            "添加数据验证：每个episode转换后验证维度和数值范围",
            "添加进度保存：支持断点续传，避免失败后重新开始",
            "内存优化：不要一次加载所有帧的图像路径，按需加载",
            "并行处理：如果IO不是瓶颈，可以考虑多进程转换",
            "日志记录：使用logging模块而不是print",
            "配置文件：将features定义放到配置文件中"
        ]
        
        for i, sug in enumerate(suggestions, 1):
            print(f"  {i}. {sug}")
    
    # ============ 步骤5: 验证LeRobot输出 ============
    def step5_verify_lerobot_output(self):
        """验证LeRobot输出的完整性"""
        print("\n" + "="*60)
        print("步骤5: 验证LeRobot输出")
        print("="*60)
        
        if not self.lerobot_dir.exists():
            self.log_warning(f"LeRobot输出目录不存在: {self.lerobot_dir}")
            print("  请先运行转换脚本生成数据")
            return
        
        print(f"\n检查输出目录: {self.lerobot_dir}")
        
        # 5.1 检查目录结构
        print("\n5.1 目录结构:")
        for item in self.lerobot_dir.iterdir():
            if item.is_dir():
                file_count = len(list(item.iterdir()))
                print(f"  📁 {item.name}/ ({file_count} 文件)")
            else:
                size_mb = item.stat().st_size / 1024 / 1024
                print(f"  📄 {item.name} ({size_mb:.2f} MB)")
        
        # 5.2 检查元数据文件
        print("\n5.2 元数据文件:")
        meta_files = ["meta.json", "stats.json"]
        for fname in meta_files:
            fpath = self.lerobot_dir / fname
            if fpath.exists():
                self.log_ok(f"{fname} 存在")
                try:
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    print(f"    键: {list(data.keys())}")
                except Exception as e:
                    self.log_issue(f"{fname} 读取失败: {e}")
            else:
                self.log_warning(f"{fname} 不存在")
        
        # 5.3 尝试加载LeRobot数据集
        print("\n5.3 加载LeRobot数据集:")
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            
            dataset = LeRobotDataset(
                repo_id="baai/bimanual_dexhand",
                root=str(self.lerobot_dir)
            )
            
            self.log_ok(f"数据集加载成功")
            print(f"    总帧数: {len(dataset)}")
            print(f"    Episodes: {dataset.num_episodes}")
            print(f"    FPS: {dataset.fps}")
            
            # 检查第一帧
            if len(dataset) > 0:
                print("\n  检查第一帧数据:")
                sample = dataset[0]
                for key, value in sample.items():
                    if isinstance(value, np.ndarray):
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                    elif hasattr(value, 'shape'):
                        print(f"    {key}: shape={value.shape}")
                    else:
                        print(f"    {key}: {type(value)}")
            
        except Exception as e:
            self.log_issue(f"加载LeRobot数据集失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ============ 主验证流程 ============
    def run_full_verification(self):
        """运行完整验证流程"""
        print("\n" + "="*70)
        print(" BSON到LeRobot转换 - 完整验证")
        print("="*70)
        
        # 运行所有验证步骤
        self.step1_verify_bson_structure()
        self.step2_verify_dimensions()
        self.step3_verify_data_ranges()
        self.step4_code_review()
        self.step5_verify_lerobot_output()
        
        # 总结
        print("\n" + "="*70)
        print(" 验证总结")
        print("="*70)
        
        if self.issues:
            print(f"\n发现 {len(self.issues)} 个问题:")
            for issue in self.issues:
                print(f"  {issue}")
        else:
            print("\n✅ 没有发现严重问题!")
        
        if self.warnings:
            print(f"\n发现 {len(self.warnings)} 个警告:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="验证BSON数据和LeRobot输出")
    parser.add_argument("--bson-dir", default="data/baai/data/", help="BSON数据目录")
    parser.add_argument("--lerobot-dir", default="data/baai/data/lerobot_baai", help="LeRobot输出目录")
    args = parser.parse_args()
    
    verifier = BSONLeRobotVerifier(args.bson_dir, args.lerobot_dir)
    verifier.run_full_verification()

