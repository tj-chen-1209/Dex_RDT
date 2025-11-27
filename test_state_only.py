#!/usr/bin/env python
"""测试 state_only 模式"""
import sys
sys.path.insert(0, '.')

from data.bson_Dex_dataset import BsonDexDataset

print("Initializing dataset...")
ds = BsonDexDataset()
print(f"Dataset size: {len(ds)} episodes")

print("\n--- Testing state_only=False (single timestep) ---")
try:
    sample = ds.get_item(index=0, state_only=False)
    print("✅ Success!")
    print(f"  State shape: {sample['state'].shape}")
    print(f"  Actions shape: {sample['actions'].shape}")
    print(f"  cam_high shape: {sample['cam_high'].shape}")
    print(f"  cam_left_wrist shape: {sample['cam_left_wrist'].shape}")
    print(f"  cam_right_wrist shape: {sample['cam_right_wrist'].shape}")
    print(f"  cam_third_view shape: {sample['cam_third_view'].shape}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Testing state_only=True (full trajectory) ---")
try:
    sample = ds.get_item(index=0, state_only=True)
    print("✅ Success!")
    print(f"  State trajectory shape: {sample['state'].shape}")
    print(f"  Action trajectory shape: {sample['action'].shape}")
    print(f"  First 3 states:\n{sample['state'][:3]}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Testing 5 random samples ---")
for i in range(5):
    try:
        sample = ds.get_item(state_only=False)
        print(f"Sample {i}: state={sample['state'].shape}, actions={sample['actions'].shape}")
    except Exception as e:
        print(f"Sample {i} failed: {e}")

print("\n✅ All tests completed!")


