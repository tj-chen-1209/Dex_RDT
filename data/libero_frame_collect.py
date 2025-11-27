import h5py
import os
import glob

# 定义所有 LIBERO 数据集
datasets = ['libero_10', 'libero_90', 'libero_spatial', 'libero_object', 'libero_goal']
base_dir = '/home/chensiqi/chensiqi/RDT_libero_finetune/data/datasets'

print("=" * 80)
print("LIBERO 数据集样本量统计")
print("=" * 80)

grand_total_steps = 0
grand_total_demos = 0
grand_total_files = 0

dataset_summary = []

for data_name in datasets:
    print(f"\n{'='*80}")
    print(f"数据集: {data_name}")
    print(f"{'='*80}")
    
    dataset_path = os.path.join(base_dir, data_name)
    files = sorted(glob.glob(os.path.join(dataset_path, '*.hdf5')))
    
    if not files:
        print(f"  未找到 HDF5 文件")
        continue
    
    total_steps = 0
    total_demos = 0
    file_count = len(files)
    
    for i, path in enumerate(files, 1):
        try:
            with h5py.File(path, 'r') as f:
                if 'data' not in f:
                    print(f"  [{i}/{file_count}] {os.path.basename(path)} - 缺少 'data' 键")
                    continue
                
                data = f['data']
                demo_count = len(data.keys())
                file_steps = 0
                
                for demo_key in data.keys():
                    demo = data[demo_key]
                    if 'actions' in demo:
                        steps = int(demo['actions'].shape[0])
                        file_steps += steps
                
                total_steps += file_steps
                total_demos += demo_count
                
        except Exception as e:
            print(f"  [{i}/{file_count}] {os.path.basename(path)} - 错误: {e}")
    
    print(f"\n{data_name} 统计: 文件数={file_count}, 总Demos={total_demos}, 总Frames={total_steps}")
    
    dataset_summary.append({
        'name': data_name,
        'files': file_count,
        'demos': total_demos,
        'steps': total_steps
    })
    
    grand_total_steps += total_steps
    grand_total_demos += total_demos
    grand_total_files += file_count

# 输出总结
print("\n" + "=" * 80)
print("总体统计汇总")
print("=" * 80)

for ds in dataset_summary:
    print(f"{ds['name']:20s} | 文件: {ds['files']:4d} | Demos: {ds['demos']:6d} | Frames: {ds['steps']:8d}")

print("\n" + "=" * 80)
print(f"所有 LIBERO 数据集总计:")
print(f"  总文件数:    {grand_total_files:,}")
print(f"  总 Demos:    {grand_total_demos:,}")
print(f"  总 Frames:   {grand_total_steps:,}")
print("=" * 80)

# 计算与训练配置的关系
train_batch_size = 32
max_train_steps = 200000
total_sample_slots = train_batch_size * max_train_steps

print(f"\n{'='*80}")
print("与训练配置的关系:")
print(f"{'='*80}")
print(f"训练配置:")
print(f"  train_batch_size = {train_batch_size}")
print(f"  max_train_steps  = {max_train_steps:,}")
print(f"  总样本槽位     = {total_sample_slots:,}")
print(f"\n数据集:")
print(f"  总 Frames       = {grand_total_steps:,}")
print(f"\n分析:")
if grand_total_steps > 0:
    epochs = total_sample_slots / grand_total_steps
    print(f"  完整遍历次数 (Epochs) = {epochs:.2f}")
    if epochs > 1:
        print(f"  ⚠️  数据会被重复使用约 {epochs:.1f} 次，可能存在过拟合风险")
    elif epochs < 0.5:
        print(f"  ⚠️  只会使用 {epochs*100:.1f}% 的数据，可能欠拟合")
    else:
        print(f"  ✓ 训练步数设置合理")
print("=" * 80)


import h5py
import os
import glob

# 定义所有 LIBERO 数据集
datasets = {
    'libero_10': 'only_demo_0',      # 只取 demo_0
    'libero_90': 'all_demos',        # 所有 demos
    'libero_spatial': 'only_demo_0', # 只取 demo_0
    'libero_object': 'only_demo_0',  # 只取 demo_0
    'libero_goal': 'only_demo_0'     # 只取 demo_0
}

base_dir = '/home/chensiqi/chensiqi/RDT_libero_finetune/data/datasets'

print("=" * 80)
print("LIBERO 数据集实际使用的 Frame 统计")
print("（libero_90 使用所有 demos，其他只使用每个文件的 demo_0）")
print("=" * 80)

grand_total_steps = 0
grand_total_demos = 0
grand_total_files = 0

dataset_summary = []

for data_name, mode in datasets.items():
    print(f"\n{'='*80}")
    print(f"数据集: {data_name} (模式: {mode})")
    print(f"{'='*80}")
    
    dataset_path = os.path.join(base_dir, data_name)
    files = sorted(glob.glob(os.path.join(dataset_path, '*.hdf5')))
    
    if not files:
        print(f"  未找到 HDF5 文件")
        continue
    
    total_steps = 0
    total_demos = 0
    file_count = len(files)
    
    for i, path in enumerate(files, 1):
        try:
            with h5py.File(path, 'r') as f:
                if 'data' not in f:
                    print(f"  [{i}/{file_count}] {os.path.basename(path)} - 缺少 'data' 键")
                    continue
                
                data = f['data']
                
                if mode == 'only_demo_0':
                    # 只统计 demo_0
                    if 'demo_0' in data:
                        demo = data['demo_0']
                        if 'actions' in demo:
                            steps = int(demo['actions'].shape[0])
                            total_steps += steps
                            total_demos += 1
                        else:
                            print(f"    ⚠️  demo_0 缺少 'actions'")
                    else:
                        print(f"  [{i}/{file_count}] {os.path.basename(path)} - 缺少 demo_0")
                        
                else:  # all_demos (libero_90)
                    demo_count = len(data.keys())
                    file_steps = 0
                    
                    for demo_key in data.keys():
                        demo = data[demo_key]
                        if 'actions' in demo:
                            steps = int(demo['actions'].shape[0])
                            file_steps += steps
                    
                    total_steps += file_steps
                    total_demos += demo_count
                
        except Exception as e:
            print(f"  [{i}/{file_count}] {os.path.basename(path)} - 错误: {e}")
    
    print(f"\n{data_name} 统计:")
    print(f"  文件数: {file_count}")
    print(f"  实际使用的 Demos: {total_demos}")
    print(f"  实际使用的 Frames: {total_steps}")
    
    dataset_summary.append({
        'name': data_name,
        'mode': mode,
        'files': file_count,
        'demos': total_demos,
        'steps': total_steps
    })
    
    grand_total_steps += total_steps
    grand_total_demos += total_demos
    grand_total_files += file_count

# 输出总结
print("\n" + "=" * 80)
print("总体统计汇总")
print("=" * 80)

for ds in dataset_summary:
    mode_str = "所有demos" if ds['mode'] == 'all_demos' else "仅demo_0"
    print(f"{ds['name']:20s} ({mode_str:10s}) | 文件: {ds['files']:4d} | Demos: {ds['demos']:6d} | Frames: {ds['steps']:8d}")

print("\n" + "=" * 80)
print(f"所有 LIBERO 数据集实际使用总计:")
print(f"  总文件数:          {grand_total_files:,}")
print(f"  实际使用的 Demos:  {grand_total_demos:,}")
print(f"  实际使用的 Frames: {grand_total_steps:,}")
print("=" * 80)

# 计算与训练配置的关系
train_batch_size = 32
max_train_steps = 200000
total_sample_slots = train_batch_size * max_train_steps

print(f"\n{'='*80}")
print("与训练配置的关系:")
print(f"{'='*80}")
print(f"训练配置:")
print(f"  train_batch_size = {train_batch_size}")
print(f"  max_train_steps  = {max_train_steps:,}")
print(f"  总样本槽位     = {total_sample_slots:,}")
print(f"\n数据集:")
print(f"  实际使用的 Frames = {grand_total_steps:,}")
print(f"\n分析:")
if grand_total_steps > 0:
    epochs = total_sample_slots / grand_total_steps
    print(f"  完整遍历次数 (Epochs) = {epochs:.2f}")
    
    # 给出建议
    if epochs > 20:
        print(f"  ⚠️  数据会被重复使用约 {epochs:.1f} 次，过拟合风险极高！")
        recommended_steps = int(grand_total_steps * 10 / train_batch_size)
        print(f"  💡 建议: 减少训练步数到 {recommended_steps:,} (约10 epochs)")
    elif epochs > 10:
        print(f"  ⚠️  数据会被重复使用约 {epochs:.1f} 次，存在过拟合风险")
        recommended_steps = int(grand_total_steps * 8 / train_batch_size)
        print(f"  💡 建议: 可考虑减少到 {recommended_steps:,} (约8 epochs)")
    elif epochs > 5:
        print(f"  ✓ 数据会被重复使用约 {epochs:.1f} 次，合理范围（5-10 epochs）")
    elif epochs >= 1:
        print(f"  ✓ 数据会被重复使用约 {epochs:.1f} 次，较为保守")
    else:
        print(f"  ⚠️  只会使用 {epochs*100:.1f}% 的数据，可能欠拟合")
        recommended_steps = int(grand_total_steps * 5 / train_batch_size)
        print(f"  💡 建议: 增加训练步数到 {recommended_steps:,} (约5 epochs)")
        
print("=" * 80)