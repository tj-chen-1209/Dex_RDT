#!/bin/bash
# ==============================================================================
# 生成预解码图像缓存 - 提升训练速度3-4倍
# ==============================================================================
# 功能：将JPEG图像预解码为numpy数组，消除训练时的JPEG解码开销
# 效果：训练速度从 16 it/s 提升到 60-80 it/s
# 磁盘：约需要 10GB 空间（压缩后约 5GB）
# ==============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          ⚡ 生成高速缓存 - 预解码图像                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# 数据集路径
DATASET_PATH="data/baai/data/lerobot_baai"

# 检查数据集是否存在
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ 错误: 数据集不存在: $DATASET_PATH"
    echo "💡 请先确保LeRobot格式数据集已准备好"
    exit 1
fi

# 选择模式
echo "请选择缓存模式："
echo ""
echo "  1) 🚀 标准模式 (预解码图像, ~10GB, 速度提升3-4倍)"
echo "  2) 📦 压缩模式 (预解码+压缩, ~5GB, 速度提升3-4倍)"
echo "  3) 💾 路径模式 (仅路径信息, ~100MB, 无速度提升)"
echo ""
read -p "请输入选项 [1/2/3]: " mode

case $mode in
    1)
        echo ""
        echo "✅ 选择: 标准预解码模式"
        echo "💾 预计磁盘空间: ~10GB"
        echo "📈 预计训练速度: 60-80 it/s"
        echo ""
        
        python preprocess_lerobot_cache.py \
            --dataset_path=$DATASET_PATH \
            --save_images \
            --decode_images
        ;;
    2)
        echo ""
        echo "✅ 选择: 压缩预解码模式"
        echo "💾 预计磁盘空间: ~5GB"
        echo "📈 预计训练速度: 60-80 it/s"
        echo ""
        
        python preprocess_lerobot_cache.py \
            --dataset_path=$DATASET_PATH \
            --save_images \
            --decode_images \
            --compress_images
        ;;
    3)
        echo ""
        echo "✅ 选择: 路径信息模式"
        echo "💾 预计磁盘空间: ~100MB"
        echo "📈 训练速度: 16 it/s (无提升)"
        echo ""
        
        python preprocess_lerobot_cache.py \
            --dataset_path=$DATASET_PATH \
            --save_images
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$mode" == "1" ] || [ "$mode" == "2" ]; then
    echo "✅ 预解码缓存生成完成！"
    echo ""
    echo "📊 性能提升预期："
    echo "  - 当前速度: 16 it/s"
    echo "  - 优化后: 60-80 it/s"
    echo "  - 提升倍数: 3.75-5倍"
    echo ""
    echo "💡 使用方法："
    echo "  直接运行训练脚本即可，数据加载器会自动检测并使用预解码缓存"
    echo "  bash train_baai.sh"
    echo ""
    echo "📂 缓存位置:"
    echo "  ${DATASET_PATH}/cache/"
else
    echo "✅ 路径缓存生成完成！"
    echo ""
    echo "⚠️  注意: 此模式不包含预解码，训练速度不会提升"
    echo "💡 如需提升速度，请选择模式1或2重新生成缓存"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""



