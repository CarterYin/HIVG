#!/bin/bash
# 简单的CLIP模型下载脚本

echo "开始下载CLIP模型..."
echo "使用镜像源: https://hf-mirror.com"

# 创建clip目录
mkdir -p /home/yinchao/HiVG/clip
cd /home/yinchao/HiVG/clip

# 下载模型
echo ""
echo "1. 下载 clip-vit-large-patch14-336..."
if [ ! -d "clip-vit-large-patch14-336" ]; then
    git clone https://hf-mirror.com/openai/clip-vit-large-patch14-336px clip-vit-large-patch14-336
else
    echo "   目录已存在，跳过"
fi

echo ""
echo "2. 下载 clip-vit-large-patch14..."
if [ ! -d "clip-vit-large-patch14" ]; then
    git clone https://hf-mirror.com/openai/clip-vit-large-patch14 clip-vit-large-patch14
else
    echo "   目录已存在，跳过"
fi

echo ""
echo "3. 下载 clip-vit-base-patch32..."
if [ ! -d "clip-vit-base-patch32" ]; then
    git clone https://hf-mirror.com/openai/clip-vit-base-patch32 clip-vit-base-patch32
else
    echo "   目录已存在，跳过"
fi

echo ""
echo "4. 下载 clip-vit-base-patch16..."
if [ ! -d "clip-vit-base-patch16" ]; then
    git clone https://hf-mirror.com/openai/clip-vit-base-patch16 clip-vit-base-patch16
else
    echo "   目录已存在，跳过"
fi

echo ""
echo "下载完成！"
echo ""
echo "正在更新HiVG.py中的模型路径..."

# 更新模型路径
cd /home/yinchao/HiVG
sed -i 's|/path_to_clip/clip-vit-large-patch14-336|/home/yinchao/HiVG/clip/clip-vit-large-patch14-336|g' models/HiVG.py
sed -i 's|/path_to_clip/clip-vit-large-patch14|/home/yinchao/HiVG/clip/clip-vit-large-patch14|g' models/HiVG.py
sed -i 's|/path_to_clip/clip-vit-base-patch32|/home/yinchao/HiVG/clip/clip-vit-base-patch32|g' models/HiVG.py
sed -i 's|/path_to_clip/clip-vit-base-patch16|/home/yinchao/HiVG/clip/clip-vit-base-patch16|g' models/HiVG.py

echo "模型路径已更新！"
echo ""
echo "所有操作完成！"
