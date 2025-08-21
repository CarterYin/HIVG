#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载HiVG项目所需的CLIP预训练模型
"""

import os
import sys
from pathlib import Path
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
import shutil

# 定义模型列表
MODELS = [
    {
        'name': 'clip-vit-large-patch14-336',
        'model_id': 'openai/clip-vit-large-patch14-336px',
        'target_dir': 'clip-vit-large-patch14-336'
    },
    {
        'name': 'clip-vit-large-patch14',
        'model_id': 'openai/clip-vit-large-patch14',
        'target_dir': 'clip-vit-large-patch14'
    },
    {
        'name': 'clip-vit-base-patch32',
        'model_id': 'openai/clip-vit-base-patch32',
        'target_dir': 'clip-vit-base-patch32'
    },
    {
        'name': 'clip-vit-base-patch16',
        'model_id': 'openai/clip-vit-base-patch16',
        'target_dir': 'clip-vit-base-patch16'
    }
]

def download_model(model_info, base_path):
    """下载单个模型"""
    print(f"\n{'='*60}")
    print(f"正在下载: {model_info['name']}")
    print(f"模型ID: {model_info['model_id']}")
    print(f"{'='*60}")
    
    target_path = os.path.join(base_path, model_info['target_dir'])
    
    # 如果目录已存在，询问是否覆盖
    if os.path.exists(target_path):
        print(f"目录 {target_path} 已存在")
        response = input("是否覆盖？(y/n): ").lower()
        if response != 'y':
            print(f"跳过 {model_info['name']}")
            return
        else:
            shutil.rmtree(target_path)
    
    os.makedirs(target_path, exist_ok=True)
    
    try:
        # 下载模型
        print(f"下载模型到: {target_path}")
        model = CLIPModel.from_pretrained(
            model_info['model_id'],
            cache_dir=target_path,
            force_download=False,
            resume_download=True
        )
        
        # 保存到目标目录
        print(f"保存模型...")
        model.save_pretrained(target_path)
        
        # 下载并保存tokenizer
        print(f"下载tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(
            model_info['model_id'],
            cache_dir=target_path
        )
        tokenizer.save_pretrained(target_path)
        
        # 下载并保存image processor
        print(f"下载image processor...")
        processor = CLIPImageProcessor.from_pretrained(
            model_info['model_id'],
            cache_dir=target_path
        )
        processor.save_pretrained(target_path)
        
        print(f"✅ {model_info['name']} 下载完成！")
        
    except Exception as e:
        print(f"❌ 下载 {model_info['name']} 时出错: {str(e)}")
        return False
    
    return True

def update_model_paths():
    """更新HiVG.py中的模型路径"""
    hivg_path = "/home/yinchao/HiVG/models/HiVG.py"
    clip_base = "/home/yinchao/HiVG/clip"
    
    print(f"\n更新模型路径...")
    
    # 读取文件
    with open(hivg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换路径
    replacements = [
        ('/path_to_clip/clip-vit-large-patch14-336', f'{clip_base}/clip-vit-large-patch14-336'),
        ('/path_to_clip/clip-vit-large-patch14', f'{clip_base}/clip-vit-large-patch14'),
        ('/path_to_clip/clip-vit-base-patch32', f'{clip_base}/clip-vit-base-patch32'),
        ('/path_to_clip/clip-vit-base-patch16', f'{clip_base}/clip-vit-base-patch16'),
    ]
    
    for old_path, new_path in replacements:
        content = content.replace(old_path, new_path)
    
    # 写回文件
    with open(hivg_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 模型路径已更新！")

def main():
    """主函数"""
    # 基础路径
    base_path = "/home/yinchao/HiVG/clip"
    
    print("HiVG CLIP模型下载工具")
    print(f"目标目录: {base_path}")
    
    # 创建基础目录
    os.makedirs(base_path, exist_ok=True)
    
    # 询问是否下载所有模型
    print(f"\n将下载以下{len(MODELS)}个模型:")
    for i, model in enumerate(MODELS, 1):
        print(f"{i}. {model['name']} ({model['model_id']})")
    
    response = input("\n是否下载所有模型？(y/n): ").lower()
    
    if response == 'y':
        # 下载所有模型
        success_count = 0
        for model in MODELS:
            if download_model(model, base_path):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"下载完成！成功: {success_count}/{len(MODELS)}")
        
        if success_count == len(MODELS):
            # 询问是否更新路径
            response = input("\n是否自动更新HiVG.py中的模型路径？(y/n): ").lower()
            if response == 'y':
                update_model_paths()
    else:
        # 选择性下载
        print("\n请选择要下载的模型（输入数字，多个用空格分隔）:")
        choices = input("选择: ").strip().split()
        
        try:
            indices = [int(c) - 1 for c in choices]
            success_count = 0
            
            for idx in indices:
                if 0 <= idx < len(MODELS):
                    if download_model(MODELS[idx], base_path):
                        success_count += 1
                else:
                    print(f"无效的选择: {idx + 1}")
            
            print(f"\n{'='*60}")
            print(f"下载完成！成功: {success_count}/{len(indices)}")
            
        except ValueError:
            print("输入错误！")
            return
    
    print("\n提示：如果下载中断，可以重新运行此脚本继续下载。")
    print("模型保存在:", base_path)

if __name__ == "__main__":
    main()
