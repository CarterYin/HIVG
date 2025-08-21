#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用镜像源下载HiVG项目所需的CLIP预训练模型
"""

import os
import sys
from pathlib import Path
import subprocess
import shutil

# 设置环境变量使用镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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

def print_manual_download_instructions():
    """打印手动下载说明"""
    print("\n" + "="*80)
    print("手动下载说明")
    print("="*80)
    print("\n如果自动下载失败，您可以手动下载模型：")
    print("\n方法1: 使用git clone（推荐）")
    print("```bash")
    print("cd /home/yinchao/HiVG/clip")
    for model in MODELS:
        print(f"git clone https://huggingface.co/{model['model_id']} {model['target_dir']}")
    print("```")
    
    print("\n方法2: 使用huggingface-cli（需要先安装: pip install huggingface-hub）")
    print("```bash")
    print("cd /home/yinchao/HiVG/clip")
    for model in MODELS:
        print(f"huggingface-cli download {model['model_id']} --local-dir {model['target_dir']}")
    print("```")
    
    print("\n方法3: 使用镜像源")
    print("```bash")
    print("export HF_ENDPOINT=https://hf-mirror.com")
    print("# 然后使用方法1或方法2")
    print("```")
    
    print("\n方法4: 从浏览器下载")
    print("访问以下链接，点击'Files and versions'，下载所有文件到对应目录：")
    for model in MODELS:
        print(f"- {model['name']}: https://huggingface.co/{model['model_id']}")
        print(f"  保存到: /home/yinchao/HiVG/clip/{model['target_dir']}/")
    print("\n" + "="*80)

def download_with_git(model_info, base_path):
    """使用git clone下载模型"""
    target_path = os.path.join(base_path, model_info['target_dir'])
    
    if os.path.exists(target_path):
        print(f"目录 {target_path} 已存在，跳过")
        return True
    
    print(f"使用git clone下载 {model_info['name']}...")
    
    # 使用镜像源
    mirror_url = f"https://hf-mirror.com/{model_info['model_id']}"
    
    try:
        # 执行git clone
        cmd = ['git', 'clone', mirror_url, target_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {model_info['name']} 下载完成！")
            return True
        else:
            print(f"❌ git clone失败: {result.stderr}")
            # 尝试使用原始URL
            print(f"尝试使用原始URL...")
            original_url = f"https://huggingface.co/{model_info['model_id']}"
            cmd = ['git', 'clone', original_url, target_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {model_info['name']} 下载完成！")
                return True
            else:
                print(f"❌ 下载失败: {result.stderr}")
                return False
    
    except Exception as e:
        print(f"❌ 下载出错: {str(e)}")
        return False

def download_with_transformers(model_info, base_path):
    """使用transformers库下载（带镜像）"""
    try:
        from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
        
        target_path = os.path.join(base_path, model_info['target_dir'])
        
        if os.path.exists(target_path):
            print(f"目录 {target_path} 已存在")
            response = input("是否覆盖？(y/n): ").lower()
            if response != 'y':
                return True
            else:
                shutil.rmtree(target_path)
        
        os.makedirs(target_path, exist_ok=True)
        
        print(f"使用transformers下载 {model_info['name']}...")
        
        # 下载模型
        model = CLIPModel.from_pretrained(model_info['model_id'])
        model.save_pretrained(target_path)
        
        # 下载tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(model_info['model_id'])
        tokenizer.save_pretrained(target_path)
        
        # 下载image processor
        processor = CLIPImageProcessor.from_pretrained(model_info['model_id'])
        processor.save_pretrained(target_path)
        
        print(f"✅ {model_info['name']} 下载完成！")
        return True
        
    except ImportError:
        print("❌ transformers库未安装")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        return False

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

def check_git_installed():
    """检查git是否安装"""
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def main():
    """主函数"""
    base_path = "/home/yinchao/HiVG/clip"
    
    print("HiVG CLIP模型下载工具（镜像版）")
    print(f"目标目录: {base_path}")
    print(f"使用镜像源: {os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')}")
    
    # 创建基础目录
    os.makedirs(base_path, exist_ok=True)
    
    # 检查git
    has_git = check_git_installed()
    if not has_git:
        print("\n⚠️ 警告: 未检测到git，将尝试使用transformers库下载")
    
    # 选择下载方式
    print("\n选择下载方式:")
    print("1. 使用git clone下载（推荐，需要安装git）")
    print("2. 使用transformers库下载")
    print("3. 查看手动下载说明")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == '3':
        print_manual_download_instructions()
        return
    
    # 下载模型
    print(f"\n将下载以下{len(MODELS)}个模型:")
    for i, model in enumerate(MODELS, 1):
        print(f"{i}. {model['name']} ({model['model_id']})")
    
    response = input("\n是否下载所有模型？(y/n): ").lower()
    
    if response == 'y':
        success_count = 0
        
        for model in MODELS:
            print(f"\n{'='*60}")
            print(f"正在下载: {model['name']}")
            print(f"模型ID: {model['model_id']}")
            print(f"{'='*60}")
            
            if choice == '1' and has_git:
                success = download_with_git(model, base_path)
            else:
                success = download_with_transformers(model, base_path)
            
            if success:
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"下载完成！成功: {success_count}/{len(MODELS)}")
        
        if success_count == len(MODELS):
            # 询问是否更新路径
            response = input("\n是否自动更新HiVG.py中的模型路径？(y/n): ").lower()
            if response == 'y':
                update_model_paths()
    
    # 如果有失败的，显示手动下载说明
    if success_count < len(MODELS):
        print_manual_download_instructions()

if __name__ == "__main__":
    main()
