#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HiVG Demo Script
用于单张图片的视觉定位（Visual Grounding）推理和可视化
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, Any

# 导入项目模块
from models import build_model
from utils.misc import nested_tensor_from_tensor_list
from utils.box_utils import xywh2xyxy
import datasets.transforms as T


def get_args_parser():
    """解析命令行参数"""
    parser = argparse.ArgumentParser('HiVG Demo', add_help=False)
    
    # 必需参数
    parser.add_argument('--input_image_path', type=str, required=True,
                        help='输入图片路径')
    parser.add_argument('--prompt', type=str, required=True,
                        help='用于检测/分割对象的文本描述')
    parser.add_argument('--output_image_path', type=str, required=True,
                        help='输出可视化结果路径')
    
    # 模型相关参数
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/home/yinchao/HiVG/mixup_pretraining_large/mixup/best_checkpoint.pth',
                        help='模型checkpoint路径')
    parser.add_argument('--model', type=str, default='ViT-L/14',
                        choices=['ViT-L/14', 'ViT-B/16', 'ViT-B/32'],
                        help='CLIP模型版本')
    # parser.add_argument('--imsize', type=int, default=224,
    #                     help='输入图像大小（建议使用224以兼容预训练模型）')
    parser.add_argument('--imsize', type=int, default=640,
                        help='输入图像大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='使用的设备 (cuda/cpu)')
    
    # 视觉化参数
    parser.add_argument('--box_color', type=str, default='red',
                        help='边界框颜色')
    parser.add_argument('--text_color', type=str, default='white',
                        help='文本颜色')
    parser.add_argument('--box_thickness', type=int, default=3,
                        help='边界框线条粗细')
    parser.add_argument('--show_confidence', action='store_true',
                        help='是否显示置信度分数')
    
    # 其他必需的参数（从训练代码中提取）
    # parser.add_argument('--vl_hidden_dim', type=int, default=768,
    #                     help='视觉-语言变换器的隐藏维度')
    # parser.add_argument('--vl_enc_layers', type=int, default=6,
    #                     help='视觉-语言编码器层数')
    # parser.add_argument('--vl_dec_layers', type=int, default=6,
    #                     help='视觉-语言解码器层数')
    # parser.add_argument('--vl_nheads', type=int, default=8,
    #                     help='注意力头数')
    # parser.add_argument('--vl_dim_feedforward', type=int, default=2048,
    #                     help='前馈网络维度')
    # parser.add_argument('--vl_dropout', type=float, default=0.1,
    #                     help='Dropout率')
    # parser.add_argument('--max_query_len', type=int, default=77,
    #                     help='最大查询长度')
    # parser.add_argument('--use_vl_type_embed', action='store_true',
    #                     help='是否使用VL类型嵌入')
    # parser.add_argument('--normalize_before', action='store_true', default=True,
    #                     help='是否在前面进行归一化')
    # parser.add_argument('--warmup', action='store_true',
    #                     help='是否使用warmup模式')
    # parser.add_argument('--hi_lora_stage', type=int, default=0,
    #                     help='HiLoRA阶段')
    # parser.add_argument('--use_mask_loss', action='store_true',
    #                     help='是否使用分割损失')
    # parser.add_argument('--mixup_pretrain', action='store_true', default=True,
    #                     help='是否使用mixup预训练')
    # parser.add_argument('--enable_adaptive_weights', action='store_true', default=True,
    #                     help='是否启用自适应权重')
    # parser.add_argument('--use_seg_mask', action='store_true',
    #                     help='是否使用分割掩码')
    # parser.add_argument('--dataset', type=str, default='mixup',
    #                     help='数据集名称')
    
    return parser


def load_model(args):
    """加载预训练模型"""
    print(f"正在加载模型: {args.model}")
    
    # 从checkpoint加载原始args并更新关键配置
    if os.path.exists(args.checkpoint_path):
        print(f"加载checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
        original_args = checkpoint['args']
        
        # 使用checkpoint中的关键配置
        args.vl_hidden_dim = original_args.vl_hidden_dim  # 768
        args.max_query_len = original_args.max_query_len  # 77
        args.imsize = original_args.imsize  # 224
        args.vl_enc_layers = original_args.vl_enc_layers  # 6
        args.vl_dec_layers = original_args.vl_dec_layers  # 6
        args.vl_nheads = original_args.vl_nheads  # 8
        args.vl_dim_feedforward = original_args.vl_dim_feedforward  # 2048
        args.vl_dropout = original_args.vl_dropout  # 0.1
        args.normalize_before = original_args.normalize_before  # True
        
        # 设置mixup_pretrain属性
        if hasattr(original_args, 'mixup_pretrain'):
            args.mixup_pretrain = original_args.mixup_pretrain
        else:
            args.mixup_pretrain = True if original_args.dataset == 'mixup' else False
        
        # 设置enable_adaptive_weights属性
        if hasattr(original_args, 'enable_adaptive_weights'):
            args.enable_adaptive_weights = original_args.enable_adaptive_weights
        else:
            args.enable_adaptive_weights = True
        
        # 设置warmup属性
        if hasattr(original_args, 'warmup'):
            args.warmup = original_args.warmup
        else:
            args.warmup = False
            
        # 设置hi_lora_stage属性
        if hasattr(original_args, 'hi_lora_stage'):
            args.hi_lora_stage = original_args.hi_lora_stage
        else:
            args.hi_lora_stage = 0
        
        print(f"使用配置: imsize={args.imsize}, vl_hidden_dim={args.vl_hidden_dim}, mixup_pretrain={args.mixup_pretrain}")
        
        # 构建模型
        model = build_model(args)
        
        # 加载checkpoint
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"成功加载checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        raise FileNotFoundError(f"找不到checkpoint文件: {args.checkpoint_path}")
    
    # 设置为评估模式
    model.eval()
    model.to(args.device)
    
    return model


def preprocess_image(image_path: str, args):
    """预处理输入图像"""
    # 读取图像
    image = Image.open(image_path).convert("RGB")
    
    # 创建变换
    transform = T.Compose([
        T.RandomResize([args.imsize]),
        T.ToTensor(),
        T.NormalizeAndPad(size=args.imsize),
    ])
    
    # 应用变换
    input_dict = {'img': image, 'box': torch.tensor([0, 0, 1, 1])}  # dummy box
    transformed = transform(input_dict)
    
    img_tensor = transformed['img']
    img_mask = transformed['mask']
    
    # 创建nested tensor
    img_data = nested_tensor_from_tensor_list([img_tensor])
    img_data.mask = img_mask.unsqueeze(0)
    
    return img_data, image


@torch.no_grad()
def forward(model: Any, image: Image.Image, description: str) -> Dict:
    """
    处理单张图片的核心函数
    
    Args:
        model: HiVG模型
        image: PIL图像对象
        description: 文本描述
    
    Returns:
        Dict: 包含处理后的结果，可直接用于可视化
    """
    args = model.args if hasattr(model, 'args') else argparse.Namespace(
        imsize=640,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 预处理图像
    transform = T.Compose([
        T.RandomResize([args.imsize]),
        T.ToTensor(),
        T.NormalizeAndPad(size=args.imsize),
    ])
    
    input_dict = {'img': image, 'box': torch.tensor([0, 0, 1, 1])}
    transformed = transform(input_dict)
    
    img_tensor = transformed['img']
    img_mask = transformed['mask']
    
    # 创建batch
    img_data = nested_tensor_from_tensor_list([img_tensor])
    img_data.mask = img_mask.unsqueeze(0)
    img_data = img_data.to(args.device)
    
    # 准备文本输入
    text_data = [description]
    
    # 前向传播
    pred_box, logits_per_text, logits_per_image, visu_token_similarity, seg_mask = model(img_data, text_data)
    
    # 处理输出
    pred_box = pred_box.squeeze(0).cpu()  # [4] - cxcywh格式
    
    # 转换为xyxy格式并反归一化
    h, w = args.imsize, args.imsize
    pred_box_xyxy = xywh2xyxy(pred_box.unsqueeze(0)).squeeze(0)
    pred_box_xyxy = pred_box_xyxy * torch.tensor([w, h, w, h])
    
    # 计算在原始图像上的坐标
    original_h, original_w = image.height, image.width
    scale = min(args.imsize / original_w, args.imsize / original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    
    # 计算padding
    pad_w = (args.imsize - new_w) // 2
    pad_h = (args.imsize - new_h) // 2
    
    # 调整边界框坐标
    pred_box_xyxy[0] = (pred_box_xyxy[0] - pad_w) / scale
    pred_box_xyxy[1] = (pred_box_xyxy[1] - pad_h) / scale
    pred_box_xyxy[2] = (pred_box_xyxy[2] - pad_w) / scale
    pred_box_xyxy[3] = (pred_box_xyxy[3] - pad_h) / scale
    
    # 确保边界框在图像范围内
    pred_box_xyxy[0] = max(0, min(pred_box_xyxy[0], original_w))
    pred_box_xyxy[1] = max(0, min(pred_box_xyxy[1], original_h))
    pred_box_xyxy[2] = max(0, min(pred_box_xyxy[2], original_w))
    pred_box_xyxy[3] = max(0, min(pred_box_xyxy[3], original_h))
    
    # 处理分割掩码（如果有）
    seg_mask_processed = None
    if seg_mask is not None:
        seg_mask = seg_mask.squeeze().cpu().numpy()
        # 裁剪和缩放分割掩码以匹配原始图像
        seg_mask_cropped = seg_mask[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
        seg_mask_processed = np.array(Image.fromarray(seg_mask_cropped).resize((original_w, original_h)))
    
    # 返回结果
    return {
        'bbox': pred_box_xyxy.numpy(),  # [x1, y1, x2, y2]
        'bbox_normalized': pred_box.numpy(),  # [cx, cy, w, h] 归一化的
        'confidence': logits_per_text[0, 0].item() if logits_per_text is not None else 1.0,
        'segmentation_mask': seg_mask_processed,
        'text': description
    }


def visualize_result(image: Image.Image, result: Dict, args):
    """可视化结果"""
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 显示图像
    ax.imshow(image)
    ax.axis('off')
    
    # 绘制边界框
    bbox = result['bbox']
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # 创建矩形
    rect = patches.Rectangle((x1, y1), width, height, 
                            linewidth=args.box_thickness,
                            edgecolor=args.box_color,
                            facecolor='none')
    ax.add_patch(rect)
    
    # 添加文本标签
    label = result['text']
    if args.show_confidence:
        label += f" ({result['confidence']:.2f})"
    
    # 计算文本位置
    text_x = x1
    text_y = y1 - 5 if y1 > 20 else y1 + height + 15
    
    # 添加文本背景
    ax.text(text_x, text_y, label,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=args.box_color, alpha=0.8),
            color=args.text_color,
            fontsize=12,
            weight='bold')
    
    # 分割掩码不进行可视化显示
    
    plt.title(f"HiVG Visual Grounding Result")
    plt.tight_layout()
    
    # 保存结果
    plt.savefig(args.output_image_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到: {args.output_image_path}")
    
    # # 也可以使用PIL直接绘制（更简洁的版本）
    # draw_image = image.copy()
    # draw = ImageDraw.Draw(draw_image)
    # draw.rectangle(bbox.tolist(), outline=args.box_color, width=args.box_thickness)
    
    # # 保存PIL版本
    # pil_output = args.output_image_path.replace('.png', '_pil.png')
    # draw_image.save(pil_output)
    # print(f"PIL版本结果已保存到: {pil_output}")
    
    return fig


def main():
    """主函数"""
    # 解析参数
    parser = argparse.ArgumentParser('HiVG Demo', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("警告: CUDA不可用，使用CPU")
        args.device = 'cpu'
        device = torch.device('cpu')
    
    # 创建输出目录
    output_dir = Path(args.output_image_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = load_model(args)
    model.args = args  # 保存args供forward函数使用
    
    # 读取图像
    print(f"正在处理图像: {args.input_image_path}")
    image = Image.open(args.input_image_path).convert("RGB")
    
    # 执行推理
    print(f"使用描述进行推理: '{args.prompt}'")
    result = forward(model, image, args.prompt)
    
    # 打印结果
    print("\n推理结果:")
    print(f"  边界框 (x1,y1,x2,y2): {result['bbox']}")
    print(f"  边界框 (归一化 cx,cy,w,h): {result['bbox_normalized']}")
    # print(f"  置信度: {result['confidence']:.4f}")
    print(f"  是否有分割掩码: {'是' if result['segmentation_mask'] is not None else '否'}")
    
    # 可视化结果
    visualize_result(image, result, args)
    
    print("\n完成!")


if __name__ == '__main__':
    main()
