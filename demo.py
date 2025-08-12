import argparse
import os
from typing import Any, Dict, Tuple

import torch
from PIL import Image, ImageDraw, ImageColor
import numpy as np

from models import build_model
from utils.misc import NestedTensor


def _resize_by_long_side(pil_image: Image.Image, target_long_side: int) -> Tuple[Image.Image, float]:
    """
    Resize image so that the longer side equals target_long_side while keeping aspect ratio.
    Returns resized image and the resize ratio (new/old on the long side).
    """
    width, height = pil_image.size
    long_side = max(width, height)
    if long_side == target_long_side:
        return pil_image, 1.0
    ratio = float(target_long_side) / float(long_side)
    new_w = int(round(width * ratio))
    new_h = int(round(height * ratio))
    resized = pil_image.resize((new_w, new_h), Image.BICUBIC)
    return resized, ratio


def _to_tensor(pil_image: Image.Image) -> torch.Tensor:
    # Convert PIL (H, W, C) in [0,255] to tensor (C, H, W) in [0,1]
    np_img = np.array(pil_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
    return tensor


def _normalize(t: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    assert t.ndim == 3 and t.shape[0] == 3
    mean_t = torch.tensor(mean, dtype=t.dtype, device=t.device)[:, None, None]
    std_t = torch.tensor(std, dtype=t.dtype, device=t.device)[:, None, None]
    return (t - mean_t) / std_t


def _center_pad_to_square(t: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Center-pad tensor image to (3, size, size). Returns (padded_img, mask, top, left).
    Mask: 0 where image exists, 1 where padded.
    """
    _, h, w = t.shape
    dh = size - h
    dw = size - w
    if dh < 0 or dw < 0:
        raise ValueError(f"Input must be <= target size before padding. Got ({h},{w}) -> {size}")
    # Match utils.transforms.NormalizeAndPad padding policy
    top = int(round(dh / 2.0 - 0.1))
    left = int(round(dw / 2.0 - 0.1))
    padded = torch.zeros((3, size, size), dtype=t.dtype)
    mask = torch.ones((size, size), dtype=torch.int)
    padded[:, top: top + h, left: left + w] = t
    mask[top: top + h, left: left + w] = 0
    return padded, mask, top, left


def _xywh_norm_to_xyxy_pixels(box_xywh_norm: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """
    Convert normalized xywh (in [0,1]) to pixel xyxy given canvas size (W,H).
    box format: (x_center, y_center, width, height)
    """
    x_c, y_c, bw, bh = box_xywh_norm.unbind(-1)
    x0 = (x_c - bw / 2.0) * W
    y0 = (y_c - bh / 2.0) * H
    x1 = (x_c + bw / 2.0) * W
    y1 = (y_c + bh / 2.0) * H
    return torch.stack([x0, y0, x1, y1], dim=-1)


def forward(model: Any, image: Image.Image, description: str) -> Dict:
    """
    Single-image inference. Returns a dict with fields ready for visualization on the original image size.

    Returns keys:
    - box_xyxy: np.ndarray shape (4,), in original image pixel coordinates (x0,y0,x1,y1)
    - seg_mask: np.ndarray shape (H, W), float32 in [0,1], aligned to original image
    - logits_per_text: optional np.ndarray
    - logits_per_image: optional np.ndarray
    """
    model_device = next(model.parameters()).device

    # 1) Preprocess image -> resize by long side, to tensor, normalize, center-pad to square of args.imsize
    image = image.convert("RGB")
    imsize = getattr(model, 'imsize', 224)
    resized_img, ratio = _resize_by_long_side(image, imsize)
    tensor = _to_tensor(resized_img)
    tensor = _normalize(tensor)
    padded_img, pad_mask, top, left = _center_pad_to_square(tensor, imsize)

    # Build NestedTensor batch of size 1
    img_tensor_bchw = padded_img.unsqueeze(0).to(model_device)
    mask_bhw = pad_mask.unsqueeze(0).to(model_device)
    nested = NestedTensor(img_tensor_bchw, mask_bhw)

    # 2) Text batch
    text_list = [description]

    # 3) Model forward
    model.eval()
    # 确保所有子模块都设置为eval模式
    for module in model.modules():
        if hasattr(module, 'training'):
            module.training = False
        # 特别处理dropout层
        if hasattr(module, 'dropout') and hasattr(module.dropout, 'p'):
            module.dropout.p = 0.0
        if hasattr(module, 'dropout1') and hasattr(module.dropout1, 'p'):
            module.dropout1.p = 0.0
        if hasattr(module, 'dropout2') and hasattr(module.dropout2, 'p'):
            module.dropout2.p = 0.0
    
    # 设置CLIP模型为eval模式
    if hasattr(model, 'clip'):
        model.clip.eval()
        for clip_module in model.clip.modules():
            if hasattr(clip_module, 'training'):
                clip_module.training = False
    
    with torch.no_grad():
        pred_box_norm, logits_per_text, logits_per_image, visu_token_similarity, seg_mask = model(nested, text_list)

    # Shapes: pred_box_norm (B,4) in normalized xywh; seg_mask (B,1,H',W')
    pred_box_norm = pred_box_norm[0].detach().cpu()
    seg_mask_small = seg_mask[0, 0].detach().cpu()  # (H', W')

    # 4) Postprocess box back to original image coordinates
    # First map to padded canvas (imsize x imsize), then remove padding, then rescale back to original
    box_xyxy_padded = _xywh_norm_to_xyxy_pixels(pred_box_norm, imsize, imsize)
    # Remove padding
    x0, y0, x1, y1 = box_xyxy_padded.tolist()
    x0 -= left; x1 -= left
    y0 -= top;  y1 -= top
    # Clip to resized image frame
    resized_w, resized_h = resized_img.size
    x0 = max(0.0, min(x0, resized_w - 1))
    x1 = max(0.0, min(x1, resized_w - 1))
    y0 = max(0.0, min(y0, resized_h - 1))
    y1 = max(0.0, min(y1, resized_h - 1))
    # Map back to original
    inv_ratio = 1.0 / ratio
    orig_w, orig_h = image.size
    x0_orig = x0 * inv_ratio
    y0_orig = y0 * inv_ratio
    x1_orig = x1 * inv_ratio
    y1_orig = y1 * inv_ratio
    # Clip to original
    x0_orig = max(0.0, min(x0_orig, orig_w - 1))
    x1_orig = max(0.0, min(x1_orig, orig_w - 1))
    y0_orig = max(0.0, min(y0_orig, orig_h - 1))
    y1_orig = max(0.0, min(y1_orig, orig_h - 1))
    box_xyxy_orig = np.array([x0_orig, y0_orig, x1_orig, y1_orig], dtype=np.float32)

    # 5) Postprocess segmentation mask back to original image size
    # Upsample seg_mask_small -> (imsize, imsize), remove padding, resize back to original
    seg_mask_small = seg_mask_small.unsqueeze(0).unsqueeze(0)  # 1x1xH'xW'
    seg_mask_padded = torch.nn.functional.interpolate(seg_mask_small, size=(imsize, imsize), mode='bilinear', align_corners=False)[0, 0]
    seg_mask_resized = seg_mask_padded[top: top + resized_h, left: left + resized_w]
    seg_mask_resized = torch.nn.functional.interpolate(seg_mask_resized.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False)[0, 0]
    seg_mask_resized = seg_mask_resized.clamp(min=0.0)
    # Normalize to [0,1] for visualization
    if seg_mask_resized.max() > 0:
        seg_mask_resized = seg_mask_resized / seg_mask_resized.max()
    seg_mask_np = seg_mask_resized.numpy().astype(np.float32)

    result: Dict[str, Any] = {
        "box_xyxy": box_xyxy_orig,
        "seg_mask": seg_mask_np,
        "logits_per_text": logits_per_text.detach().cpu().numpy() if isinstance(logits_per_text, torch.Tensor) else None,
        "logits_per_image": logits_per_image.detach().cpu().numpy() if isinstance(logits_per_image, torch.Tensor) else None,
    }
    return result


def _draw_visualization(image: Image.Image, box_xyxy: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.35) -> Image.Image:
    vis = image.convert("RGBA")
    draw = ImageDraw.Draw(vis)
    x0, y0, x1, y1 = box_xyxy.tolist()
    # Box
    color_hex = "#00FFFF"  # cyan
    draw.rectangle([(x0, y0), (x1, y1)], outline=color_hex, width=3)
    # Mask overlay
    mask_img = Image.fromarray((seg_mask * 255.0).astype(np.uint8), mode='L').resize(image.size, Image.BILINEAR)
    overlay = Image.new("RGBA", image.size, ImageColor.getrgb("red") + (0,))
    overlay.putalpha(mask_img)
    vis = Image.blend(vis, overlay, alpha=alpha)
    return vis.convert("RGB")


def load_model(checkpoint_path: str, device: str = "cuda", model_name: str = "ViT-B/16", imsize: int = 224,
               dataset: str = "referit") -> Any:
    """
    Build HiVG model and load pretrained checkpoint.
    The CLIP pretrained paths are defined inside models/HiVG.py.
    """
    class SimpleArgs:
        # Match hivg_eval defaults for required fields
        sup_type = 'full'
        lr = 1e-4
        lr_visu_tra = 1e-5
        batch_size = 1
        weight_decay = 1e-4
        epochs = 1
        lr_power = 0.9
        clip_max_norm = 0.0
        eval = True
        optimizer = 'rmsprop'
        lr_scheduler = 'poly'
        lr_drop = 80
        aug_blur = False
        aug_crop = False
        aug_scale = False
        aug_translate = False
        model = 'CLIP-VG'  # 固定值
        model_name = 'CLIP-VG'
        extract_layer = 0
        warmup = False
        dilation = False
        position_embedding = 'sine'
        dim_feedforward = 2048
        dropout = 0.1
        num_queries = 100
        pre_norm = False
        imsize = 224  # 固定值
        emb_size = 768
        use_vl_type_embed = False
        vl_dropout = 0.1
        vl_nheads = 8
        vl_hidden_dim = 512  # 使用base模型配置以匹配checkpoint
        vl_dim_feedforward = 2048
        vl_enc_layers = 6
        vl_dec_layers = 6
        data_root = './data/image_data/'
        split_root = './data/pseudo_samples/'
        dataset = 'referit'  # 固定值
        max_query_len = 77
        output_dir = './outputs'
        device = 'cuda'  # 固定值
        seed = 13
        resume = ''
        detr_model = './saved_models/detr-r50.pth'
        bert_model = 'bert-base-uncased'
        light = False
        start_epoch = 0
        num_workers = 0
        world_size = 1
        dist_url = 'env://'
        eval_set = 'test'
        eval_model = ''
        prompt = '{pseudo_query}'
        adapt_mlp = False
        use_loss_coef = False
        normalize_before = True
        save_hilora_clip = False
        hi_lora_stage = 0
        use_seg_mask = True
        use_mask_loss = False
        mixup_pretrain = False
        enable_adaptive_weights = True

    args = SimpleArgs()
    
    # 在创建args后设置动态值
    args.model = model_name
    args.imsize = imsize
    args.dataset = dataset
    args.device = device
    
    model = build_model(args)
    model.to(torch.device(device))
    
    # 设置随机种子以确保结果一致性
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置更严格的确定性
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    
    # 设置环境变量
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            # 尝试使用weights_only=True加载
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            state = ckpt.get('model', ckpt)
            missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint with weights_only=True: {e}")
            print("Trying to load with weights_only=False (less secure but may work)...")
            try:
                # 如果weights_only=True失败，尝试weights_only=False
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                state = ckpt.get('model', ckpt)
                missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
                print(f"Loaded checkpoint from {checkpoint_path} with weights_only=False. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
            except Exception as e2:
                print(f"Warning: Failed to load checkpoint: {e2}")
                print("Running with randomly initialized fusion head.")
    else:
        print("Warning: checkpoint not provided or not found. Running with randomly initialized fusion head.")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='HiVG demo for single-image grounding/segmentation')
    parser.add_argument('--input_image_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--output_image_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', type=str, default='ViT-B/16', choices=['ViT-B/16', 'ViT-B/32', 'ViT-L/14', 'ViT-L/14-336'])
    parser.add_argument('--imsize', type=int, default=224)
    parser.add_argument('--dataset', type=str, default='referit', choices=['referit', 'flickr', 'unc', 'unc+', 'gref', 'gref_umd', 'mixup'])
    args = parser.parse_args()

    model = load_model(args.checkpoint, device=args.device, model_name=args.model, imsize=args.imsize, dataset=args.dataset)

    pil_image = Image.open(args.input_image_path).convert('RGB')
    outputs = forward(model, pil_image, args.prompt)

    vis = _draw_visualization(pil_image, outputs['box_xyxy'], outputs['seg_mask'])
    os.makedirs(os.path.dirname(args.output_image_path) or '.', exist_ok=True)
    vis.save(args.output_image_path)
    print(f"Saved visualization to {args.output_image_path}")


if __name__ == '__main__':
    main()


