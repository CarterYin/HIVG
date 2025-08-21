# demo.py使用说明
## 1. 创建环境与安装依赖
```bash
conda env create --file environment.yml
```

## 2.相关文件下载
- 下载best_checkpoint.pth到mixip_pretraining_large文件夹下
- 下载clip的四个模型到clip文件夹下(git clone或者使用原有脚本)
- 下载clip_l_ml_cascade_maskrcnn_model_224.pth到clip文件夹下



## 3.使用命令示例
```bash
conda activate hivg && python demo.py --input_image_path /home/yinchao/HiVG/images/COCO_train2014_000000435940.jpg --prompt "boy wearing purple t-shirt" --output_image_path /home/yinchao/HiVG/images/boy_result.jpg --device cpu
```
```bash
conda activate hivg && python demo.py --input_image_path /home/yinchao/HiVG/images/COCO_train2014_000000144725.jpg --prompt "bird" --output_image_path /home/yinchao/HiVG/images/bird_result.jpg --device cpu
```
```bash
conda activate hivg && python demo.py --input_image_path /home/yinchao/HiVG/images/COCO_train2014_000000291797.jpg --prompt "man" --output_image_path /home/yinchao/HiVG/images/man_result.jpg --device cpu
```
```bash
conda activate hivg && python demo.py --input_image_path /home/yinchao/HiVG/images/COCO_train2014_000000581921.jpg --prompt "snowboard" --output_image_path /home/yinchao/HiVG/images/snowboard_result.jpg --device cpu
```
