# HIVG Demo使用说明

## 1. 安装环境
很遗憾服务器无法安装本来要求的python版本3.9.10，所以只能安装3.9.11
```bash
conda create --name hivg python=3.9.11
```
```bash
conda activate hivg
```
```bash
pip install -r requirements.txt
```
<!-- 
在安装 PyArrow 时遇到了 CMake 配置错误。问题出现在 CMake 无法找到 "ArrowCompute" 包。

Arrow 库文件存在于 /home/yinchao/anaconda3/lib/ 目录中，但是缺少 libarrow_compute.so 文件。这就是为什么 CMake 无法找到 "ArrowCompute" 包的原因。

在 /home/yinchao/anaconda3/lib/pkgconfig/arrow-compute.pc 存在，这说明 Arrow Compute 的 pkg-config 文件是存在的。问题可能是缺少 libarrow_compute.so 库文件。

Arrow 安装缺少 libarrow_compute.so 库文件。这是一个常见的 Arrow 安装不完整的问题。

解决方案 1：重新安装完整的 Arrow 包
```bash
conda install -c conda-forge arrow-cpp pyarrow
```
但是仍然缺少。pip命令也无效。

尝试安装一个更新的pyarrow版本
```bash
conda install -c conda-forge pyarrow=14.0.2
``` -->

## 2. Huggingface

```bash
pip install -U huggingface_hub
```
设置国内镜像站
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
登录
```bash
huggingface-cli login
```

```bash
cd /home/yinchao/HiVG/pretrained/clip
```

下载模型
```bash
hf download openai/clip-vit-large-patch14-336 
hf download openai/clip-vit-large-patch14       
hf download openai/clip-vit-base-patch32        
hf download openai/clip-vit-base-patch16        
```

下载预训练权重文件到本地电脑

https://drive.google.com/file/d/1vM_568M7DwnYmjEiJgXRnrDL5UT65CGJ/view

推送到远程服务器
```bash
cd Desktop
```

```bash
scp -r finetuning_base yinchao@i*****:/home/yinchao/HiVG/
```






## 3. 运行


