# 1902 Fine-Tuning

基于 Qwen2.5-VL-7B 的验证码识别微调项目，使用 LoRA 技术对视觉语言模型进行参数高效微调。

## 项目结构
fine-tuning/
├── data/              # 验证码图片数据集（png格式，文件名即标签）
├── qwen2.5-vl-7b/    # Qwen2.5-VL-7B 本地模型文件
├── output/            # 训练输出的模型权重
├── config.py          # 所有超参数和路径配置
├── dataset.py         # 数据加载和DataLoader构建
├── processor.py       # 图片和标签的预处理
├── model.py           # 模型加载和LoRA配置
├── train.py           # 训练循环和验证逻辑
└── main.py            # 项目入口

## 模型下载

```bash

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./qwen2.5-vl-7b

```