# 1902 Fine-Tuning

基于 Qwen2.5-VL-7B 的验证码识别微调项目，使用 LoRA 技术对视觉语言模型进行参数高效微调。

## 项目结构
```
fine-tuning/
├── data/              # 由以标签为文件名的png图片作为数据的验证码数据集
├── qwen2.5-vl-7b/    # 本地模型文件
├── output/            # 训练输出的模型权重
├── config.py          # 超参数和路径配置
├── dataset.py         # 数据加载
├── processor.py       # 数据预处理
├── model.py           # 模型加载和LoRA配置
├── train.py           # 训练循环
└── main.py            # 项目入口
```

## 技术细节

- 基础模型：Qwen2.5-VL-7B-Instruct
- 微调方法：LoRA（Low-Rank Adaptation）
- 训练数据：10万张验证码图片，数字+字母组合
- 训练设备：NVIDIA RTX 4060（8GB显存）
- LoRA 参数：rank=16，alpha=32

## 模型下载

```bash

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./qwen2.5-vl-7b

```