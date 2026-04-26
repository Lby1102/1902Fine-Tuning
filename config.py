import os

#dir
root_dir=r'C:\Users\lby13\Desktop\fine-tuning'
data_dir=os.path.join(root_dir,'data')
model_dir=os.path.join(root_dir,'qwen2.5-vl-7b')
output_dir=os.path.join(root_dir,'output')

#size of image
image_hight=40
image_width=95

#training hyperparameters
batch_size=8
learning_rate=2e-4
num_epochs=3
train_ratio=0.9

#LoRA
lora_rank=16
lora_alpha=32
lora_dropout=0.05

#device
device='cuda'

