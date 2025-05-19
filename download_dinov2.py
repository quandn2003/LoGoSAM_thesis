import os
import torch

# Thiết lập thư mục cache
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'pretrained_model')
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)

# Lần đầu chạy sẽ clone repo và download weights vào pretrained_model
torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
print("Đã tải xong dinov2_vitb14 vào:", os.environ['TORCH_HOME'])
