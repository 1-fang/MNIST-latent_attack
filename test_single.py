'''import torch
from torch import nn
import numpy as np
from vae import VAE


input_dim = 784
hidden_dim = 400
latent_dim = 20

vae_model = VAE(input_dim, hidden_dim, latent_dim)
vae_model.load_state_dict(torch.load("./models/vae.pth"))
vae_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_model.to(device)

def generate_single_latent(mode, latent_dim, device, filename):
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        np.save(filename, z.cpu().numpy())

generate_single_latent(vae_model, latent_dim, device, "single_latent.npy")'''



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from vae import VAE

# 定义 VAE 模型参数
input_dim = 784
hidden_dim = 400
latent_dim = 20

# 加载预训练的 VAE 模型
vae_model = VAE(input_dim, hidden_dim, latent_dim)
vae_model.load_state_dict(torch.load("./models/vae.pth"))
vae_model.eval()

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_model.to(device)

# 加载潜在向量
latent_vector = np.load('single_latent.npy')

# 将潜在向量转换为 PyTorch 张量
z = torch.tensor(latent_vector, dtype=torch.float32).to(device)

# 解码潜在向量并保存图像
with torch.no_grad():
    generated_image = vae_model.decoder(z).view(28, 28)

# 将图像转换为 NumPy 数组并保存为 PNG 文件
plt.imshow(generated_image.cpu().numpy(), cmap='gray')
plt.axis('off')
plt.savefig('fig1.png', bbox_inches='tight', pad_inches=0)
plt.close()