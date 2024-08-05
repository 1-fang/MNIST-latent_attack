import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Variance and mean
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def Reparamiterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=1)
        z = self.Reparamiterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var

# 加载预训练的 VAE 模型
input_dim = 784  # 28x28
hidden_dim = 400
latent_dim = 20
vae = VAE(input_dim, hidden_dim, latent_dim)

# 加载预训练的权重
vae.load_state_dict(torch.load('vae.pth'))
vae.eval()

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # 展平图像为 784 维向量
])

# 加载 MNIST 数据集
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

# 编码 MNIST 数据集并保存潜在表示和标签
latent_codes = []
labels = []

with torch.no_grad():
    for data, target in dataloader:
        data = data.to(device)
        _, mu, _ = vae(data)  # 只需要编码器的均值
        latent_codes.append(mu.cpu().numpy())
        labels.append(target.cpu().numpy())

latent_codes = np.concatenate(latent_codes, axis=0)
labels = np.concatenate(labels, axis=0)

# 创建保存特征的文件夹
os.makedirs('features', exist_ok=True)

# 保存潜在表示和标签为 .npy 文件
np.save('features/latent_codes.npy', latent_codes)
np.save('features/labels.npy', labels)

print(f'Latent codes shape: {latent_codes.shape}')
print(f'Labels shape: {labels.shape}')