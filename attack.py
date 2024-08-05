import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from vae import VAE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# 定义分类器模型
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 定义参数
input_dim = 784
hidden_dim = 400
latent_dim = 20
num_classes = 10

# 加载模型
vae_model = VAE(input_dim, hidden_dim, latent_dim)
vae_model.load_state_dict(torch.load("./models/vae.pth"))
vae_model.eval()

classifier = Classifier(latent_dim, num_classes)
classifier.load_state_dict(torch.load('./models/best.pth'))
classifier.eval()

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_model.to(device)
classifier.to(device)

# 加载 MNIST 测试数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 定义攻击参数
epsilon = 8 / 255

# 初始化统计变量
total_samples = 0
successful_attacks = 0

# 对每个测试样本进行 FGSM 攻击
for data, target in test_loader:
    origin_img = data.to(device)
    origin_img.requires_grad = True

    # 将原始图像通过 VAE 编码器获取潜在编码 z
    encoded = vae_model.encoder(origin_img)
    mu, log_var = torch.chunk(encoded, 2, dim=1)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + eps * std
    z = z.detach().requires_grad_(True)  # 确保 z 是叶子张量并启用梯度计算

    # 对潜在向量进行预测
    output = classifier(z)
    _, predicted = torch.max(output.data, 1)

    # 计算损失函数（交叉熵损失）
    criterion = nn.CrossEntropyLoss()
    labels = target.to(device)
    loss = criterion(output, labels)

    # 计算 \partial y / \partial z
    classifier.zero_grad()
    loss.backward(retain_graph=True)
    grad_y_z = z.grad.clone()
    #print(f'grad_y_z shape: {grad_y_z.shape}')
    print(f'grad_y_z: {grad_y_z}')


    # 计算 \partial z / \partial x
    vae_model.zero_grad()
    origin_img.requires_grad_(True)  # 确保 origin_img 是一个叶子张量
    encoded = vae_model.encoder(origin_img)
    mu, log_var = torch.chunk(encoded, 2, dim=1)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + eps * std
    z.backward(torch.ones_like(z), retain_graph=True)
    grad_z_x = origin_img.grad.clone()
    #print(f'grad_z_x shape: {grad_z_x.shape}')
    print(f'grad_z_x: {grad_z_x}')


    # 计算 \partial y / \partial x = (\partial y / \partial z) * (\partial z / \partial x)
    grad_y_x = torch.matmul(grad_y_z.t(), grad_z_x)
    grad_y_x = grad_y_x.sum(dim=0, keepdim=True)
    #print(f'grad_y_x shape: {grad_y_x.shape}')
    print(f'grad_y_x: {grad_y_x}')


    # FGSM 攻击
    x_adv = origin_img + epsilon * grad_y_x.sign()
    x_adv = x_adv.detach()

    # 对对抗样本进行预测
    encoded_adv = vae_model.encoder(x_adv)
    mu_adv, log_var_adv = torch.chunk(encoded_adv, 2, dim=1)
    std_adv = torch.exp(0.5 * log_var_adv)
    z_adv = mu_adv + std_adv * torch.randn_like(std_adv)
    output_adv = classifier(z_adv)
    _, predicted_adv = torch.max(output_adv.data, 1)

    # 统计攻击成功率
    total_samples += 1
    if predicted.item() != predicted_adv.item():
        successful_attacks += 1

# 计算攻击成功率
attack_success_rate = successful_attacks / total_samples * 100
print(f'Attack Success Rate: {attack_success_rate:.2f}%')