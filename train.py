import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from vae import VAE

def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train_vae(model, dataloader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in tqdm(range(epochs)):
        train_loss = 0  # Initialize train_loss inside the epoch loop
        for data, _ in dataloader:
            data = data.view(data.size(0), -1).to(device)  # Move data to device
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# 加载数据
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 初始化和训练模型
input_dim = 784  # 28x28
hidden_dim = 400
latent_dim = 20
epochs = 10
lr = 1e-3

model = VAE(input_dim, hidden_dim, latent_dim).to(device)  # Move model to device
train_vae(model, dataloader, epochs, lr, device)

# 保存模型
torch.save(model.state_dict(), 'vae.pth')