import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, z_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, z_dim)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

# 定义编码器
input_dim = 784  # 假设输入是28x28的图像
hidden_dim = 400
z_dim = 20
encoder = Encoder(input_dim, hidden_dim, z_dim)

# 生成一个随机输入
x = torch.randn((1, input_dim), requires_grad=True)

# 前向传播得到 mu 和 logvar
mu, logvar = encoder(x)

# 使用重参数化技巧生成 z
z = encoder.reparameterize(mu, logvar)

# 使用 torch.autograd.grad 计算 z 对 x 的梯度
grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=torch.ones_like(z), create_graph=True)[0]

# 打印梯度
print(grad_x)