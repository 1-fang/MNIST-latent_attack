import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from vae import VAE

def save_origin_image_and_latent_code(vae_model, device, save_img_path='figs/origin_img.png', save_latent_path='latent_vector.npy'):
    # 加载 MNIST 测试数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # 展平图像为 784 维向量
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 取出一张原始图像
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    origin_img = images[0].view(28, 28).cpu().numpy()

    # 保存原始图像
    plt.imshow(origin_img, cmap='gray')
    plt.axis('off')
    plt.savefig(save_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 将原始图像转换为 PyTorch 张量
    x = images[0].view(1, -1).to(device)

    # 编码原始图像
    with torch.no_grad():
        encoded = vae_model.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=1)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        latent_vector = z.cpu().numpy()
        np.save(save_latent_path, latent_vector)

    return x, labels, z

# 示例用法
if __name__ == "__main__":
    input_dim = 784
    hidden_dim = 400
    latent_dim = 20

    vae_model = VAE(input_dim, hidden_dim, latent_dim)
    vae_model.load_state_dict(torch.load("./models/vae.pth"))
    vae_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model.to(device)

    x, labels, z = save_origin_image_and_latent_code(vae_model, device)