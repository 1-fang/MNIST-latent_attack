import torch
import torch.nn as nn
from vae import VAE
import matplotlib.pyplot as plt

input_dim = 784
hidden_dim = 400
latent_dim = 20

vae_model = VAE(input_dim, hidden_dim, latent_dim)
vae_model.load_state_dict(torch.load("vae.pth"))
vae_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#generate new images
def generate_random_images(model, latent_dim, device, filename):
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        generated_images = model.decoder(z).view(28,28)
        plt.imshow(generated_images.cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
    
vae_model.to(device)

generate_random_images(vae_model, latent_dim, device, 'generated_image.png')


