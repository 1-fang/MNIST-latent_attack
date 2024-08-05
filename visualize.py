import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import os

# 创建保存图像的文件夹
os.makedirs('figs', exist_ok=True)

# 加载潜在编码和标签
latent_codes = np.load('features/latent_codes.npy')
labels = np.load('features/labels.npy')

print(f'Latent codes shape: {latent_codes.shape}')
print(f'Labels shape: {labels.shape}')

# 使用 PCA 将潜在编码降维到 2 维
pca = PCA(n_components=2)
latent_codes_2d = pca.fit_transform(latent_codes)

# 可视化潜在编码
plt.figure(figsize=(10, 7))
plt.scatter(latent_codes_2d[:, 0], latent_codes_2d[:, 1], c=labels, cmap='viridis', s=5)
plt.colorbar()
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Visualization of Latent Codes')
plt.savefig('figs/latent_codes_2d.png')  # 保存图像
plt.close()

# 创建标签的 DataFrame
labels_df = pd.DataFrame(labels, columns=['label'])

# 可视化标签分布
plt.figure(figsize=(10, 7))
sns.countplot(x='label', data=labels_df)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.savefig('figs/label_distribution.png')  # 保存图像
plt.close()