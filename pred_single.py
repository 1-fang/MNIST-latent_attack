import torch
import torch.nn as nn
import numpy as np

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

# 定义分类器模型参数
input_dim = 20  # 潜在编码的维度
num_classes = 10  # MNIST 数据集有 10 个类别

# 加载分类器模型
classifier = Classifier(input_dim, num_classes)
classifier.load_state_dict(torch.load('./models/best.pth'))
classifier.eval()

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

# 加载潜在向量
latent_vector = np.load('latent_vector.npy')

# 将潜在向量转换为 PyTorch 张量
z = torch.tensor(latent_vector, dtype=torch.float32).to(device)

# 对潜在向量进行预测
with torch.no_grad():
    output = classifier(z)
    _, predicted = torch.max(output.data, 1)

# 打印预测结果
print(f'Predicted digit: {predicted.item()}')