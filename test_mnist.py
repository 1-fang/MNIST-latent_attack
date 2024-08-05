import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 加载潜在编码和标签
test_latent_codes = np.load('features/test_latent_codes.npy')
test_labels = np.load('features/test_labels.npy')

# 将数据转换为 PyTorch 张量
X_test = torch.tensor(test_latent_codes, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)

# 创建 DataLoader
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义分类器
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

input_dim = test_latent_codes.shape[1]  # 潜在编码的维度
num_classes = 10  # MNIST 数据集有 10 个类别
classifier = Classifier(input_dim, num_classes)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

# 加载最佳模型
classifier.load_state_dict(torch.load('./models/best.pth'))
classifier.eval()

# 测试分类器
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = classifier(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

test_accuracy = 100 * correct / total
print(f'Final Test Accuracy: {test_accuracy}%')