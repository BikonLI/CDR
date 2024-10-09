import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2

# 常量设置
INPUT_CHANNELS = 1  # 输入通道数（灰度图像）
INPUT_HEIGHT = 50   # 输入图像高度
INPUT_WIDTH = 125   # 输入图像宽度
NUM_CLASSES = 10    # 每个数字的类别数（0-9）
OUTPUT_NUMBERS = 2  # 输出的数字个数

class DigitRecognitionModel(nn.Module):
    def __init__(self):
        super(DigitRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(23808, 128)  # 根据卷积后特征图的尺寸调整
        self.fc2 = nn.Linear(128, OUTPUT_NUMBERS * NUM_CLASSES)  # 输出20个类（0-9的两个数字）

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, OUTPUT_NUMBERS, NUM_CLASSES)  # 调整输出形状为 [batch_size, 2, 10]

# 创建模型实例
model = DigitRecognitionModel()


import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

def train_model(model: torch.nn.Module, dataset, num_epochs=70, batch_size=64, learning_rate=0.001):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # 确保保存模型的目录存在
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        total_loss = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 清空梯度

            outputs = model(images)  # 输出形状应为 [batch_size, 20]
            first_digit_logits = outputs[:, 0, :]  # 取第一个数字的概率
            second_digit_logits = outputs[:, 1, :]  # 取后10个类用于第二个数字
            
            # 假设标签的形状是 [batch_size, 2]
            first_digit_labels = labels[:, 0]  # 第一个数字的标签
            second_digit_labels = labels[:, 1]  # 第二个数字的标签

            # 计算损失
            loss_1 = criterion(first_digit_logits, first_digit_labels)
            loss_2 = criterion(second_digit_logits, second_digit_labels)

            # 计算均值损失
            total_loss = (loss_1 + loss_2) / 2

            # 反向传播
            total_loss.backward()
            
            optimizer.step()  # 更新权重
            total_loss += total_loss.item()  # 累加损失

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # 每个 epoch 完成后保存模型
        model_path = 'OCR.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model saved at {model_path}')
    
def recognize(img):
    img = torch.tensor(img, dtype=torch.float32)
    img = img / 255
    img = img.unsqueeze(0)
    dic = torch.load("OCR.pth")
    model.load_state_dict(dic)
    result = model(img)
    print(result)
    return result

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 125))
    return img
    
# 调用示例
# train_model(model, dataset)
