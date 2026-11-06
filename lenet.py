# 导入PyTorch库
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# 定义LeNet-5架构的神经网络类
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一卷积层：输入1通道（灰度图像），输出6通道，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第一池化层：最大池化，池化窗口大小为2x2，步幅为2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二卷积层：输入6通道，输出16通道，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第二池化层：最大池化，池化窗口大小为2x2，步幅为2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第一个全连接层：输入维度是16*4*4，输出维度是120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 第二个全连接层：输入维度是120，输出维度是84
        self.fc2 = nn.Linear(120, 84)
        # 第三个全连接层：输入维度是84，输出维度是10，对应10个类别
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 前向传播函数定义网络的数据流向
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义数据变换和加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 训练数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 测试数据集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化LeNet-5模型以及定义损失函数和优化器
net = LeNet5()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，用于分类问题
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Adam优化器，学习率为0.001

# 训练循环
train_loss_history = []   # 新增：记录每个 epoch 的训练损失
train_acc_history = []    # 新增：记录每个 epoch 的训练准确率
test_acc_history = []     # 新增：记录每个 epoch 的测试准确率

for epoch in range(5):  # 可以根据需要调整训练的轮数
    running_loss = 0.0
    correct_train = 0     # 新增
    total_train = 0       # 新增
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()  # 清零梯度

        outputs = net(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新权重

        running_loss += loss.item()

        # 新增：累计训练集上的预测正确数，用于统计训练准确率
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_train_acc = 100 * correct_train / total_train

    # 新增：每个 epoch 结束后在测试集上评估一次，用于画准确率曲线
    correct_eval = 0
    total_eval = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_eval += labels.size(0)
            correct_eval += (predicted == labels).sum().item()
    epoch_test_acc = 100 * correct_eval / total_eval

    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_train_acc)
    test_acc_history.append(epoch_test_acc)

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, "
          f"Train Acc: {epoch_train_acc:.2f}%, Test Acc: {epoch_test_acc:.2f}%")

print("Finished Training")

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = net(inputs)  # 前向传播
        _, predicted = torch.max(outputs.data, 1)  # 找到最大概率的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on the test set: {accuracy}%")

# 可视化测试集中的预测结果
images, labels = next(iter(test_loader))
with torch.no_grad():
    outputs = net(images)
    predictions = outputs.argmax(dim=1)

# 显示前 10 张图片
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    # Normalize((0.5,), (0.5,)) 的反变换：恢复到 [0, 1] 便于显示
    image = images[i].squeeze().numpy() * 0.5 + 0.5

    ax.imshow(image, cmap="gray")
    ax.set_title(f"真实值: {labels[i].item()}\n预测值: {predictions[i].item()}")
    ax.axis("off")

plt.tight_layout()
plt.show()


# ==================== 新增可视化 1：测试集最后 10 个样本 ====================
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
with torch.no_grad():
    for i, ax in enumerate(axes.flat):
        idx = len(test_dataset) - 10 + i          # 取测试集最后 10 个样本
        image, label = test_dataset[idx]
        pred = net(image.unsqueeze(0)).argmax(dim=1).item()

        img = image.squeeze().numpy() * 0.5 + 0.5  # 反变换到 [0, 1] 便于显示
        ax.imshow(img, cmap="gray")
        ax.set_title(f"真实值: {label}\n预测值: {pred}")
        ax.axis("off")

plt.suptitle("测试集最后 10 个样本的预测结果")
plt.tight_layout()
plt.show()


# ==================== 新增可视化 2：全程准确率变化折线图 ====================
epochs = range(1, len(train_acc_history) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc_history, marker='o', label='训练准确率')
plt.plot(epochs, test_acc_history, marker='s', label='测试准确率')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy over Epochs")
plt.xticks(list(epochs))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()