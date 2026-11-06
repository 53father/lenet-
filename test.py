import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from net import MyLeNet5
import matplotlib.pyplot as plt

# 1. 数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 2. 加载测试数据集 (路径改回 ./data)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# 3. 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"当前使用设备: {device}")

# 4. 调用net里面定义的模型，将模型数据转到GPU
model = MyLeNet5().to(device)

# 5. 加载训练好的模型权重 (路径指向 save_model 文件夹)
model.load_state_dict(torch.load('save_model/best_model.pth', map_location=device))
print("成功加载模型权重！")

# 6. 获取结果
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 把tensor转化为图片，方便可视化
show = ToPILImage()

# 进入验证阶段
model.eval()  # 切换到评估模式
correct = 0
total = 0

# 准备画板
plt.figure(figsize=(12, 10))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

with torch.no_grad():
    for batch_idx, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)

        # 前向传播
        output = model(X)
        # 取概率最大的类别作为预测
        predict = output.argmax(dim=1)

        # 统计准确率
        total += y.size(0)
        correct += (predict == y).sum().item()

        # 只取第一个 batch 可视化
        if batch_idx == 0:
            for i in range(len(X)):
                plt.subplot(4, 4, i + 1)
                plt.axis('off')
                img = show(X[i].cpu())
                plt.imshow(img, cmap='gray')

                pred_label = classes[predict[i].item()]
                true_label = classes[y[i].item()]
                title_color = 'green' if pred_label == true_label else 'red'
                plt.title(f"预测: {pred_label}\n真实: {true_label}", color=title_color)

# 显示图片
plt.tight_layout()
plt.show()

# 打印整体准确率
accuracy = correct / total * 100
print(f"\n整个测试集的准确率: {accuracy:.2f}%")