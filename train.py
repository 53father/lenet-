import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from net import MyLeNet5

# ----------------------- 超参数集中放这里，方便改 -----------------------
BATCH_SIZE = 64
EPOCH = 50
LR = 1e-3
MOMENTUM = 0.9
SAVE_DIR = "save_model"
BEST_PATH = os.path.join(SAVE_DIR, "best_model.pth")

# 有显卡就用 GPU，没有就用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备：", device)

# ----------------------- 1. 数据 -----------------------
# PIL 图片 [0,255] -> Tensor [0,1]，形状 [1, 28, 28]
data_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 训练集和测试集分开；root 相对当前工作目录
train_dataset = datasets.MNIST(root="./data", train=True,
                               transform=data_transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False,
                              transform=data_transform, download=True)

# 注意关键字参数是 dataset（单数），不是 datasets
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# 测试集不需要打乱
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------- 2. 模型 / 损失 / 优化器 -----------------------
model = MyLeNet5().to(device)

# 交叉熵损失内部自带 log_softmax，所以网络最后一层不要加 softmax
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

# 每 10 个 epoch 学习率乘 0.1（注意：名字别和模块 lr_scheduler 重名）
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# ----------------------- 3. 训练一个 epoch -----------------------
def train(dataloader, model, loss_fn, optimizer):
    model.train()  # 切换到训练模式
    total_loss, total_acc, n = 0.0, 0.0, 0

    for X, y in dataloader:
        # 前向传播
        X, y = X.to(device), y.to(device)
        output = model(X)                  # [N, 10] 的 logits
        cur_loss = loss_fn(output, y)

        # 取概率最大的类别作为预测（等价于 torch.max(output, dim=1)[1]）
        pred = output.argmax(dim=1)
        cur_acc = (pred == y).sum().item() / output.shape[0]

        # 反向传播 + 更新参数
        optimizer.zero_grad()   # 清空上一步残留的梯度
        cur_loss.backward()     # 计算梯度
        optimizer.step()        # 更新权重

        total_loss += cur_loss.item()
        total_acc += cur_acc
        n += 1

    print(f"train_loss: {total_loss / n:.4f}  train_acc: {total_acc / n:.4f}")
    return total_acc / n


# ----------------------- 4. 验证 -----------------------
@torch.no_grad()  # 验证阶段不需要计算梯度，省显存/内存
def val(dataloader, model, loss_fn):
    model.eval()  # 切换到评估模式
    total_loss, total_acc, n = 0.0, 0.0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        total_loss += loss_fn(output, y).item()
        total_acc += (output.argmax(dim=1) == y).sum().item() / output.shape[0]
        n += 1

    print(f"val_loss:   {total_loss / n:.4f}  val_acc:   {total_acc / n:.4f}")
    return total_acc / n


# ----------------------- 5. 开始训练 -----------------------
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)  # 目录不存在就建，已存在就跳过

    best_acc = 0.0
    for t in range(EPOCH):
        print(f"epoch {t + 1}/{EPOCH}\n" + "-" * 30)
        train(train_dataloader, model, loss_fn, optimizer)
        acc = val(test_dataloader, model, loss_fn)
        scheduler.step()  # 每个 epoch 结束后才调整学习率

        # 保存准确率最高的一次权重
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_PATH)
            print(f"save best model (acc={acc:.4f}) -> {BEST_PATH}")

    print(f"Done! 最佳准确率：{best_acc:.4f}")
