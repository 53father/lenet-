# -*- coding: utf-8 -*-
"""
LeNet-5 + MNIST：训练与可视化完整示例

相比原始脚本，这里补充了 6 组可视化：
  01_training_curves.png   训练损失、训练准确率、测试准确率曲线
  02_dataset_samples.png   训练集样本预览（随机 32 张）
  03_predictions.png       测试集预测结果网格（绿色=预测正确，红色=预测错误）
  04_confusion_matrix.png  10×10 混淆矩阵
  05_misclassified.png     被分错的样本（按置信度从高到低，即“错得最自信”的）
  06_conv1_features.png    第一层 6 个 5×5 卷积核 + 对应特征图

用法：
  python lenet5_mnist_visual.py                # 训练 5 轮，保存图片并弹窗显示
  python lenet5_mnist_visual.py --epochs 10    # 训练 10 轮
  python lenet5_mnist_visual.py --no-show      # 只保存图片，不弹窗（服务器/无界面环境）
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import font_manager
from torch.utils.data import DataLoader

plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


class LeNet5(nn.Module):
    """LeNet-5 网络结构。"""

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 第一卷积层：输入 1 通道（灰度图），输出 6 通道，卷积核 5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第一池化层：最大池化，窗口 2x2，步幅 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二卷积层：输入 6 通道，输出 16 通道，卷积核 5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第二池化层：最大池化，窗口 2x2，步幅 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 三个全连接层：16*4*4 -> 120 -> 84 -> 10
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # 前向传播：卷积 -> 激活 -> 池化，重复两次，再展平接全连接层
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def configure_chinese_font():
    """让 matplotlib 能正常显示中文标题和坐标轴文字。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                 "WenQuanYi Micro Hei", "PingFang SC", "Arial Unicode MS"):
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            print("[matplotlib] 使用中文字体: %s" % name)
            return name
    print("[matplotlib] 未找到中文字体，图中的中文可能显示为方块。")
    return None


def softmax(x):
    """numpy 版 softmax，用于把置信度转成概率。"""
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def to_display(image):
    """Normalize((0.5,), (0.5,)) 的逆变换：把张量还原到 [0, 1]，便于用 imshow 显示。"""
    return (image.squeeze().cpu().numpy() * 0.5 + 0.5).clip(0, 1)


def save_figure(fig, out_dir, filename):
    """保存图片到 out_dir，并返回完整路径。"""
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("  [保存] %s" % path)
    return path


@torch.no_grad()
def evaluate(model, loader, device, num_classes=10):
    """在给定数据集上评估，返回准确率以及全部图片、标签、预测和概率。"""
    model.eval()
    correct = 0
    total = 0
    all_images, all_labels, all_preds, all_probs = [], [], [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_images.append(inputs.cpu())
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    accuracy = 100.0 * correct / total
    return (accuracy, torch.cat(all_images), torch.cat(all_labels),
            torch.cat(all_preds), torch.cat(all_probs))


def confusion_matrix_np(y_true, y_pred, num_classes=10):
    """用 bincount 快速计算混淆矩阵，避免依赖 sklearn。"""
    idx = y_true * num_classes + y_pred
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def plot_training_curves(history, out_dir):
    """图 1：损失曲线 + 准确率曲线。"""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "o-", color="#d62728", linewidth=1.8)
    ax.set_title("训练损失（CrossEntropyLoss）")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], "o-", color="#1f77b4",
            linewidth=1.8, label="训练集准确率")
    ax.plot(epochs, history["test_acc"], "s--", color="#2ca02c",
            linewidth=1.8, label="测试集准确率")
    ax.set_title("准确率随训练轮数的变化")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(epochs)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_figure(fig, out_dir, "01_training_curves.png")


def plot_dataset_samples(dataset, out_dir, n=32, seed=0):
    """图 2：随机抽 32 张训练图片，看看数据长什么样。"""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=n, replace=False)

    cols = 8
    rows = int(np.ceil(n / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.55))
    for ax, index in zip(axes.flat, indices):
        image, label = dataset[int(index)]
        ax.imshow(to_display(image), cmap="gray", interpolation="nearest")
        ax.set_title("标签 %d" % label, fontsize=9)
        ax.axis("off")
    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle("训练集样本预览", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.8, h_pad=1.2)
    save_figure(fig, out_dir, "02_dataset_samples.png")


def plot_predictions(images, labels, preds, probs, out_dir, n=24, seed=1):
    """图 3：预测结果网格。刻意混入若干错误样本，方便一眼看出对错。"""
    rng = np.random.RandomState(seed)
    labels_np = labels.numpy()
    preds_np = preds.numpy()

    wrong_idx = np.where(labels_np != preds_np)[0]
    right_idx = np.where(labels_np == preds_np)[0]

    n_wrong = min(len(wrong_idx), 6)
    picked = list(rng.choice(wrong_idx, size=n_wrong, replace=False)) if n_wrong else []
    picked += list(rng.choice(right_idx, size=n - n_wrong, replace=False))
    rng.shuffle(picked)

    cols = 6
    rows = int(np.ceil(len(picked) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.1))
    for ax, index in zip(axes.flat, picked):
        true_label = int(labels_np[index])
        pred_label = int(preds_np[index])
        confidence = float(probs[index, pred_label])
        is_correct = true_label == pred_label

        ax.imshow(to_display(images[index]), cmap="gray", interpolation="nearest")
        ax.set_title("真值 %d / 预测 %d\n置信度 %.1f%%" % (true_label, pred_label, confidence * 100),
                     fontsize=9,
                     color="#2ca02c" if is_correct else "#d62728")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#2ca02c" if is_correct else "#d62728")
            spine.set_linewidth(2)
    for ax in axes.flat[len(picked):]:
        ax.axis("off")

    fig.suptitle("测试集预测结果（绿色=正确，红色=错误）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.8)
    save_figure(fig, out_dir, "03_predictions.png")


def plot_confusion_matrix(cm, out_dir):
    """图 4：10×10 混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(cm, cmap="Blues")

    threshold = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, "%d" % cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > threshold else "black")

    ax.set_title("混淆矩阵（测试集）")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, out_dir, "04_confusion_matrix.png")


def plot_misclassified(images, labels, preds, probs, out_dir, n=12, seed=2):
    """图 5：被分错的样本，按“错得有多自信”排序展示。"""
    labels_np = labels.numpy()
    preds_np = preds.numpy()
    wrong = np.where(labels_np != preds_np)[0]

    if len(wrong) == 0:
        print("  [跳过] 测试集没有分错的样本，无需绘制错误样例图。")
        return None

    # 取预测概率最高的那些错误样本：模型“最自信却答错”的案例最有分析价值
    confidences = probs.numpy()[wrong, preds_np[wrong]]
    order = wrong[np.argsort(-confidences)]
    picked = order[:n]

    cols = 4
    rows = int(np.ceil(len(picked) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.6))
    for ax, index in zip(axes.flat, picked):
        true_label = int(labels_np[index])
        pred_label = int(preds_np[index])
        confidence = float(probs[index, pred_label])

        ax.imshow(to_display(images[index]), cmap="gray", interpolation="nearest")
        ax.set_title("真值 %d → 预测 %d\n置信度 %.1f%%" % (true_label, pred_label, confidence * 100),
                     fontsize=9, color="#d62728")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#d62728")
            spine.set_linewidth(2)
    for ax in axes.flat[len(picked):]:
        ax.axis("off")

    fig.suptitle("被分错的样本（按置信度从高到低）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.8)
    save_figure(fig, out_dir, "05_misclassified.png")
    return picked


def plot_conv1_features(net, image, label, out_dir):
    """图 6：第一层卷积的 6 个 5×5 卷积核，以及同一张图经过它们之后的特征图。"""
    device = next(net.parameters()).device
    with torch.no_grad():
        activations = torch.relu(net.conv1(image.unsqueeze(0).to(device)))[0].cpu()

    kernels = net.conv1.weight.detach().cpu()

    fig, axes = plt.subplots(2, 6, figsize=(13, 4.9))
    for i in range(6):
        ax = axes[0, i]
        ax.imshow(kernels[i, 0].numpy(), cmap="RdBu_r", interpolation="nearest")
        ax.set_title("卷积核 %d（5×5）" % (i + 1), fontsize=9)
        ax.axis("off")

        ax = axes[1, i]
        ax.imshow(activations[i].numpy(), cmap="viridis", interpolation="nearest")
        ax.set_title("特征图 %d（24×24）" % (i + 1), fontsize=9)
        ax.axis("off")

    fig.suptitle("第一层卷积：6 个卷积核，以及输入数字 %d 时各自的激活（ReLU 后）" % label,
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92], h_pad=2.2)
    save_figure(fig, out_dir, "06_conv1_features.png")


def main():
    parser = argparse.ArgumentParser(description="LeNet-5 + MNIST 训练与可视化")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数（默认 5）")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小（默认 64）")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率（默认 0.001）")
    parser.add_argument("--data-dir", default="./data", help="MNIST 数据目录")
    parser.add_argument("--out-dir", default="./figures", help="图片输出目录")
    parser.add_argument("--show", dest="show", action="store_true", default=True,
                        help="训练结束后弹窗显示图片（默认开启）")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="只保存图片，不弹窗")
    args = parser.parse_args()

    configure_chinese_font()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备: %s" % device)

    # 数据变换：转张量 + 归一化到 [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    test_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    print("训练集: %d 张, 测试集: %d 张" % (len(train_dataset), len(test_dataset)))

    net = LeNet5().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print("模型参数量: %d" % n_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    # ---------------- 训练 + 每轮评估 ----------------
    for epoch in range(1, args.epochs + 1):
        net.train()
        start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += inputs.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        test_acc, _, _, _, _ = evaluate(net, test_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print("Epoch %d/%d | loss: %.4f | 训练准确率: %.2f%% | 测试准确率: %.2f%% | %.1fs"
              % (epoch, args.epochs, train_loss, train_acc, test_acc, time.time() - start))

    print("训练结束，最终测试集准确率: %.2f%%" % history["test_acc"][-1])

    # ---------------- 用训练好的模型收集测试集全部预测 ----------------
    test_acc, images, labels, preds, probs = evaluate(net, test_loader, device)
    cm = confusion_matrix_np(labels.numpy(), preds.numpy())

    print("逐类准确率:")
    for c in range(10):
        total_c = cm[c].sum()
        acc_c = 100.0 * cm[c, c] / total_c if total_c else 0.0
        print("  数字 %d: %6.2f%% (%d/%d)" % (c, acc_c, cm[c, c], total_c))
    print("平均准确率: %.2f%%" % test_acc)

    # ---------------- 生成可视化 ----------------
    print("生成可视化图片 ...")
    plot_training_curves(history, args.out_dir)
    plot_dataset_samples(train_dataset, args.out_dir)
    plot_predictions(images, labels, preds, probs, args.out_dir)
    plot_confusion_matrix(cm, args.out_dir)
    plot_misclassified(images, labels, preds, probs, args.out_dir)
    plot_conv1_features(net, images[0], int(labels[0]), args.out_dir)

    if args.show:
        print("关闭图片窗口后程序结束。")
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
