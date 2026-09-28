"""LeNet-5 手写数字分类（MNIST）——无独显 CPU 训练脚本

作业题目(1)要求自行用 PyTorch 搭建神经网络（提示中明确列出 LeNet）。
本文件从零实现经典 LeNet-5，使用上级目录中已下载好的本地 MNIST 数据
（60000 训练 + 10000 测试，共 70000 幅、10 个类别，满足"不少于1万幅、不少于10类"）。

训练结束输出报告所需的全部指标：
  Top-1 / Top-5 精度、每类精确率/召回率/F1、宏平均 F1、单图推理时延、每 epoch 耗时。
权重保存到 checkpoint/lenet/<时间戳>/，与仓库其它模型目录结构一致。
"""
import os
import sys
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 注册 Windows 中文字体，避免图中中文显示为方框
for _f in (r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc'):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        matplotlib.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=_f).get_name()]
        break
matplotlib.rcParams['axes.unicode_minus'] = False

# 本脚本在 pytorch-cifar100/ 下，MNIST 数据在上一级目录
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(HERE)


class LeNet(nn.Module):
    """LeNet-5 (LeCun, 1998)

    输入: 1x28x28 灰度图
    结构: 2 个卷积块(Conv+ReLU+MaxPool) + 3 个全连接层
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # 卷积块1: 1->6 通道, 5x5 卷积; padding=2 保持 28x28, 池化后 14x14
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        # 卷积块2: 6->16 通道, 5x5 卷积; 14x14->10x10, 池化后 5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # 全连接层: 16*5*5 -> 120 -> 84 -> 10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # [B,6,14,14]
        x = self.pool(self.relu(self.conv2(x)))   # [B,16,5,5]
        x = x.view(x.size(0), -1)                 # 展平
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=128, help='batch size')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate (Adam)')
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-num_workers', type=int, default=2)
    args = parser.parse_args()

    # ---- 运行环境说明（无独显时明确提示）----
    print('PyTorch 版本:', torch.__version__, '| CUDA 可用:', torch.cuda.is_available(),
          '| CUDA 版本:', torch.version.cuda)
    if not torch.cuda.is_available():
        print('=> 未检测到独显 / CPU 版 PyTorch，使用 CPU 训练。')
    torch.set_num_threads(os.cpu_count())
    print('CPU 线程数:', torch.get_num_threads())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 数据预处理 ----
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))   # MNIST 全局均值/标准差
    ])
    train_set = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=False, transform=transform)
    test_set = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.b, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.num_workers)
    print('训练集 %d 幅, 测试集 %d 幅, 类别数 10' % (len(train_set), len(test_set)))

    # ---- 模型 / 损失 / 优化器 ----
    net = LeNet().to(device)
    print('LeNet 参数量:', sum(p.numel() for p in net.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    ckpt_dir = os.path.join(HERE, 'checkpoint', 'lenet', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- 训练 + 每 epoch 评估 ----
    history = {'loss': [], 'acc': [], 'time': []}
    best_acc = 0.0
    log_lines = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        ep_loss, n_batch = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batch += 1
        ep_time = time.time() - t0

        net.eval()
        correct = correct5 = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                _, top5 = outputs.topk(5, 1)
                correct5 += labels.view(-1, 1).eq(top5).sum().item()
                total += labels.size(0)
        acc, acc5 = 100.0 * correct / total, 100.0 * correct5 / total

        history['loss'].append(ep_loss / n_batch)
        history['acc'].append(acc)
        history['time'].append(ep_time)
        line = 'epoch %2d 训练loss=%.4f 测试Top-1=%.2f%% Top-5=%.2f%% 耗时=%.1fs' % (
            epoch, ep_loss / n_batch, acc, acc5, ep_time)
        print(line)
        log_lines.append(line)
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), os.path.join(ckpt_dir, 'lenet-best.pth'))

    # ---- 最终完整指标（用最佳权重重新评估）----
    net.load_state_dict(torch.load(os.path.join(ckpt_dir, 'lenet-best.pth'), weights_only=True))
    net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = net(images.to(device))
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_true.extend(labels.numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Top-5（整测试集）
    top5_hits = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = net(images.to(device))
            _, top5 = outputs.topk(5, 1)
            top5_hits += labels.view(-1, 1).eq(top5.cpu()).sum().item()

    # 单图推理时延（200 次单图前向的平均值）
    sample = test_set[0][0].unsqueeze(0)
    for _ in range(10):          # 预热
        net(sample)
    t0 = time.time()
    for _ in range(200):
        net(sample)
    infer_ms = (time.time() - t0) / 200 * 1000

    report = classification_report(y_true, y_pred, digits=4, output_dict=False)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    summary = []
    summary.append('===== LeNet / MNIST 最终测试结果 =====')
    summary.append('Top-1 精度: %.2f%%' % (100.0 * (y_true == y_pred).mean()))
    summary.append('Top-5 精度: %.2f%%' % (100.0 * top5_hits / len(test_set)))
    summary.append('宏平均 F1 : %.4f' % macro_f1)
    summary.append('单图推理时延: %.2f ms' % infer_ms)
    summary.append('平均每 epoch 训练耗时: %.1fs（单批次训练 <5 分钟要求满足情况见上）' % np.mean(history['time']))
    summary.append('')
    summary.append(report)
    summary.append('混淆矩阵 (行=真实, 列=预测):')
    summary.append(str(cm))
    result_text = '\n'.join(log_lines + [''] + summary)
    print('\n'.join(summary))
    with open(os.path.join(ckpt_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(result_text)

    # ---- 训练曲线图 ----
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ep = range(1, args.epochs + 1)
    ax1.plot(ep, history['loss'], 'o-', color='#3b5bdb', label='训练损失')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('训练损失', color='#3b5bdb')
    ax1.tick_params(axis='y', labelcolor='#3b5bdb')
    ax2 = ax1.twinx()
    ax2.plot(ep, history['acc'], 'o-', color='#94d82d', label='测试Top-1精度')
    ax2.set_ylabel('Top-1 精度(%)', color='#5c940d'); ax2.tick_params(axis='y', labelcolor='#5c940d')
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right')
    plt.title('LeNet-5 在 MNIST 的 CPU 训练曲线（%d epoch）' % args.epochs)
    fig.tight_layout()
    fig.savefig(os.path.join(ckpt_dir, 'lenet_curve.png'), dpi=130)
    print('结果已保存到:', ckpt_dir)


if __name__ == '__main__':
    main()
