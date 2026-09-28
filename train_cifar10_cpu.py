"""CIFAR-10 CPU 训练脚本（无独显电脑用）

目标：满足课程"分类不少于10类、准确率不低于90%"指标。
数据集：CIFAR-10（6万训练 + 1万测试，10类），数据包由官方源下载到 ./data/。
网络：复用仓库 models/ 下的 CIFAR 版模型，构建时改为 10 类输出：
  - resnet18   主力模型（残差结构，CPU 约 3~4 分钟/epoch）
  - seresnet18 对比模型（在 ResNet 基础上加入 SE 通道注意力机制）
训练策略：与仓库 train.py 一致（SGD+momentum0.9+wd5e-4、warmup、MultiStepLR、
RandomCrop/HorizontalFlip 数据增强、每 epoch 测试集评估），支持断点续训。
产出：Top-1/Top-5、每类精确率/召回率/F1、混淆矩阵、单图推理时延、中文训练曲线图。
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

from conf import settings
from utils import WarmUpLR

# CIFAR-10 类别名（评估报告用）
CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

# CIFAR-10 全局均值/标准差
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_net(name, num_classes=10):
    """构建 CIFAR-10 版网络（仓库默认 100 类，这里改为 10 类）"""
    if name == 'resnet18':
        from models.resnet import ResNet, BasicBlock
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif name == 'seresnet18':
        from models.senet import SEResNet, BasicResidualSEBlock
        return SEResNet(BasicResidualSEBlock, [2, 2, 2, 2], class_num=num_classes)
    elif name == 'mobilenet':
        from models.mobilenet import MobileNet
        return MobileNet(class_num=num_classes)
    else:
        raise ValueError('不支持的网络: %s（可选 resnet18 / seresnet18 / mobilenet）' % name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18',
                        help='resnet18 / seresnet18(带SE注意力) / mobilenet')
    parser.add_argument('-b', type=int, default=128, help='batch size')
    parser.add_argument('-lr', type=float, default=0.1, help='initial lr')
    parser.add_argument('-epochs', type=int, default=60, help='total epochs')
    parser.add_argument('-milestones', type=str, default='30,45,55',
                        help='学习率衰减 epoch 节点，逗号分隔')
    parser.add_argument('-warm', type=int, default=1, help='warmup epochs')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-resume', type=str, default='', help='历史权重路径（断点续训）')
    parser.add_argument('-start_epoch', type=int, default=1, help='起始 epoch')
    parser.add_argument('-best_acc', type=float, default=0.0, help='历史最佳精度')
    args = parser.parse_args()

    print('PyTorch 版本:', torch.__version__, '| CUDA 可用:', torch.cuda.is_available())
    if not torch.cuda.is_available():
        print('=> 无独显，使用 CPU 训练。')
    torch.set_num_threads(os.cpu_count())
    print('CPU 线程数:', torch.get_num_threads())

    milestones = [int(x) for x in args.milestones.split(',')]
    print('网络: %s | 类别数: 10 | 学习率: %g | 衰减节点: %s | 总epoch: %d' % (
        args.net, args.lr, milestones, args.epochs))

    # ---- 数据加载（增强策略与仓库 train.py 一致）----
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = torchvision.datasets.CIFAR10('./data', train=True, download=False,
                                             transform=transform_train)
    test_set = torchvision.datasets.CIFAR10('./data', train=False, download=False,
                                            transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=args.b, shuffle=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=args.num_workers)
    print('训练集 %d 幅, 测试集 %d 幅, 类别数 10' % (len(train_set), len(test_set)))

    # ---- 模型 / 损失 / 优化器 ----
    net = build_net(args.net)
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location='cpu', weights_only=True))
        print('已加载历史权重:', args.resume, '，从 epoch %d 继续训练' % args.start_epoch)
    print('参数量(M):', round(sum(p.numel() for p in net.parameters()) / 1e6, 2))

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * args.warm)

    # ---- checkpoint 目录 ----
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ckpt_dir = os.path.join('checkpoint', 'cifar10_%s' % args.net, now)
    os.makedirs(ckpt_dir, exist_ok=True)
    print('checkpoint 目录:', ckpt_dir)

    # ---- 训练循环 ----
    best_acc = args.best_acc
    log_lines = []
    ep_times = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        net.train()
        t0 = time.time()
        ep_loss, n_batch = 0.0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batch += 1
            if epoch <= args.warm:
                warmup_scheduler.step()
        ep_time = time.time() - t0
        ep_times.append(ep_time)

        # 测试集评估（Top-1 / Top-5）
        net.eval()
        correct1 = correct5 = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = net(images)
                _, pred = outputs.max(1)
                correct1 += pred.eq(labels).sum().item()
                _, top5 = outputs.topk(5, 1)
                correct5 += labels.view(-1, 1).eq(top5).sum().item()
                total += labels.size(0)
        acc1, acc5 = 100.0 * correct1 / total, 100.0 * correct5 / total

        lr_now = optimizer.param_groups[0]['lr']
        line = 'epoch %d 训练loss=%.4f 测试Top-1=%.2f%% Top-5=%.2f%% lr=%.4f 耗时=%.1fs' % (
            epoch, ep_loss / n_batch, acc1, acc5, lr_now, ep_time)
        print(line)
        log_lines.append(line)

        if acc1 > best_acc:
            best_acc = acc1
            torch.save(net.state_dict(), os.path.join(ckpt_dir, '%s-%d-best.pth' % (args.net, epoch)))
            print('  -> 保存 best 权重 epoch %d acc %.2f%%' % (epoch, acc1))
        if epoch % 10 == 0:
            torch.save(net.state_dict(), os.path.join(ckpt_dir, '%s-%d-regular.pth' % (args.net, epoch)))

    # ---- 最终完整指标（用最佳权重评估）----
    best_files = [f for f in os.listdir(ckpt_dir) if f.endswith('-best.pth')]
    best_files = sorted(best_files, key=lambda w: int(w.split('-')[1]))
    net.load_state_dict(torch.load(os.path.join(ckpt_dir, best_files[-1]),
                                   map_location='cpu', weights_only=True))
    net.eval()
    y_true, y_pred = [], []
    top5_hits = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = net(images)
            y_pred.extend(outputs.argmax(1).numpy())
            y_true.extend(labels.numpy())
            _, top5 = outputs.topk(5, 1)
            top5_hits += labels.view(-1, 1).eq(top5).sum().item()
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # 单图推理时延（200 次平均）
    sample = test_set[0][0].unsqueeze(0)
    with torch.no_grad():
        for _ in range(10):
            net(sample)
        t0 = time.time()
        for _ in range(200):
            net(sample)
        infer_ms = (time.time() - t0) / 200 * 1000

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    summary = []
    summary.append('===== %s / CIFAR-10 最终测试结果 =====' % args.net)
    summary.append('Top-1 精度: %.2f%%' % (100.0 * (y_true == y_pred).mean()))
    summary.append('Top-5 精度: %.2f%%' % (100.0 * top5_hits / len(test_set)))
    summary.append('宏平均 F1 : %.4f' % macro_f1)
    summary.append('单图推理时延: %.2f ms' % infer_ms)
    summary.append('平均每 epoch 训练耗时: %.1fs' % (sum(ep_times) / len(ep_times)))
    summary.append('')
    summary.append(classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES, digits=4))
    summary.append('混淆矩阵 (行=真实, 列=预测):')
    summary.append(str(cm))
    print('\n'.join(summary))
    with open(os.path.join(ckpt_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n\n' + summary[0].replace('=====', '') + '\n' +
                '\n'.join(summary[1:]) + '\n')

    # ---- 训练曲线图 ----
    losses = [float(l.split('loss=')[1].split()[0]) for l in log_lines]
    accs = [float(l.split('Top-1=')[1].split('%')[0]) for l in log_lines]
    ep = range(args.start_epoch, args.start_epoch + len(log_lines))
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(ep, losses, 'o-', color='#3b5bdb', label='训练损失')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('训练损失', color='#3b5bdb')
    ax1.tick_params(axis='y', labelcolor='#3b5bdb')
    ax2 = ax1.twinx()
    ax2.plot(ep, accs, 'o-', color='#94d82d', label='测试Top-1精度')
    ax2.set_ylabel('Top-1 精度(%)', color='#5c940d'); ax2.tick_params(axis='y', labelcolor='#5c940d')
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc='lower right')
    plt.title('%s 在 CIFAR-10 的 CPU 训练曲线（epoch %d-%d）' % (args.net, args.start_epoch, args.epochs))
    fig.tight_layout()
    fig.savefig(os.path.join(ckpt_dir, 'curve.png'), dpi=130)
    print('结果已保存到:', ckpt_dir)


if __name__ == '__main__':
    main()
