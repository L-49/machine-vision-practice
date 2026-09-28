"""CPU 训练脚本（无独显电脑用）
完全复用原仓库 train.py 的训练流程（SGD+momentum0.9+wd5e-4、warmup、MultiStepLR、
RandomCrop/HorizontalFlip/Rotation 数据增强、每 epoch 测试集评估），仅将 epoch 数改为可配置，
并去掉 GPU 分支与 TensorBoard（避免额外依赖）。checkpoint 保存到 checkpoint/<net>/<时间戳>/，
命名格式与原仓库一致，可直接用 test.py 加载。
"""
import os
import sys
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='mobilenet', help='net type')
    parser.add_argument('-b', type=int, default=128, help='batch size')
    parser.add_argument('-warm', type=int, default=1, help='warmup epochs')
    parser.add_argument('-lr', type=float, default=0.1, help='initial lr')
    parser.add_argument('-epochs', type=int, default=10, help='total epochs (CPU 用)')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-resume', type=str, default='', help='要加载的历史权重路径（断点续训）')
    parser.add_argument('-start_epoch', type=int, default=1, help='起始 epoch（续训时设为上次结束 epoch+1）')
    parser.add_argument('-best_acc', type=float, default=0.0, help='历史最佳精度，续训时传入以保持 best 保存逻辑')
    parser.add_argument('-milestones', type=str, default='',
                        help='学习率衰减点，逗号分隔（如 40,50）；留空用 conf 默认 60,120,160')
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count())

    net = get_network(argparse.Namespace(net=args.net, gpu=False))
    print('网络:', args.net, '参数量(M):', round(sum(p.numel() for p in net.parameters()) / 1e6, 2))

    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location='cpu', weights_only=True))
        print('已加载历史权重:', args.resume, '，从 epoch %d 继续训练' % args.start_epoch)

    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD,
        num_workers=args.num_workers, batch_size=args.b, shuffle=True)
    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD,
        num_workers=args.num_workers, batch_size=args.b, shuffle=False)
    print('训练批数=%d, 测试批数=%d' % (len(train_loader), len(test_loader)))

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    milestones = ([int(x) for x in args.milestones.split(',') if x.strip()]
                   if args.milestones else settings.MILESTONES)
    print('学习率衰减点:', milestones)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * args.warm)

    # 与原仓库一致的 checkpoint 目录结构
    now = datetime.now().strftime(DATE_FORMAT)
    ckpt_dir = os.path.join(settings.CHECKPOINT_PATH, args.net, now)
    os.makedirs(ckpt_dir, exist_ok=True)
    print('checkpoint 目录:', ckpt_dir)

    best_acc = args.best_acc
    log_lines = []
    ep_times = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        net.train()
        ep_t0 = time.time()
        ep_loss = 0.0
        for batch_index, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            if epoch <= args.warm:
                warmup_scheduler.step()
        ep_time = time.time() - ep_t0
        ep_times.append(ep_time)

        # 测试集评估
        net.eval()
        correct = 0.0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = net(images)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / total
        line = 'epoch %d 训练loss=%.4f 耗时=%.1fs 测试Top-1精度=%.2f%%' % (
            epoch, ep_loss / len(train_loader), ep_time, acc)
        print(line)
        log_lines.append(line)

        # 保存 best 权重（与原仓库一致：超过历史最佳才覆盖）
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), os.path.join(ckpt_dir, '%s-%d-best.pth' % (args.net, epoch)))
            print('  -> 保存 best 权重 epoch %d acc %.2f%%' % (epoch, acc))
        if epoch % 5 == 0:
            torch.save(net.state_dict(), os.path.join(ckpt_dir, '%s-%d-regular.pth' % (args.net, epoch)))

    summary = '训练完成: epoch %d-%d, 最佳 Top-1 精度 %.2f%%, 单 epoch 平均耗时 %.1fs' % (
        args.start_epoch, args.epochs, best_acc, sum(ep_times) / len(ep_times))
    print(summary)
    with open(os.path.join(ckpt_dir, 'train_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n' + summary + '\n')


if __name__ == '__main__':
    main()
