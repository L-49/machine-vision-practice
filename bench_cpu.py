"""CPU 环境基准测试：实测 1 个 epoch 的训练耗时（供无独显电脑评估用）
复用原仓库 utils.py 的数据加载与模型构建逻辑，不做任何网络/参数改动。
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='mobilenet')
    parser.add_argument('-b', type=int, default=128)
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-epochs', type=int, default=1, help='基准测试运行的 epoch 数')
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count())
    print('CPU threads:', torch.get_num_threads())

    net = get_network(argparse.Namespace(net=args.net, gpu=False))
    print('网络:', args.net, '参数量(M):', round(sum(p.numel() for p in net.parameters()) / 1e6, 2))

    print('加载训练集...')
    t0 = time.time()
    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD,
        num_workers=args.num_workers, batch_size=args.b, shuffle=True)
    print('训练集加载完成, 耗时 %.1fs, 批数=%d' % (time.time() - t0, len(train_loader)))

    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD,
        num_workers=args.num_workers, batch_size=args.b, shuffle=False)
    print('测试集加载完成, 批数=%d' % len(test_loader))

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * 1)

    for epoch in range(1, args.epochs + 1):
        net.train()
        ep_t0 = time.time()
        batch_times = []
        bt0 = time.time()
        for batch_index, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            if epoch <= 1:
                warmup_scheduler.step()
            batch_times.append(time.time() - bt0)
            bt0 = time.time()
            if (batch_index + 1) % 50 == 0 or batch_index == len(train_loader) - 1:
                print('  epoch %d batch %d/%d loss=%.4f 批次耗时均值=%.2fs' % (
                    epoch, batch_index + 1, len(train_loader), loss.item(),
                    sum(batch_times) / len(batch_times)))
        ep_time = time.time() - ep_t0
        avg_batch = sum(batch_times) / len(batch_times)
        print('epoch %d 总耗时 %.1fs, 平均每批 %.3fs' % (epoch, ep_time, avg_batch))
        print('=> 按此速度, 200 epoch 预计需 %.1f 小时' % (ep_time * 200 / 3600))

        # 每个 epoch 结束做一次测试集精度评估
        net.eval()
        correct = 0.0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = net(images)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        print('epoch %d 测试集 Top-1 精度: %.2f%%' % (epoch, 100.0 * correct / total))


if __name__ == '__main__':
    main()
