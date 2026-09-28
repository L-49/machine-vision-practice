# 机器视觉课程实践：CIFAR / MNIST 图像分类实验系统

本项目在 [weiaicunzai/pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100) 开源框架基础上，
面向机器视觉课程实践完成二次开发，实现 **CPU 端可复现的图像分类训练 + Web 可视化实验平台**。

## 项目目标

- 在无 GPU 的教学环境下，完成 CIFAR-10 / CIFAR-100 / MNIST 的模型训练与精度对比
- 通过 SE 注意力机制（SE-ResNet18）等结构差异实验，分析模型容量与数据集复杂度的匹配关系
- 提供可视化 Web 界面，支持在线调参、实时训练曲线、实验对比和单图预测

## 数据集

| 数据集 | 类别数 | 训练集 | 测试集 | 用途 |
| :----: | :----: | :----: | :----: | :--- |
| CIFAR-10 | 10 | 50000 | 10000 | 基础分类（ResNet18、MobileNet） |
| CIFAR-100 | 100 | 50000 | 10000 | 难度对比（MobileNet） |
| MNIST | 10 | 60000 | 10000 | 入门验证（LeNet） |

数据集压缩包因体积较大未纳入仓库，可从以下地址获取并放到项目根目录，由脚本自动解压：

- CIFAR-10: `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
- CIFAR-100: `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`
- MNIST 由 `torchvision.datasets.MNIST` 自动下载

## 实验模型

- **ResNet18**：CIFAR-10 主干模型，60 epoch，目标 Top-1 ≥ 90%
- **SE-ResNet18**：在 ResNet18 基础上加入 Squeeze-and-Excitation 注意力，用于对比实验
- **MobileNet**（3.3M 参数）：CIFAR-100 轻量模型，CPU 端可达约 65% Top-1
- **LeNet**：MNIST 入门模型

## 功能模块

### 训练脚本

| 文件 | 说明 |
| :--- | :--- |
| `train_cifar10_cpu.py` | CIFAR-10 的 ResNet18 / SE-ResNet18 / MobileNet CPU 训练，支持断点续训 |
| `mnist_lenet.py` | MNIST LeNet 训练 |
| `train.py` / `train_cpu.py` | 原框架的多模型通用训练脚本 |
| `bench_cpu.py` | CPU 推理基准测试 |
| `lr_finder.py` | 学习率查找 |

### 自动化脚本

| 文件 | 说明 |
| :--- | :--- |
| `auto_train_cifar10.ps1` | CIFAR-10 多实验串联自动训练 |
| `auto_chain_mobilenet.ps1` | MobileNet 多实验串联训练 |
| `monitor_chain.ps1` | 串联训练进程监控 |

### Web 可视化系统（`web/`）

基于 Flask + Chart.js，紫色渐变 UI，提供以下页面：

- **在线调参（train）**：选择模型/数据集/学习率等超参数，实时训练曲线（loss、Top-1、Top-5），每 5 秒轮询刷新
- **实验仪表盘（dashboard）**：全部实验概况，Top-1 精度对比柱形图 + 训练 Loss 曲线对比折线图，左右布局，窄屏自动堆叠
- **模型评估报告（evaluate）**：混淆矩阵 + 综合评估结论，宽屏左右排版
- **在线预测（predict）**：上传图片即时识别

后端 API 包括训练启动、进度查询、历史训练记录、权重下载、日志删除等。

## 项目结构

```
pytorch-cifar100/
├── conf/                  # 全局超参数配置
├── models/                # 各网络结构实现（resnet, mobilenet, senet 等）
├── web/                   # Flask Web 可视化系统
│   ├── app.py             # 后端入口
│   ├── static/            # CSS / JS / 样例图片
│   └── templates/         # HTML 页面
├── train_cifar10_cpu.py   # CIFAR-10 CPU 训练主脚本
├── mnist_lenet.py         # MNIST 训练
├── bench_cpu.py           # CPU 基准测试
├── lr_finder.py           # 学习率查找
├── auto_*.ps1             # 自动训练脚本
└── dataset.py / utils.py  # 数据加载与工具函数
```

> `data/`、`checkpoint/`、`logs/`、`web/uploads/` 等运行时产物已通过 `.gitignore` 排除，
> 需自行下载数据集并训练生成权重。

## 运行环境

- Python 3.6+
- PyTorch 1.6.0+（CPU 版即可）
- Flask
- Chart.js（已内置在 `web/static/js/`）

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision flask
```

### 2. 启动训练

```bash
# CIFAR-10 ResNet18，60 epoch
python train_cifar10_cpu.py -net resnet18 -epoch 60

# MNIST LeNet
python mnist_lenet.py
```

### 3. 启动 Web 系统

```bash
cd web
python app.py
# 浏览器访问 http://127.0.0.1:5000
```

## 协作规范（Fork + Pull Request）

本仓库采用 **Fork + PR** 工作流，主干由维护者审核合并，确保贡献可追溯：

1. **Fork 仓库**：在 GitHub 页面点击 Fork，将仓库复制到自己账号
2. **克隆 Fork**：`git clone https://github.com/<你的用户名>/machine-vision-practice.git`
3. **新建分支**：`git checkout -b feat/<功能名>`
4. **提交改动**：`git add -A && git commit -m "feat: 简要描述"`
5. **推送分支**：`git push origin feat/<功能名>`
6. **发起 PR**：在 GitHub 上从你的分支向 `L-49/machine-vision-practice` 的 `main` 分支发起 Pull Request
7. **审核合并**：维护者审核通过后合并，合并后分支自动删除

提交信息建议遵循 Conventional Commits 规范：`feat:` / `fix:` / `docs:` / `refactor:` 等。

## 许可

基于 [weiaicunzai/pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100) 二次开发，
保留原项目许可协议。
