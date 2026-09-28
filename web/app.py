"""机器视觉课程实践 —— Flask Web 系统

功能：
  1. 训练仪表盘：实时解析训练日志，Chart.js 展示 loss / Top-1 / Top-5 曲线
  2. 在线调参：选择网络(含 SE 注意力开关)、学习率、batch、epoch，后台启动训练（带任务锁）
  3. 在线推理：上传图片，加载最新权重输出 Top-3 预测类别与置信度
  4. 评估报告：展示 metrics.txt 的总体指标、每类精确率/召回率/F1、混淆矩阵
所有产物（日志/权重/上传文件）均位于 D 盘项目目录内。
"""
import os
import re
import sys
import json
import subprocess
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory

# ---- 路径配置（全在 D 盘）----
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(WEB_DIR)          # pytorch-cifar100/
LOG_DIR = os.path.join(PROJ_DIR, 'logs')
CKPT_DIR = os.path.join(PROJ_DIR, 'checkpoint')
UPLOAD_DIR = os.path.join(WEB_DIR, 'uploads')
LOCK_FILE = os.path.join(LOG_DIR, 'web_train.lock')

sys.path.insert(0, PROJ_DIR)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True  # 模板改动后刷新即生效，免重启
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024   # 上传上限 8MB

CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

# 实验前缀 -> (日志文件前缀, checkpoint 目录名, 网络参数描述)
EXPERIMENTS = {
    'cifar10_resnet18':   ('cifar10_resnet18',   'cifar10_resnet18'),
    'cifar10_seresnet18': ('cifar10_seresnet18', 'cifar10_seresnet18'),
    'cifar10_mobilenet':  ('cifar10_mobilenet',  'cifar10_mobilenet'),
    'cifar100_mobilenet': ('cifar100_mobilenet', 'mobilenet'),  # CIFAR-100 的 checkpoint 目录名是 mobilenet
}

# ---------------------------------------------------------------- 日志解析
# 格式 A（train_cifar10_cpu.py 输出，CIFAR-10 系列）：
# epoch 1 训练loss=1.6382 测试Top-1=50.07% Top-5=93.36% lr=0.1000 耗时=735.6s
LINE_RE_CIFAR10 = re.compile(
    r'epoch\s*(\d+).*?loss=([\d.]+).*?Top-1=([\d.]+)%.*?Top-5=([\d.]+)%'
    r'.*?lr=([\d.]+).*?耗时=([\d.]+)s')
# 格式 B（train_cpu.py 输出，CIFAR-100 MobileNet）：
# epoch 27 训练loss=0.1234 耗时=315.2s 测试Top-1精度=44.86%
LINE_RE_CIFAR100 = re.compile(
    r'epoch\s*(\d+).*?loss=([\d.]+).*?测试Top-1精度=([\d.]+)%')


def list_log_files(prefix):
    """返回指定实验前缀的所有日志文件（按时间倒序）"""
    if not os.path.isdir(LOG_DIR):
        return []
    fs = [f for f in os.listdir(LOG_DIR)
          if f.startswith(prefix + '_') and f.endswith('.log')]
    return sorted(fs, reverse=True)


def parse_log(path):
    """解析训练日志，自动兼容两种格式，返回每个 epoch 的指标列表"""
    rows = []
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = LINE_RE_CIFAR10.search(line)
                if m:
                    rows.append({
                        'epoch': int(m.group(1)),
                        'loss': float(m.group(2)),
                        'top1': float(m.group(3)),
                        'top5': float(m.group(4)),
                        'lr': float(m.group(5)),
                        'time': float(m.group(6)),
                    })
                    continue
                m = LINE_RE_CIFAR100.search(line)
                if m:
                    rows.append({
                        'epoch': int(m.group(1)),
                        'loss': float(m.group(2)),
                        'top1': float(m.group(3)),
                        'top5': None,
                        'lr': None,
                        'time': None,
                    })
    except OSError:
        pass
    return rows


def parse_log_total_epochs(path):
    """从日志头部解析'总epoch: X'（训练脚本输出的实际计划 epochs），
    返回 X 或 None（无此行时回退到 TOTAL_EPOCHS 硬编码值）。"""
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            for _ in range(30):   # 只看前 30 行
                line = f.readline()
                if not line:
                    break
                m = re.search(r'总epoch[:：]\s*(\d+)', line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return None


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False
    except Exception:
        # Windows 下无权限时 os.kill(pid,0) 也可能成功；按存活处理
        return True


def read_lock():
    """读取训练任务锁；日志停滞或 epoch 已到目标才视为无锁并清理。
    Windows 下 os.kill(pid,0) 对权限/复用 PID 不可靠，故优先用日志活跃度判定：
    日志 20 分钟内更新且未到目标 epoch -> 视为训练在跑，直接返回 lock。
    （阈值 1200s 容忍 CIFAR-10 单 epoch 9-12 分钟的间隔，避免 epoch 间误判 stale）
    无日志路径时才用 pid_alive 兜底。"""
    if not os.path.exists(LOCK_FILE):
        return None
    try:
        with open(LOCK_FILE, encoding='utf-8') as f:
            lock = json.load(f)
        log_path = lock.get('log')
        if log_path and os.path.exists(log_path):
            age = (datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(log_path))).total_seconds()
            if age > 1200:
                raise OSError('log stale %.0fs' % age)
            rows = parse_log(log_path)
            if rows and rows[-1]['epoch'] >= int(lock.get('epochs', 10**9)):
                raise OSError('reached target epoch %d' % rows[-1]['epoch'])
            # 日志在更新 -> 训练在跑，直接返回（不依赖 pid_alive）
            return lock
        # 无日志路径，用 pid_alive 兜底
        if not pid_alive(int(lock.get('pid', -1))):
            raise OSError('process dead')
        return lock
    except (ValueError, OSError):
        pass
    try:
        os.remove(LOCK_FILE)
    except OSError:
        pass
    return None


def collect_rows(prefix):
    """合并某实验所有日志文件的行（按 epoch 去重，新文件的行覆盖旧文件），
    使分段训练（如 CIFAR-100 的 1-30 与 31-60）在曲线上连续。"""
    merged = {}
    for f in list_log_files(prefix):
        for r in parse_log(os.path.join(LOG_DIR, f)):
            merged[r['epoch']] = r
    return [merged[e] for e in sorted(merged)]


def latest_log_age(prefix):
    """某实验最新日志距现在的秒数（无日志返回 None）"""
    files = list_log_files(prefix)
    if not files:
        return None
    return (datetime.now() - datetime.fromtimestamp(
        os.path.getmtime(os.path.join(LOG_DIR, files[0])))).total_seconds()


# 各实验计划总 epoch：最后一行 epoch 达到此值即视为训练已结束，
# 不再因日志文件时间戳被同步刷新而误报“训练中”。
TOTAL_EPOCHS = {
    'cifar10_resnet18': 60,
    'cifar10_seresnet18': 60,
    'cifar10_mobilenet': 60,
    'cifar100_mobilenet': 60,
}


def latest_best_weight(ckpt_subdir):
    """找某实验最新时间戳目录下的最佳权重"""
    base = os.path.join(CKPT_DIR, ckpt_subdir)
    if not os.path.isdir(base):
        return None
    stamps = sorted(os.listdir(base), reverse=True)
    for s in stamps:
        d = os.path.join(base, s)
        if os.path.isdir(d):
            best = sorted([f for f in os.listdir(d) if f.endswith('-best.pth')],
                          key=lambda w: int(w.split('-')[1]))
            if best:
                return os.path.join(d, best[-1])
    return None


# ---------------------------------------------------------------- 页面
@app.route('/')
def dashboard():
    return render_template('dashboard.html', active='dashboard')


@app.route('/train')
def train_page():
    return render_template('train.html', active='train')


@app.route('/predict')
def predict_page():
    return render_template('predict.html', active='predict')


@app.route('/evaluate')
def evaluate_page():
    return render_template('evaluate.html', active='evaluate')


# ---------------------------------------------------------------- API
@app.route('/api/progress')
def api_progress():
    """返回某实验（默认当前 CIFAR-10 训练）的曲线数据与状态"""
    exp = request.args.get('exp', 'cifar10_resnet18')
    prefix = EXPERIMENTS.get(exp, (exp, ''))[0]
    files = list_log_files(prefix)
    log_name = files[0] if files else ''
    rows = collect_rows(prefix)

    last_epoch = rows[-1]['epoch'] if rows else 0
    reached_total = last_epoch >= TOTAL_EPOCHS.get(exp, 10**9)

    lock = read_lock()
    running = bool(lock and lock.get('exp') == exp)
    # 兜底：无锁但日志 20 分钟内还在更新，也视为训练中（auto 脚本启动的训练）；
    # 但若最后一行已到计划总 epoch，则训练确已结束，不再误报。
    age = latest_log_age(prefix)
    if not running and age is not None and age < 1200 and not reached_total:
        running = True

    return jsonify({
        'exp': exp,
        'log': log_name,
        'running': running,
        'rows': rows,
        'lock': lock,
    })


@app.route('/api/trainings')
def api_trainings():
    """汇总三个实验的最新简要结果，供仪表盘卡片使用"""
    out = []
    for exp, (prefix, subdir) in EXPERIMENTS.items():
        rows = collect_rows(prefix)
        w = latest_best_weight(subdir)
        last_epoch = rows[-1]['epoch'] if rows else 0
        reached_total = last_epoch >= TOTAL_EPOCHS.get(exp, 10**9)
        age = latest_log_age(prefix)
        active = age is not None and age < 1200 and not reached_total
        files = list_log_files(prefix)        # 按时间倒序
        latest_log_name = files[0] if files else ''
        out.append({
            'exp': exp,
            'epochs_done': len(rows),
            'best_top1': max((r['top1'] for r in rows), default=0),
            'last_loss': rows[-1]['loss'] if rows else None,
            'has_weight': bool(w),
            'active': active,
            'latest_log': latest_log_name,
        })
    return jsonify({'items': out, 'lock': read_lock()})


def any_log_active():
    """是否存在任何活跃训练日志（auto 脚本启动、非 web 锁的训练也能检测）。
    排除 flask_*.log、chain_monitor.log 等非训练日志；空文件、已到计划总 epoch
    的日志也不算活跃（避免某次启动失败留下的空壳文件长期误报为训练中）。"""
    if not os.path.isdir(LOG_DIR):
        return False
    now = datetime.now()
    TRAIN_PREFIXES = ('cifar10_', 'cifar100_', 'mnist_')
    for f in os.listdir(LOG_DIR):
        if not f.endswith('.log'):
            continue
        if not any(f.startswith(p) for p in TRAIN_PREFIXES):
            continue  # 跳过 flask_*.log、chain_monitor.log 等
        path = os.path.join(LOG_DIR, f)
        # 空文件：上次启动失败留下的空壳，不算活跃
        if os.path.getsize(path) == 0:
            continue
        age = (now - datetime.fromtimestamp(os.path.getmtime(path))).total_seconds()
        if age >= 1200:
            continue
        # 20 分钟内更新过且非空 —— 再看是否已产出 epoch 数据
        rows = parse_log(path)
        if not rows:
            # 非空但解析不出 epoch 行（如刚启动就崩溃只剩环境信息），不算活跃
            continue
        last_epoch = rows[-1]['epoch']
        # 优先从日志内容解析实际总 epoch（支持用户填 5/3 等非 60 的值），
        # 回退到 EXPERIMENTS 硬编码值
        total = parse_log_total_epochs(path)
        if not total:
            for exp, (prefix, _sub) in EXPERIMENTS.items():
                if f.startswith(prefix):
                    total = TOTAL_EPOCHS.get(exp)
                    break
        if total and last_epoch >= total:
            continue  # 已到总 epoch，视为已结束
        return True
    return False


@app.route('/api/start', methods=['POST'])
def api_start():
    """在线调参：启动后台训练（同一时刻只允许一个任务）"""
    if read_lock():
        return jsonify({'ok': False, 'msg': '已有训练任务在运行，请等待其结束'})
    if any_log_active():
        return jsonify({'ok': False, 'msg': '检测到训练日志正在更新（可能由自动脚本启动），请等待其结束'})

    net = request.form.get('net', 'resnet18')
    net_map = {
        'resnet18': ('cifar10_resnet18', 'resnet18'),
        'seresnet18': ('cifar10_seresnet18', 'seresnet18'),
        'mobilenet': ('cifar10_mobilenet', 'mobilenet'),
    }
    if net not in net_map:
        return jsonify({'ok': False, 'msg': '不支持的网络'})
    exp, net_arg = net_map[net]

    try:
        lr = float(request.form.get('lr', 0.1))
        batch = int(request.form.get('batch', 128))
        epochs = int(request.form.get('epochs', 60))
        milestones = request.form.get('milestones', '30,45,55')
        # 简单校验，防止非法参数拖垮机器
        if not (0.00001 <= lr <= 1.0):
            raise ValueError('学习率需在 1e-5 ~ 1.0 之间')
        if not (16 <= batch <= 512):
            raise ValueError('batch 需在 16 ~ 512 之间')
        if not (1 <= epochs <= 200):
            raise ValueError('epoch 需在 1 ~ 200 之间')
        [int(x) for x in milestones.split(',')]
    except ValueError as e:
        return jsonify({'ok': False, 'msg': '参数错误: %s' % e})

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_DIR, '%s_%s.log' % (exp, stamp))
    err_path = os.path.join(LOG_DIR, '%s_%s.err' % (exp, stamp))

    # 用 python.exe（不是 pythonw）启动训练：pythonw 无控制台句柄，
    # DataLoader 的 multiprocessing worker spawn 时会因无 console 静默崩溃。
    # 改用 python.exe + CREATE_NO_WINDOW：进程有 console 句柄（worker 能正常 spawn），
    # 但不弹出窗口；stdout/stderr 重定向到文件，避免 stderr 警告经管道拖崩 Flask。
    py = sys.executable
    train_script = os.path.join(PROJ_DIR, 'train_cifar10_cpu.py')
    # 先打开文件句柄，传给子进程；子进程退出后文件句柄由 OS 自动回收
    log_fp = open(log_path, 'w', encoding='utf-8')
    err_fp = open(err_path, 'w', encoding='utf-8')
    proc = subprocess.Popen(
        [py, '-X', 'utf8', '-u', train_script,
         '-net', net_arg, '-b', str(batch), '-lr', str(lr),
         '-epochs', str(epochs), '-milestones', milestones],
        cwd=PROJ_DIR,
        stdout=log_fp, stderr=err_fp,
        # CREATE_NEW_PROCESS_GROUP：子进程不受 Flask 信号影响
        # CREATE_NO_WINDOW：不弹出控制台窗口（multiprocessing worker 也不弹窗）
        creationflags=0x00000200 | 0x08000000,
        close_fds=True)
    # 不立即关闭文件句柄，子进程需要；Python GC 会延迟回收，不影响子进程
    log_fp.close()
    err_fp.close()

    lock = {'pid': proc.pid, 'exp': exp, 'net': net_arg,
            'lr': lr, 'batch': batch, 'epochs': epochs,
            'start': datetime.now().strftime('%H:%M:%S'), 'log': log_path}
    with open(LOCK_FILE, 'w', encoding='utf-8') as f:
        json.dump(lock, f, ensure_ascii=False, indent=2)
    return jsonify({'ok': True, 'lock': lock})


SAMPLES_DIR = os.path.join(WEB_DIR, 'static', 'samples')


@app.route('/api/samples')
def api_samples():
    """返回样本库 manifest（按类别分组，每次随机抽样 5 张/类，避免同一批被怀疑过拟合）"""
    import random as _r
    manifest_path = os.path.join(SAMPLES_DIR, 'manifest.json')
    if not os.path.exists(manifest_path):
        return jsonify({'ok': False, 'msg': '样本库未生成，请先运行 extract_samples.py'})
    with open(manifest_path, encoding='utf-8') as f:
        items = json.load(f)
    groups = {}
    for it in items:
        groups.setdefault(it['cls'], []).append(it)
    rng = _r.Random()
    sampled = {}
    for cls, lst in groups.items():
        k = min(5, len(lst))
        sampled[cls] = rng.sample(lst, k)
    return jsonify({'ok': True, 'groups': sampled,
                    'total': sum(len(v) for v in sampled.values()),
                    'pool_total': len(items)})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """上传图片或选择样本库图片进行在线推理，返回 Top-3 类别与置信度"""
    import torch
    from PIL import Image
    from train_cifar10_cpu import build_net
    import torchvision.transforms as transforms

    sample_name = request.form.get('sample')  # 样本库文件名，如 airplane_0.png
    if sample_name:
        sample_path = os.path.join(SAMPLES_DIR, sample_name)
        if not os.path.exists(sample_path):
            return jsonify({'ok': False, 'msg': '样本不存在: ' + sample_name})
        # 直接用样本原图作为展示图（32x32 PNG）
        save_name = datetime.now().strftime('%H%M%S_') + sample_name
        save_path = os.path.join(UPLOAD_DIR, save_name)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        # 拷贝一份到 uploads 便于历史回顾
        from shutil import copyfile
        copyfile(sample_path, save_path)
    else:
        f = request.files.get('image')
        if not f:
            return jsonify({'ok': False, 'msg': '未收到图片'})
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        save_name = datetime.now().strftime('%H%M%S_') + f.filename
        save_path = os.path.join(UPLOAD_DIR, save_name)
        f.save(save_path)

    # 优先用 resnet18 最新权重，没有则尝试 seresnet18
    net_name, weight = 'resnet18', latest_best_weight('cifar10_resnet18')
    if not weight:
        net_name, weight = 'seresnet18', latest_best_weight('cifar10_seresnet18')
    if not weight:
        return jsonify({'ok': False, 'msg': '暂无训练好的模型权重，请先完成训练'})

    net = build_net(net_name)
    net.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True))
    net.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    img = Image.open(save_path).convert('RGB')
    x = transform(img).unsqueeze(0)

    import time as _t
    t0 = _t.time()
    with torch.no_grad():
        prob = torch.softmax(net(x), dim=1)[0]
    infer_ms = (_t.time() - t0) * 1000
    topv, topi = prob.topk(3)
    preds = [{'cls': CIFAR10_CLASSES[i], 'conf': round(v * 100, 2)}
             for v, i in zip(topv.tolist(), topi.tolist())]
    return jsonify({'ok': True, 'preds': preds, 'net': net_name,
                    'infer_ms': round(infer_ms, 2),
                    'img': '/uploads/' + save_name})


@app.route('/uploads/<path:name>')
def uploads(name):
    return send_from_directory(UPLOAD_DIR, name)


# 每类指标行：类名(单词) precision recall f1 support
CLASS_ROW_RE = re.compile(
    r'^\s+([A-Za-z][A-Za-z0-9_-]*)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$',
    re.M)
_SKIP_CLASS_NAMES = {'accuracy', 'macro', 'weighted'}


def parse_metrics_structured(text):
    """把 metrics.txt 解析为：总体指标 summary、每类指标 per_class、混淆矩阵 confusion。
    解析失败的部分返回 None，前端可降级显示原文。"""
    def grab(pat):
        m = pat.search(text)
        return float(m.group(1)) if m else None

    summary = {
        'top1': grab(re.compile(r'Top-1 精度[:：]\s*([\d.]+)%')),
        'top5': grab(re.compile(r'Top-5 精度[:：]\s*([\d.]+)%')),
        'f1': grab(re.compile(r'宏平均 F1\s*[:：]\s*([\d.]+)')),
        'infer_ms': grab(re.compile(r'单图推理时延[:：]\s*([\d.]+)\s*ms')),
        'epoch_s': grab(re.compile(r'平均每 epoch 训练耗时[:：]\s*([\d.]+)\s*s')),
    }

    per_class = []
    for m in CLASS_ROW_RE.finditer(text):
        name = m.group(1)
        if name in _SKIP_CLASS_NAMES:
            continue
        per_class.append({
            'class': name,
            'precision': float(m.group(2)),
            'recall': float(m.group(3)),
            'f1': float(m.group(4)),
            'support': int(m.group(5)),
        })

    # 混淆矩阵：取 [[...]] 块。文件中数字、行之间均只有空白、没有逗号，
    # 无法直接用 ast 解析，因此提取所有整数后按行数重塑。
    confusion = None
    cm = re.search(r'\[\[.*?\]\]', text, re.S)
    if cm:
        raw = cm.group(0)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        nrows = raw.count('\n') + 1
        if vals and len(vals) % nrows == 0:
            ncols = len(vals) // nrows
            confusion = [vals[i*ncols:(i+1)*ncols] for i in range(nrows)]

    return {'summary': summary, 'per_class': per_class, 'confusion': confusion}


@app.route('/api/metrics')
def api_metrics():
    """读取评估报告（最新实验的 metrics.txt），同时返回原文与结构化数据"""
    exp = request.args.get('exp', 'cifar10_resnet18')
    subdir = EXPERIMENTS.get(exp, ('', exp))[1]
    base = os.path.join(CKPT_DIR, subdir)
    if not os.path.isdir(base):
        return jsonify({'ok': False, 'msg': '暂无该实验的评估结果'})
    for s in sorted(os.listdir(base), reverse=True):
        mp = os.path.join(base, s, 'metrics.txt')
        if os.path.exists(mp):
            text = open(mp, encoding='utf-8', errors='ignore').read()
            return jsonify({
                'ok': True, 'text': text, 'stamp': s,
                'data': parse_metrics_structured(text),
            })
    return jsonify({'ok': False, 'msg': 'metrics.txt 尚未生成（训练全部结束后才有）'})


# ---------------------------------------------------------------- 历史训练曲线
EXP_NAME_MAP = {
    'cifar10_resnet18':   'CIFAR-10 · ResNet18',
    'cifar10_seresnet18': 'CIFAR-10 · SEResNet18',
    'cifar10_mobilenet':  'CIFAR-10 · MobileNet',
    'cifar100_mobilenet': 'CIFAR-100 · MobileNet',
}


def parse_log_filename(fname):
    """从训练日志文件名解析 (exp_key, prefix, datetime)，非训练日志返回 (None,None,None)"""
    for exp, (prefix, _sub) in EXPERIMENTS.items():
        if fname.startswith(prefix + '_') and fname.endswith('.log'):
            ts = fname[len(prefix) + 1:].rsplit('.', 1)[0]
            try:
                dt = datetime.strptime(ts, '%Y%m%d_%H%M%S')
                return exp, prefix, dt
            except ValueError:
                return exp, prefix, None
    return None, None, None


@app.route('/api/history')
def api_history():
    """列出所有历次训练（每个日志文件 = 一次启动），按时间倒序"""
    if not os.path.isdir(LOG_DIR):
        return jsonify({'items': []})
    out = []
    for f in os.listdir(LOG_DIR):
        exp, prefix, dt = parse_log_filename(f)
        if exp is None:
            continue
        path = os.path.join(LOG_DIR, f)
        if os.path.getsize(path) == 0:
            continue  # 空壳日志（启动失败留下的），不计入历史
        rows = parse_log(path)
        best = max((r['top1'] for r in rows), default=0)
        last_epoch = rows[-1]['epoch'] if rows else 0
        last_loss = rows[-1]['loss'] if rows else None
        last_lr = rows[-1].get('lr') if rows else None
        total = parse_log_total_epochs(path) or TOTAL_EPOCHS.get(exp, 0)
        done = last_epoch >= total if total else False
        subdir = EXPERIMENTS[exp][1]
        weight = latest_best_weight(subdir)
        out.append({
            'file': f,
            'exp': exp,
            'exp_name': EXP_NAME_MAP.get(exp, exp),
            'time': dt.strftime('%Y-%m-%d %H:%M:%S') if dt else '',
            'time_ts': dt.timestamp() if dt else 0,
            'epochs_done': len(rows),
            'last_epoch': last_epoch,
            'total_epochs': total,
            'done': done,
            'best_top1': best,
            'last_loss': last_loss,
            'last_lr': last_lr,
            'has_weight': bool(weight),
        })
    out.sort(key=lambda x: x['time_ts'], reverse=True)
    return jsonify({'items': out})


@app.route('/api/history_progress')
def api_history_progress():
    """返回单次训练（指定日志文件）的曲线数据"""
    fname = request.args.get('file', '')
    exp, prefix, dt = parse_log_filename(fname)
    if exp is None or not fname.endswith('.log'):
        return jsonify({'ok': False, 'msg': '非法文件'}), 400
    path = os.path.join(LOG_DIR, fname)
    if not os.path.isfile(path):
        return jsonify({'ok': False, 'msg': '文件不存在'}), 404
    rows = parse_log(path)
    total = parse_log_total_epochs(path) or TOTAL_EPOCHS.get(exp, 0)
    return jsonify({
        'ok': True,
        'file': fname,
        'exp': exp,
        'exp_name': EXP_NAME_MAP.get(exp, exp),
        'time': dt.strftime('%Y-%m-%d %H:%M:%S') if dt else '',
        'rows': rows,
        'total_epochs': total,
        'best_top1': max((r['top1'] for r in rows), default=0),
        'has_top5': exp != 'cifar100_mobilenet',
    })


@app.route('/api/history_delete', methods=['POST'])
def api_history_delete():
    """删除指定的训练日志文件（仅允许训练日志）"""
    fname = request.form.get('file', '')
    exp, prefix, _ = parse_log_filename(fname)
    if exp is None or not fname.endswith('.log'):
        return jsonify({'ok': False, 'msg': '非法文件'})
    path = os.path.join(LOG_DIR, fname)
    if not os.path.isfile(path):
        return jsonify({'ok': False, 'msg': '文件不存在'})
    try:
        os.remove(path)
        return jsonify({'ok': True})
    except OSError as e:
        return jsonify({'ok': False, 'msg': str(e)})


@app.route('/api/history_download')
def api_history_download():
    """下载指定训练对应的最佳权重文件"""
    fname = request.args.get('file', '')
    exp, prefix, _ = parse_log_filename(fname)
    if exp is None:
        return '非法文件', 400
    subdir = EXPERIMENTS[exp][1]
    weight = latest_best_weight(subdir)
    if not weight or not os.path.isfile(weight):
        return '未找到权重文件', 404
    return send_from_directory(
        os.path.dirname(weight), os.path.basename(weight), as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
