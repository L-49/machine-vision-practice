# -*- coding: utf-8 -*-
# Extract 20 samples per class (200 total) from CIFAR-10 test_batch as 32x32 PNG.
import os, sys, pickle, json, random
import numpy as np
from PIL import Image

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJ_DIR, 'data', 'cifar-10-batches-py', 'test_batch')
OUT_DIR = os.path.join(PROJ_DIR, 'web', 'static', 'samples')
CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
PER_CLASS = 20  # pool size: 200 total, frontend samples 50 randomly each time

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # clear old samples
    for f in os.listdir(OUT_DIR):
        if f.endswith('.png') or f == 'manifest.json':
            try: os.remove(os.path.join(OUT_DIR, f))
            except OSError: pass
    with open(DATA_PATH, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    data = d[b'data']
    labels = d[b'labels']
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    # group indices by class
    by_class = {i: [] for i in range(10)}
    for idx, lab in enumerate(labels):
        by_class[lab].append(idx)
    # random sample PER_CLASS per class
    rng = random.Random(2026)  # reproducible
    manifest = []
    for cls_idx, idxs in by_class.items():
        cls_name = CIFAR10_CLASSES[cls_idx]
        chosen = rng.sample(idxs, PER_CLASS) if len(idxs) >= PER_CLASS else idxs
        for k, img_idx in enumerate(chosen):
            img = Image.fromarray(data[img_idx], mode='RGB')
            fname = '{}_{:02d}.png'.format(cls_name, k)
            img.save(os.path.join(OUT_DIR, fname))
            manifest.append({'cls': cls_name, 'idx': k, 'file': fname})
    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print('OK: pool of {} samples written to {}'.format(len(manifest), OUT_DIR))
    print('Per class:', {CIFAR10_CLASSES[i]: PER_CLASS for i in range(10)})

if __name__ == '__main__':
    main()