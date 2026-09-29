"""Global measured-intensity inversion followed by a trained residual CNN.

The ridge layer is frozen, with a convolutional residual trained on only the
40,000 training pairs. Validation chooses training noise and the checkpoint;
the holdout is evaluated once after selection.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from scipy.io import savemat
from skimage.metrics import structural_similarity

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
RIDGE = REPO_ROOT / "models" / "ridge12544"
OUT = REPO_ROOT / "models" / "neural"


class ResidualCNN(nn.Module):
    def __init__(self, channels=32, depth=6):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU()]
        layers += [nn.Conv2d(channels, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return x + self.body(x)


def metrics(p, x, with_ssim=True):
    p = np.clip(p, 0, 1).reshape(-1, 28, 28)
    x = x.reshape(-1, 28, 28)
    mse = ((p - x) ** 2).mean((1, 2))
    result = dict(mse=float(mse.mean()), psnr_mean_db=float((-10 * np.log10(np.maximum(mse, 1e-15))).mean()), count=len(x))
    if with_ssim:
        result['ssim_mean'] = float(np.mean([structural_similarity(a, b, data_range=1) for a, b in zip(x, p)]))
    return result


def save_grid(images, filename, labels=None, scale=10):
    images = np.clip(images.reshape(-1, 28, 28), 0, 1)
    n = len(images)
    cell = 28 * scale
    canvas = Image.new('RGB', (cell * n + 20 * (n + 1), cell + 60), '#eff1f4')
    draw = ImageDraw.Draw(canvas)
    for i, arr in enumerate(images):
        im = Image.fromarray(np.rint(arr * 255).astype('uint8')).resize((cell, cell), Image.Resampling.NEAREST).convert('RGB')
        canvas.paste(im, (20 + i * (cell + 20), 40))
        draw.text((20 + i * (cell + 20), 15), labels[i] if labels else f'Test {i + 1}', fill='black')
    canvas.save(filename)


@torch.inference_mode()
def predict(net, x, batch=512):
    return torch.cat([net(x[i:i+batch]).clamp(0, 1) for i in range(0, len(x), batch)])


def prepare():
    cache = OUT / 'ridge_raw.npy'
    cache_test = OUT / 'ridge_test_raw.npy'
    if cache.exists() and cache_test.exists():
        return np.load(cache), np.load(cache_test)
    m = np.load(RIDGE / 'model.npz')
    w = torch.from_numpy(m['weights']).cuda()
    ym = torch.from_numpy(m['y_mean']).cuda()
    xm = torch.from_numpy(m['x_mean']).cuda()
    feat = m['features']
    arrays = []
    for name in ['Y.npy', 'test_Y.npy']:
        y = np.load(PROCESSED / name, mmap_mode='r').reshape(-1, 12544)
        output = np.empty((len(y), 1, 28, 28), np.float32)
        with torch.inference_mode():
            for lo in range(0, len(y), 512):
                raw = torch.tensor(np.asarray(y[lo:lo+512]), device='cuda')
                norm = raw.mean(1, keepdim=True)
                e = raw.sum(1, keepdim=True).sqrt()
                inp = (raw[:, feat] / norm) - ym
                output[lo:lo+len(raw)] = ((inp @ w + xm) * e).reshape(-1, 1, 28, 28).cpu().numpy()
        arrays.append(output)
    np.save(cache, arrays[0])
    np.save(cache_test, arrays[1])
    return arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=int, default=420)
    ap.add_argument('--max_epochs', type=int, default=30)
    args = ap.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(20260929)
    np.random.seed(20260929)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    start = time.time()
    raw, raw_test = prepare()
    target = np.asarray(np.load(PROCESSED / 'X.npy', mmap_mode='r')[:50000]).copy()[:, None]
    split = np.load(RIDGE / 'split.npz')
    tr, va, ho = (split[k] for k in ['train', 'validation', 'holdout'])
    train_error = raw[tr] - target[tr]
    val_error = raw[va] - target[va]
    diagnostic = dict(train_raw_mse=float((train_error ** 2).mean()), validation_raw_mse=float((val_error ** 2).mean()))
    noise_sigma = float(np.sqrt(max(0, diagnostic['validation_raw_mse'] - diagnostic['train_raw_mse'])))
    diagnostic['noise_sigma_match'] = noise_sigma
    print('diagnostic', json.dumps(diagnostic), flush=True)
    (OUT / 'diagnostic.json').write_text(json.dumps(diagnostic, indent=2), encoding='utf8')
    train_x = torch.from_numpy(raw[tr]).cuda()
    train_y = torch.from_numpy(target[tr]).cuda()
    valid_x = torch.from_numpy(raw[va]).cuda()
    valid_y = torch.from_numpy(target[va]).cuda()
    test_x = torch.from_numpy(raw_test).cuda()
    net = ResidualCNN().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    log = []
    best = float('inf')
    best_epoch = -1
    train_start = time.time()
    for epoch in range(args.max_epochs):
        net.train()
        permutation = torch.randperm(len(train_x), device='cuda')
        sums = 0
        for j in range(0, len(train_x), 256):
            ids = permutation[j:j+256]
            inp, out = train_x[ids], train_y[ids]
            # Vary augmentation around validation-calibrated noise level.
            sigma = noise_sigma * (0.5 + torch.rand(len(ids), 1, 1, 1, device='cuda'))
            inp = inp + torch.randn_like(inp) * sigma
            if torch.rand(()) < 0.5:
                inp, out = inp.flip(-1), out.flip(-1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                pred = net(inp)
                loss = (pred.float() - out).square().mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sums += float(loss.detach()) * len(ids)
        net.eval()
        with torch.inference_mode():
            valp = predict(net, valid_x)
            m = (valp - valid_y).square().mean((1, 2, 3))
            mse = m.mean().item()
            psnr = (-10 * torch.log10(m.clamp_min(1e-15))).mean().item()
        record = dict(epoch=epoch+1, train_mse=sums/len(train_x), val_mse=mse, val_psnr_db=psnr, elapsed_seconds=time.time()-start)
        log.append(record)
        print(json.dumps(record), flush=True)
        (OUT / 'training_log.json').write_text(json.dumps(log, indent=2), encoding='utf8')
        if mse < best:
            best, best_epoch = mse, epoch+1
            torch.save(dict(state_dict=net.state_dict(), channels=32, depth=6, epoch=best_epoch, validation=record, noise_sigma=noise_sigma), OUT / 'best_cnn.pt')
            with torch.inference_mode():
                testp = predict(net, test_x).cpu().numpy()
            np.save(OUT / 'test_recovered.npy', testp[:, 0])
            save_grid(testp, OUT / 'Figure_test_neural.png')
        if epoch in [9, 19]:
            for group in opt.param_groups:
                group['lr'] *= 0.4
        if time.time() - train_start > args.seconds:
            break
    checkpoint = torch.load(OUT / 'best_cnn.pt', weights_only=False)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    with torch.inference_mode():
        hp = predict(net, torch.from_numpy(raw[ho]).cuda()).cpu().numpy()[:, 0]
        vp = predict(net, valid_x).cpu().numpy()[:, 0]
        tp = predict(net, test_x).cpu().numpy()[:, 0]
    np.save(OUT / 'holdout_recovered.npy', hp)
    np.save(OUT / 'validation_recovered.npy', vp)
    np.save(OUT / 'test_recovered.npy', tp)
    savemat(OUT / 'test_recovered.mat', dict(X_recovered=tp.transpose(1, 2, 0)))
    save_grid(tp, OUT / 'Figure_test_neural.png')
    for i, arr in enumerate(tp):
        Image.fromarray(np.rint(arr * 255).astype('uint8')).save(OUT / f'test_{i+1}_28x28.png')
        Image.fromarray(np.rint(arr * 255).astype('uint8')).resize((336, 336), Image.Resampling.NEAREST).save(OUT / f'test_{i+1}_display.png')
    report = dict(method='Frozen global ridge layer plus trained six-layer residual CNN', validation=metrics(vp, target[va, 0]), holdout=metrics(hp, target[ho, 0]), selected_epoch=best_epoch, diagnostic=diagnostic, seconds=time.time()-start, test_ground_truth_available=False)
    (OUT / 'final_info.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print('FINAL', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
