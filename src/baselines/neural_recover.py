r"""Recover measurements using the saved ridge + residual CNN.

Example from the project root:
  .recovery-venv\Scripts\python.exe test恢复研究\neural_recovery\recover.py
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from neural_train import ResidualCNN, OUT, REPO_ROOT, save_grid


def recover(y, device='cpu'):
    y = np.asarray(y, np.float32).reshape(-1, 12544)
    model = np.load(REPO_ROOT / 'models' / 'ridge12544' / 'model.npz')
    checkpoint = torch.load(OUT / 'best_cnn.pt', map_location=device, weights_only=False)
    cnn = ResidualCNN(checkpoint['channels'], checkpoint['depth']).to(device)
    cnn.load_state_dict(checkpoint['state_dict'])
    cnn.eval()
    w = torch.from_numpy(model['weights']).to(device)
    ym = torch.from_numpy(model['y_mean']).to(device)
    xm = torch.from_numpy(model['x_mean']).to(device)
    results = []
    with torch.inference_mode():
        for lo in range(0, len(y), 512):
            measurement = torch.tensor(y[lo:lo+512], device=device)
            energy = measurement.sum(1, keepdim=True).sqrt()
            features = measurement[:, model['features']] / measurement.mean(1, keepdim=True) - ym
            raw = ((features @ w + xm) * energy).reshape(-1, 1, 28, 28)
            results.append(cnn(raw).clamp(0, 1)[:, 0].cpu().numpy())
    return np.concatenate(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=REPO_ROOT / 'data' / 'processed' / 'test_Y.npy')
    parser.add_argument('--output', type=Path, default=OUT / 'test_recovered_reproduced.npy')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.set_num_threads(6)
    result = recover(np.load(args.input), args.device)
    np.save(args.output, result)
    if len(result) <= 12:
        save_grid(result, args.output.with_suffix('.png'))
    print(f'Saved {len(result)} restored 28 x 28 images to {args.output}')


if __name__ == '__main__':
    main()
