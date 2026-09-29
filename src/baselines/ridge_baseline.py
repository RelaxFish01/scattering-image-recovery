"""Normalized global ridge inverse used as a reproducible baseline."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.linalg import cho_factor, cho_solve
from skimage.metrics import structural_similarity

SRC = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from prepare_data import grid  # noqa: E402


def metrics(prediction, target):
    mse = ((prediction - target) ** 2).mean((1, 2))
    psnr = -10 * np.log10(np.maximum(mse, 1e-15))
    ssim = [structural_similarity(t, p, data_range=1) for t, p in zip(target, prediction)]
    return {
        "mse": float(mse.mean()),
        "psnr_mean_db": float(psnr.mean()),
        "ssim_mean": float(np.mean(ssim)),
        "count": len(target),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "models" / "ridge12544")
    parser.add_argument("--features", type=int, default=4096)
    parser.add_argument("--transform", choices=["intensity", "sqrt"], default="intensity")
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    processed = REPO_ROOT / "data" / "processed"
    start = time.time()

    x = np.load(processed / "X.npy", mmap_mode="r")
    y = np.load(processed / "Y.npy", mmap_mode="r")
    test = np.load(processed / "test_Y.npy")
    rng = np.random.default_rng(20260929)
    ids = rng.permutation(50000)
    train, validation, holdout = ids[:40000], ids[40000:45000], ids[45000:]
    np.savez(out / "split.npz", train=train, validation=validation, holdout=holdout)

    features = np.sort(rng.choice(12544, args.features, replace=False))
    measurements = np.empty((50000, args.features), np.float32)
    energy = np.empty(50000, np.float32)
    for start_index in range(0, 50000, 1000):
        batch = np.asarray(y[start_index:start_index + 1000]).reshape(-1, 12544)
        energy[start_index:start_index + len(batch)] = batch.sum(1)
        normalized = batch[:, features] / batch.mean(1)[:, None]
        measurements[start_index:start_index + len(batch)] = (
            np.sqrt(normalized) if args.transform == "sqrt" else normalized
        )
    test_flat = test.reshape(4, -1)
    test_measurements = test_flat[:, features] / test_flat.mean(1)[:, None]
    test_measurements = np.sqrt(test_measurements) if args.transform == "sqrt" else test_measurements

    image_norm = np.linalg.norm(x[:50000].reshape(50000, -1), axis=1)
    target = x[:50000].reshape(50000, -1) / image_norm[:, None]
    measurement_mean = measurements[train].mean(0)
    image_mean = target[train].mean(0)
    measurements -= measurement_mean
    test_measurements -= measurement_mean
    gram = measurements[train].T @ measurements[train]
    rhs = measurements[train].T @ (target[train] - image_mean)

    records = []
    best = None
    for regularization in [1.0, 10.0, 100.0]:
        matrix = gram.copy()
        matrix.flat[::args.features + 1] += regularization
        weights = cho_solve(
            cho_factor(matrix, lower=True, overwrite_a=True, check_finite=False),
            rhs,
            check_finite=False,
        )
        prediction = np.clip(
            (measurements[validation] @ weights + image_mean) * np.sqrt(energy[validation])[:, None], 0, 1
        ).reshape(-1, 28, 28)
        record = metrics(prediction, x[validation])
        record["regularization"] = regularization
        records.append(record)
        if best is None or record["mse"] < best[0]:
            best = (record["mse"], regularization, weights.copy())

    _, regularization, weights = best
    holdout_prediction = np.clip(
        (measurements[holdout] @ weights + image_mean) * np.sqrt(energy[holdout])[:, None], 0, 1
    ).reshape(-1, 28, 28)
    holdout_metrics = metrics(holdout_prediction, x[holdout])
    test_prediction = np.clip(
        (test_measurements @ weights + image_mean) * np.sqrt(test_flat.sum(1))[:, None], 0, 1
    ).reshape(4, 28, 28)
    np.save(out / "test_recovered.npy", test_prediction)
    savemat(out / "test_recovered.mat", {"X_recovered": test_prediction.transpose(1, 2, 0)})
    np.savez(
        out / "model.npz",
        weights=weights,
        features=features,
        y_mean=measurement_mean,
        x_mean=image_mean,
        regularization=regularization,
        transform=args.transform,
    )
    grid(test_prediction, out / "test_preview.png", scale=12)
    grid(np.concatenate([x[holdout[:8]], holdout_prediction[:8]]), out / "holdout_preview.png", scale=5)
    report = {
        "method": "Normalized global ridge inversion",
        "features": args.features,
        "transform": args.transform,
        "validation": records,
        "holdout": holdout_metrics,
        "selected_regularization": regularization,
        "test_ground_truth_available": False,
        "seconds": time.time() - start,
    }
    (out / "final_info.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
