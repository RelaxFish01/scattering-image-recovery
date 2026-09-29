"""Calibrate a complex transmission matrix from paired measurements."""

import argparse
import json
import os
import time
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "4"

import numpy as np
import torch
from scipy.linalg import eigh

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "artifacts" / "physical"
SPLIT = REPO_ROOT / "models" / "ridge12544" / "split.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detectors", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--train", type=int, default=10000)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--adam", type=int, default=1000)
    parser.add_argument("--lbfgs", type=int, default=800)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    torch.manual_seed(74031)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    rng = np.random.default_rng(74031)
    split = np.load(SPLIT)
    train = split["train"][: args.train]
    validation = split["validation"][: args.validation]
    x = np.load(PROCESSED / "X.npy", mmap_mode="r").reshape(52000, -1)
    y = np.load(PROCESSED / "Y.npy", mmap_mode="r").reshape(50000, -1)
    pixels = np.sort(rng.choice(12544, args.detectors, replace=False))
    sample_ids = np.r_[train, validation]
    images = np.array(x[sample_ids])
    energy = (images * images).sum(1)
    images /= np.sqrt(energy[:, None])
    measurements = np.empty((len(sample_ids), len(pixels)), np.float32)
    for start_index in range(0, len(sample_ids), 1000):
        end_index = min(start_index + 1000, len(sample_ids))
        measurements[start_index:end_index] = (
            np.asarray(y[sample_ids[start_index:end_index]])[:, pixels]
            * 12544
            / energy[start_index:end_index, None]
        )

    eigenvalues, eigenvectors = eigh(
        images[: len(train)].T @ images[: len(train)] / len(train), check_finite=False
    )
    keep = eigenvalues > max(eigenvalues) * 1e-8
    projection = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    transformed = torch.tensor(images @ projection, device=device)
    dimension = transformed.shape[1]
    np.savez(
        OUT / "design.npz",
        pixels=pixels,
        train_indices=train,
        validation_indices=validation,
        whitening=projection,
        input_eigenvalues=eigenvalues,
    )

    matrix_all = np.zeros((784, 2 * len(pixels)), np.float32)
    error_all = np.full(len(pixels), np.inf, np.float32)
    reports = []
    start = time.time()
    for start_index in range(0, len(pixels), args.batch):
        end_index = min(start_index + args.batch, len(pixels))
        batch_size = end_index - start_index
        part = OUT / f"batch_{start_index:04d}.npz"
        if args.resume and part.exists():
            old = np.load(part)
            matrix_all[:, 2 * start_index:2 * end_index] = old["matrix"]
            error_all[start_index:end_index] = old["validation_relative_mse"]
            continue

        target = torch.tensor(measurements[:, start_index:end_index], device=device)
        weights = torch.nn.Parameter(torch.randn(dimension, batch_size * 2, device=device) / np.sqrt(2 * dimension))
        optimizer = torch.optim.Adam([weights], lr=0.02)
        batch_start = time.time()
        for step in range(args.adam):
            optimizer.zero_grad(set_to_none=True)
            prediction = (transformed[: len(train)] @ weights).reshape(-1, batch_size, 2).square().sum(2)
            loss = (prediction - target[: len(train)]).square().mean()
            loss.backward()
            optimizer.step()
            if step in [350, 650, 850]:
                for group in optimizer.param_groups:
                    group["lr"] *= 0.4

        optimizer = torch.optim.LBFGS(
            [weights],
            lr=0.8,
            max_iter=args.lbfgs,
            history_size=15,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-13,
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            prediction = (transformed[: len(train)] @ weights).reshape(-1, batch_size, 2).square().sum(2)
            loss = (prediction - target[: len(train)]).square().mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            validation_prediction = (transformed[len(train):] @ weights).reshape(-1, batch_size, 2).square().sum(2)
            validation_error = (
                (validation_prediction - target[len(train):]).square().mean(0)
                / target[len(train):].square().mean(0)
            )
            training_prediction = (transformed[: len(train)] @ weights).reshape(-1, batch_size, 2).square().sum(2)
            training_error = (
                (training_prediction - target[: len(train)]).square().mean(0)
                / target[: len(train)].square().mean(0)
            )

        matrix = projection @ weights.detach().cpu().numpy()
        validation_error = validation_error.cpu().numpy()
        training_error = training_error.cpu().numpy()
        np.savez(
            part,
            matrix=matrix,
            pixels=pixels[start_index:end_index],
            validation_relative_mse=validation_error,
            training_relative_mse=training_error,
        )
        matrix_all[:, 2 * start_index:2 * end_index] = matrix
        error_all[start_index:end_index] = validation_error
        record = {
            "detector_start": start_index,
            "detector_end": end_index,
            "seconds": time.time() - batch_start,
            "validation_error_quantiles": np.quantile(validation_error, [0, 0.25, 0.5, 0.75, 0.9, 1]).tolist(),
            "good_1e-8": int((validation_error < 1e-8).sum()),
            "good_1e-6": int((validation_error < 1e-6).sum()),
            "elapsed": time.time() - start,
        }
        reports.append(record)
        (OUT / "calibration_progress.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(json.dumps(record))

    np.savez(OUT / "transmission_matrix.npz", matrix=matrix_all, pixels=pixels, validation_relative_mse=error_all)
    report = {
        "device": device,
        "detectors": len(pixels),
        "train": len(train),
        "validation": len(validation),
        "input_dimensions": int(dimension),
        "good_1e-8": int((error_all < 1e-8).sum()),
        "good_1e-6": int((error_all < 1e-6).sum()),
        "seconds": time.time() - start,
        "batches": reports,
    }
    (OUT / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
