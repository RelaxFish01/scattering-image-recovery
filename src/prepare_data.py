"""Convert the supplied MATLAB files into canonical local NumPy caches."""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "raw"
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"


def grid(arr, path, labels=None, scale=5):
    arr = np.asarray(arr)
    n, height, width = arr.shape
    canvas = Image.new("RGB", (n * (width * scale + 16), height * scale + 46), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(arr):
        preview = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).resize(
            (width * scale, height * scale), Image.Resampling.NEAREST
        )
        left = index * (width * scale + 16) + 8
        canvas.paste(preview, (left, 32))
        draw.text((left, 8), str(labels[index] if labels is not None else index + 1), fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_RAW / "Train.mat")
    parser.add_argument("--test-file", type=Path, default=DEFAULT_RAW / "test.mat")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    with h5py.File(args.train_file, "r") as source:
        x = np.asarray(source["X"], dtype=np.float32).transpose(0, 2, 1).copy()
        np.save(args.output_dir / "X.npy", x)
        test = loadmat(args.test_file)["Y"].transpose(2, 0, 1).astype(np.float32)
        np.save(args.output_dir / "test_Y.npy", test)

        y_count = source["Y"].shape[0]
        y = np.lib.format.open_memmap(
            args.output_dir / "Y.npy", mode="w+", dtype="float32", shape=(y_count, 112, 112)
        )
        nearest_distance = np.full((len(test), 10), np.inf, dtype=np.float32)
        nearest_index = np.zeros((len(test), 10), dtype=int)
        test_flat = test.reshape(len(test), -1)
        test_norm = np.linalg.norm(test_flat, axis=1)
        means = []
        for start_index in range(0, y_count, 500):
            end_index = min(start_index + 500, y_count)
            batch = np.asarray(source["Y"][start_index:end_index], dtype=np.float32).transpose(0, 2, 1).copy()
            y[start_index:end_index] = batch
            flat = batch.reshape(len(batch), -1)
            norm = np.linalg.norm(flat, axis=1)
            correlation = (flat @ test_flat.T) / (norm[:, None] * test_norm[None, :])
            distance = 1 - correlation
            for test_index in range(len(test)):
                values = np.r_[nearest_distance[test_index], distance[:, test_index]]
                indices = np.r_[nearest_index[test_index], np.arange(start_index, end_index)]
                order = np.argsort(values)[:10]
                nearest_distance[test_index] = values[order]
                nearest_index[test_index] = indices[order]
            means.extend(batch.mean((1, 2)).tolist())
        y.flush()

    means = np.asarray(means)
    report = {
        "x_count": int(len(x)),
        "y_count": int(y_count),
        "test_count": int(len(test)),
        "x_shape": list(x.shape),
        "y_shape": [int(y_count), 112, 112],
        "test_shape": list(test.shape),
        "nearest_test_cosine_distances": nearest_distance.tolist(),
        "nearest_test_train_indices": nearest_index.tolist(),
        "mean_intensity_corr_x": float(np.corrcoef(x[:y_count].mean((1, 2)), means)[0, 1]),
        "mean_intensity_corr_x_squared": float(np.corrcoef((x[:y_count] ** 2).mean((1, 2)), means)[0, 1]),
        "orientation": "Canonical arrays are N,H,W; HDF5 samples are transposed on read.",
        "test_ground_truth_in_test_file": False,
        "elapsed_seconds": time.time() - start,
    }
    (args.output_dir / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for test_index in range(len(test)):
        grid(
            x[nearest_index[test_index, :6]],
            args.output_dir / f"nearest_test_{test_index + 1}.png",
            nearest_index[test_index, :6],
            scale=6,
        )
    normalized_test = np.stack([np.clip(image / np.quantile(image, 0.995), 0, 1) for image in test])
    grid(normalized_test, args.output_dir / "test_speckle.png", scale=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
