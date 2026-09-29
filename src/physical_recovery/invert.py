"""Evaluate the calibrated model on a holdout set and reconstruct test data."""

import argparse
import json
import os
import time
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "4"

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.io import savemat
from skimage.metrics import structural_similarity

from recover import reconstruct

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
CALIBRATION = REPO_ROOT / "artifacts" / "physical"
MODEL_DIR = REPO_ROOT / "models" / "physical"
RIDGE_MODEL = REPO_ROOT / "models" / "ridge12544" / "model.npz"
TEST_RESULTS = REPO_ROOT / "results" / "test"
HOLDOUT_RESULTS = REPO_ROOT / "results" / "holdout"


def metrics(prediction, truth):
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    errors = (prediction - truth) ** 2
    return {
        "n": len(prediction),
        "mse": float(errors.mean()),
        "psnr_mean_db": float(np.mean(-10 * np.log10(np.maximum(errors.mean((1, 2)), 1e-20)))),
        "ssim_mean": float(
            np.mean([structural_similarity(a, b, data_range=1) for a, b in zip(prediction, truth)])
        ),
        "maximum_absolute_error": float(np.abs(prediction - truth).max()),
    }


def grid(images, path, labels):
    images = np.asarray(images).reshape(-1, 28, 28)
    canvas = Image.new("RGB", (len(images) * 300, 342), "#f7f7f7")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(images):
        preview = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).resize(
            (280, 280), Image.Resampling.NEAREST
        )
        canvas.paste(preview, (index * 300 + 10, 48))
        draw.text((index * 300 + 10, 15), labels[index], fill="#222222")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=1e-11)
    parser.add_argument("--holdout", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    CALIBRATION.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_RESULTS.mkdir(parents=True, exist_ok=True)
    HOLDOUT_RESULTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.time()

    if args.partial:
        parts = [np.load(path) for path in sorted(CALIBRATION.glob("batch_*.npz"))]
        matrix = np.concatenate([part["matrix"] for part in parts], 1)
        pixels = np.concatenate([part["pixels"] for part in parts])
        errors = np.concatenate([part["validation_relative_mse"] for part in parts])
    else:
        saved = np.load(CALIBRATION / "transmission_matrix.npz")
        matrix, pixels, errors = saved["matrix"], saved["pixels"], saved["validation_relative_mse"]

    good = errors < args.threshold
    selected_pixels = pixels[good]
    selected_matrix = matrix[:, np.flatnonzero(np.repeat(good, 2))]
    if len(selected_pixels) < 500:
        raise ValueError("Too few calibrated detector channels meet the threshold")

    withheld = np.sort(np.random.default_rng(3219).choice(len(selected_pixels), 512, replace=False))
    fit = np.setdiff1d(np.arange(len(selected_pixels)), withheld)
    model_path = MODEL_DIR / "inversion_model.npz"
    np.savez(
        model_path,
        matrix=selected_matrix,
        pixels=selected_pixels,
        fit_indices=fit,
        withheld_indices=withheld,
        calibration_threshold=args.threshold,
    )

    split = np.load(REPO_ROOT / "models" / "ridge12544" / "split.npz")
    holdout_ids = split["holdout"][: args.holdout]
    x = np.load(PROCESSED / "X.npy", mmap_mode="r")
    y = np.load(PROCESSED / "Y.npy", mmap_mode="r").reshape(50000, -1)
    test = np.load(PROCESSED / "test_Y.npy").reshape(4, -1)
    query = np.concatenate([np.asarray(y[holdout_ids]), test])
    energy = query.sum(1)
    normalized_query = query / energy[:, None] * 12544
    ridge = np.load(RIDGE_MODEL)
    initial = np.clip(
        ((normalized_query[:, ridge["features"]] - ridge["y_mean"]) @ ridge["weights"] + ridge["x_mean"])
        * np.sqrt(energy[:, None]),
        0,
        1,
    )
    output, diagnostics = reconstruct(query, np.load(model_path), None, args.steps, RIDGE_MODEL)
    test_output = output[-4:]
    holdout_output = output[:-4]
    truth = np.asarray(x[holdout_ids])
    initial_images = initial[:-4].reshape(-1, 28, 28)
    report = {
        "method": "calibrated complex transmission matrix plus amplitude least squares",
        "calibrated_detector_count": int(len(selected_pixels)),
        "calibration_error_threshold": args.threshold,
        "fit_detector_count": len(fit),
        "withheld_detector_count": len(withheld),
        "holdout_ridge": metrics(initial_images, truth),
        "ridge_baseline_note": "Uses all measurement pixels for baseline comparison; not passed to physical reconstruction as initialization.",
        "holdout_physics": metrics(holdout_output, truth),
        "holdout_fit_measurement_relative_mse_quantiles": np.quantile(
            diagnostics["fit_relative_mse"][:-4], [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "holdout_withheld_measurement_relative_mse_quantiles": np.quantile(
            diagnostics["withheld_relative_mse"][:-4], [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "test_fit_measurement_relative_mse": diagnostics["fit_relative_mse"][-4:],
        "test_withheld_measurement_relative_mse": diagnostics["withheld_relative_mse"][-4:],
        "elapsed_seconds": time.time() - start,
        "test_ground_truth_available": False,
        "residuals_evaluated_after_final_clipping": True,
        "withheld_measurements_excluded_from_initialization_and_normalization": diagnostics[
            "withheld_measurements_excluded_from_initialization_and_normalization"
        ],
        "normalization": diagnostics["normalization"],
        "initialization": diagnostics["initialization"],
    }
    np.save(TEST_RESULTS / "test_recovered.npy", test_output)
    savemat(TEST_RESULTS / "test_recovered.mat", {"X_recovered": test_output.transpose(1, 2, 0)})
    np.save(HOLDOUT_RESULTS / "holdout_recovered.npy", holdout_output)
    np.save(HOLDOUT_RESULTS / "holdout_indices.npy", holdout_ids)
    grid(test_output, TEST_RESULTS / "test_recovered.png", [f"Test {i + 1}" for i in range(4)])
    grid(truth[:4], HOLDOUT_RESULTS / "holdout_truth.png", [f"Ground truth {i + 1}" for i in range(4)])
    grid(holdout_output[:4], HOLDOUT_RESULTS / "holdout_recovered.png", [f"Physics recovery {i + 1}" for i in range(4)])
    (CALIBRATION / "inversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
