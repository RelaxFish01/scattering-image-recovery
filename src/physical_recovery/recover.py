"""Standalone reconstruction using the saved calibrated physical model."""

import argparse
import json
import os
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "4"

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.io import loadmat, savemat

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "physical" / "inversion_model.npz"
DEFAULT_RIDGE = REPO_ROOT / "models" / "ridge12544" / "model.npz"


def ridge_initial(query, ridge_model_path=DEFAULT_RIDGE):
    model = np.load(ridge_model_path)
    energy = query.sum(1)
    features = query[:, model["features"]] / energy[:, None] * 12544
    return np.clip(
        ((features - model["y_mean"]) @ model["weights"] + model["x_mean"])
        * np.sqrt(energy[:, None]),
        0,
        1,
    )


def reconstruct(query, model, initial=None, steps=500, ridge_model_path=DEFAULT_RIDGE):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query = np.asarray(query, dtype=np.float32).reshape(-1, 12544)
    fit = model["fit_indices"]
    withheld = model["withheld_indices"]
    pixels = model["pixels"]
    matrix = model["matrix"]

    blocked_pixels = pixels[withheld]
    available = np.ones(12544, dtype=bool)
    available[blocked_pixels] = False
    available_mean = query[:, available].mean(1)
    energy = available_mean * 12544
    independent_initialization = initial is None
    if initial is None:
        initialization_query = query.copy()
        initialization_query[:, blocked_pixels] = available_mean[:, None]
        initial = ridge_initial(initialization_query, ridge_model_path)
    initial = np.asarray(initial, dtype=np.float32).reshape(-1, 784)

    columns = np.flatnonzero(np.repeat(np.isin(np.arange(len(pixels)), fit), 2))
    transmission = torch.tensor(matrix[:, columns], device=device)
    target = torch.tensor(query[:, pixels[fit]] / energy[:, None] * 12544, device=device)
    image = torch.nn.Parameter(torch.tensor(initial / np.sqrt(energy[:, None]), device=device))
    optimizer = torch.optim.LBFGS(
        [image],
        lr=0.8,
        max_iter=steps,
        history_size=15,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-11,
        tolerance_change=1e-15,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        prediction = (image @ transmission).reshape(-1, len(fit), 2).square().sum(2)
        loss = (torch.sqrt(prediction + 1e-12) - torch.sqrt(target + 1e-12)).square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    raw = image.detach().cpu().numpy() * np.sqrt(energy[:, None])
    output = np.clip(raw, 0, 1)

    normalized = output / np.sqrt(energy[:, None])
    prediction = ((normalized @ matrix).reshape(-1, len(pixels), 2) ** 2).sum(2)
    expected = query[:, pixels] / energy[:, None] * 12544

    def score(indices):
        return (
            ((prediction[:, indices] - expected[:, indices]) ** 2).mean(1)
            / (expected[:, indices] ** 2).mean(1)
        ).tolist()

    return output.reshape(-1, 28, 28), {
        "fit_detector_count": len(fit),
        "withheld_detector_count": len(withheld),
        "fit_relative_mse": score(fit),
        "withheld_relative_mse": score(withheld),
        "raw_minimum": float(raw.min()),
        "raw_maximum": float(raw.max()),
        "final_output_clipped_to": [0, 1],
        "device": device,
        "withheld_measurements_excluded_from_initialization_and_normalization": independent_initialization,
        "normalization": "mean of all non-withheld detector measurements, multiplied by 12544",
        "initialization": (
            "withheld measurements replaced by non-withheld measurement mean before ridge initialization"
            if independent_initialization
            else "externally provided; independence from withheld measurements is not verified"
        ),
    }


def load_measurements(path):
    path = Path(path)
    if path.suffix.lower() == ".mat":
        return loadmat(path)["Y"].transpose(2, 0, 1)
    return np.load(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ridge-model", type=Path, default=DEFAULT_RIDGE)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    measurements = load_measurements(args.input)
    model = np.load(args.model)
    initial = np.load(args.initial) if args.initial else None
    result, diagnostics = reconstruct(
        measurements, model, initial=initial, steps=args.steps, ridge_model_path=args.ridge_model
    )

    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination.with_suffix(".npy"), result)
    savemat(destination.with_suffix(".mat"), {"X_recovered": result.transpose(1, 2, 0)})
    canvas = Image.new("RGB", (300 * len(result), 332), "#f7f7f7")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(result):
        preview = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).resize(
            (280, 280), Image.Resampling.NEAREST
        )
        canvas.paste(preview, (300 * index + 10, 40))
        draw.text((300 * index + 10, 12), f"Test {index + 1}", fill="black")
    canvas.save(destination.with_suffix(".png"))
    destination.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
