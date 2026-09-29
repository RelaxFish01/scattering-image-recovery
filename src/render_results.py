"""Render the four recovered test images without changing their numeric values.

Usage: python render_results.py --pred predictions.npy --out output_directory
The prediction input must have shape (4, 28, 28). Test speckle data are read
from test_Y.npy next to this script. Only preview PNGs are clipped to [0, 1].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat, savemat


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/arialbd.ttf"]
        if bold
        else ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf"]
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default(size=size)


def gray_image(array: np.ndarray) -> Image.Image:
    """Use one fixed intensity range for every recovery image."""
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(pixels)


def center_text(
    draw: ImageDraw.ImageDraw,
    x_center: float,
    y: float,
    value: str,
    font: ImageFont.ImageFont,
    fill: str = "#222222",
) -> None:
    box = draw.textbbox((0, 0), value, font=font)
    draw.text((x_center - (box[2] - box[0]) / 2.0, y), value, font=font, fill=fill)


def render(pred_path: Path, speckle_path: Path, out_dir: Path) -> dict:
    predictions = np.load(pred_path, allow_pickle=False)
    if not isinstance(predictions, np.ndarray) or predictions.shape != (4, 28, 28):
        raise ValueError("--pred must be a NumPy .npy array with shape (4, 28, 28).")
    if not np.issubdtype(predictions.dtype, np.number) or np.iscomplexobj(predictions):
        raise ValueError("Predictions must contain real numeric intensities.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Predictions contain NaN or infinite values.")

    if speckle_path.suffix.lower() == ".mat":
        speckles = loadmat(speckle_path)["Y"].transpose(2, 0, 1)
    else:
        speckles = np.load(speckle_path, allow_pickle=False)
    if speckles.shape != (4, 112, 112) or not np.all(np.isfinite(speckles)):
        raise ValueError("test_Y.npy must contain four finite 112 by 112 speckle images.")

    out_dir.mkdir(parents=True, exist_ok=True)
    native_files: list[str] = []
    enlarged_files: list[str] = []
    recovered_previews: list[Image.Image] = []
    speckle_previews: list[Image.Image] = []
    speckle_scales: list[float] = []

    for index, prediction in enumerate(predictions, start=1):
        native = gray_image(prediction)
        enlarged = native.resize((448, 448), Image.Resampling.NEAREST)
        native_name = f"test_{index:02d}_recovered_28px.png"
        enlarged_name = f"test_{index:02d}_recovered_448px.png"
        native.save(out_dir / native_name)
        enlarged.save(out_dir / enlarged_name)
        native_files.append(native_name)
        enlarged_files.append(enlarged_name)
        recovered_previews.append(enlarged)

        display_scale = float(np.percentile(speckles[index - 1], 99.5))
        if display_scale <= 0.0:
            display_scale = 1.0
        speckle_scales.append(display_scale)
        speckle_preview = gray_image(speckles[index - 1] / display_scale)
        speckle_previews.append(speckle_preview.resize((448, 448), Image.Resampling.NEAREST))

    # Save the original floating point values, including their original dtype.
    # MATLAB uses H x W x N; NumPy uses N x H x W here.
    np.save(out_dir / "X_recovered.npy", predictions, allow_pickle=False)
    savemat(
        out_dir / "X_recovered.mat",
        {"X_recovered": predictions.transpose(1, 2, 0)},
        do_compression=True,
    )

    margin = 40
    tile = 448
    gap = 32
    width = 2 * margin + 4 * tile + 3 * gap
    lefts = [margin + index * (tile + gap) for index in range(4)]
    title_font = load_font(32, bold=True)
    label_font = load_font(25, bold=True)
    body_font = load_font(19)
    small_font = load_font(17)

    overview = Image.new("RGB", (width, 610), "white")
    draw = ImageDraw.Draw(overview)
    draw.text((margin, 22), "Recovered test images", font=title_font, fill="#151515")
    draw.text(
        (margin, 67),
        "Model estimates | native resolution 28 x 28 | common display range [0, 1]",
        font=body_font,
        fill="#555555",
    )
    for index, (left, preview) in enumerate(zip(lefts, recovered_previews), start=1):
        center_text(draw, left + tile / 2, 100, f"Test {index}", label_font)
        overview.paste(preview, (left, 140))
    overview.save(out_dir / "recovered_overview.png")

    comparison = Image.new("RGB", (width, 1192), "white")
    draw = ImageDraw.Draw(comparison)
    draw.text((margin, 22), "Test speckles and recovered images", font=title_font, fill="#151515")
    draw.text(
        (margin, 67),
        "Reconstructed image estimates; the test file does not include ground truth images.",
        font=body_font,
        fill="#555555",
    )
    for index, left in enumerate(lefts, start=1):
        center_text(draw, left + tile / 2, 103, f"Test {index}", label_font)
    draw.text((margin, 145), "Input speckle (112 x 112)", font=body_font, fill="#333333")
    for left, preview in zip(lefts, speckle_previews):
        comparison.paste(preview, (left, 181))
    draw.text(
        (margin, 639),
        "Speckle display scaling: divide each image by its own 99.5th percentile, then clip to [0, 1].",
        font=small_font,
        fill="#666666",
    )
    draw.text(
        (margin, 684),
        "Recovered image (28 x 28; nearest-neighbor enlargement)",
        font=body_font,
        fill="#333333",
    )
    for left, preview in zip(lefts, recovered_previews):
        comparison.paste(preview, (left, 721))
    comparison.save(out_dir / "speckle_vs_recovered.png")

    manifest = {
        "prediction_source": str(pred_path.resolve()),
        "speckle_source": str(speckle_path),
        "prediction_shape": list(predictions.shape),
        "prediction_dtype": str(predictions.dtype),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "numeric_values_preserved": True,
        "numpy_layout": "N,H,W; (4,28,28)",
        "matlab_variable": "X_recovered",
        "matlab_layout": "H,W,N; (28,28,4)",
        "recovery_png_display": "Clip to [0,1], multiply by 255, round to uint8; same range for all images.",
        "speckle_png_display": "Per-image 99.5th percentile display scaling; for visualization only.",
        "speckle_display_scale_values": speckle_scales,
        "native_pngs": native_files,
        "enlarged_pngs": enlarged_files,
        "overview": "recovered_overview.png",
        "comparison": "speckle_vs_recovered.png",
        "numeric_outputs": ["X_recovered.npy", "X_recovered.mat"],
    }
    (out_dir / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="Prediction .npy file with shape (4,28,28).")
    parser.add_argument("--speckle", type=Path, required=True, help="Test measurements with shape (4,112,112).")
    parser.add_argument("--out", type=Path, required=True, help="Directory for numeric results and PNG previews.")
    args = parser.parse_args()
    result = render(args.pred, args.speckle, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
