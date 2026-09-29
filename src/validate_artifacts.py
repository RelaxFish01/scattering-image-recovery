"""Validate numeric exports and their MATLAB round trip."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", type=Path, required=True)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prediction = np.load(args.npy, allow_pickle=False)
    round_trip = loadmat(args.mat)["X_recovered"].transpose(2, 0, 1)
    report = {
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "finite": bool(np.isfinite(prediction).all()),
        "range": [float(prediction.min()), float(prediction.max())],
        "npy_mat_exact": bool(np.array_equal(prediction, round_trip)),
        "npy": str(args.npy.resolve()),
        "mat": str(args.mat.resolve()),
    }
    if not report["finite"] or not report["npy_mat_exact"]:
        raise AssertionError(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
