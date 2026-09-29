"""Verify that withheld test measurements cannot change the reconstruction."""

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from recover import reconstruct

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MODEL = REPO_ROOT / "models" / "physical" / "inversion_model.npz"
TEST_FILE = REPO_ROOT / "data" / "raw" / "test.mat"
REPORT = REPO_ROOT / "results" / "test" / "measurement_independence.json"


def main():
    model = np.load(MODEL)
    blocked = model["pixels"][model["withheld_indices"]]
    query = loadmat(TEST_FILE)["Y"].transpose(2, 0, 1).astype(np.float32).reshape(4, -1)
    changed = query.copy()
    changed[:, blocked] = changed[:, blocked] * 7 + np.float32(0.02)
    original, original_diagnostics = reconstruct(query, model)
    altered, altered_diagnostics = reconstruct(changed, model)
    report = {
        "test": "Alter only all withheld intensities before the entire reconstruction pipeline",
        "withheld_detector_count": len(blocked),
        "maximum_output_change": float(np.max(np.abs(original - altered))),
        "exactly_equal_outputs": bool(np.array_equal(original, altered)),
        "original_fit_residuals": original_diagnostics["fit_relative_mse"],
        "changed_fit_residuals": altered_diagnostics["fit_relative_mse"],
        "original_withheld_residuals": original_diagnostics["withheld_relative_mse"],
        "changed_withheld_residuals": altered_diagnostics["withheld_relative_mse"],
        "original_data_modified": False,
    }
    if not report["exactly_equal_outputs"]:
        raise AssertionError(report)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
