"""Select a sigmoid intensity shrinkage after ridge inversion."""

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from baselines.ridge_baseline import metrics  # noqa: E402
from prepare_data import grid  # noqa: E402


def predict(ids, model, measurements):
    chunks = []
    for start in range(0, len(ids), 500):
        batch = np.asarray(measurements[ids[start:start + 500]]).reshape(-1, 12544)
        energy = batch.sum(1)
        features = batch[:, model["features"]] / batch.mean(1)[:, None] - model["y_mean"]
        chunks.append(
            ((features @ model["weights"] + model["x_mean"]) * np.sqrt(energy)[:, None]).reshape(-1, 28, 28)
        )
    return np.concatenate(chunks)


def main():
    output = REPO_ROOT / "artifacts" / "baselines" / "ridge_shrinkage"
    output.mkdir(parents=True, exist_ok=True)
    processed = REPO_ROOT / "data" / "processed"
    x = np.load(processed / "X.npy", mmap_mode="r")
    y = np.load(processed / "Y.npy", mmap_mode="r")
    model = np.load(REPO_ROOT / "models" / "ridge12544" / "model.npz")
    split = np.load(REPO_ROOT / "models" / "ridge12544" / "split.npz")
    validation_prediction = predict(split["validation"], model, y)
    holdout_prediction = predict(split["holdout"], model, y)
    records = []
    best = (float("inf"), None)
    for threshold in [0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]:
        for softness in [0.005, 0.01, 0.02, 0.04]:
            transformed = np.clip(validation_prediction, 0, 1) / (
                1 + np.exp(np.clip(-(validation_prediction - threshold) / softness, -50, 50))
            )
            mse = float(np.mean((transformed - x[split["validation"]]) ** 2))
            records.append({"threshold": threshold, "softness": softness, "mse": mse})
            if mse < best[0]:
                best = (mse, (threshold, softness))

    threshold, softness = best[1]

    def transform(prediction):
        return np.clip(prediction, 0, 1) / (
            1 + np.exp(np.clip(-(prediction - threshold) / softness, -50, 50))
        )

    test_prediction = transform(np.load(REPO_ROOT / "models" / "ridge12544" / "test_recovered.npy"))
    np.save(output / "test_recovered.npy", test_prediction)
    grid(test_prediction, output / "test_preview.png", scale=12)
    report = {
        "method": "Validation-selected sigmoid intensity shrinkage after ridge",
        "parameters": {"threshold": threshold, "softness": softness},
        "validation": metrics(transform(validation_prediction), x[split["validation"]]),
        "holdout": metrics(transform(holdout_prediction), x[split["holdout"]]),
        "parameter_search": records,
    }
    (output / "final_info.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
