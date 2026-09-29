# Data

`raw/test.mat` is the supplied test measurement file. It contains the MATLAB variable `Y` with shape `112×112×4` and does not contain the corresponding image variable `X`.

The supplied `Train.mat` is intentionally not committed: its size is approximately 4.8 GB. Place it at `data/raw/Train.mat` before running `src/prepare_data.py`. The preparation script reads the HDF5 datasets `X` and `Y`, canonicalizes them to `N×H×W`, and writes local float32 caches under `data/processed/`.

The repository records the observed sample counts, array orientation and audit results in `artifacts/audits/data_report.json`. Raw data redistribution remains subject to the course's data-use terms.
