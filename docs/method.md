# Method

## Forward model

For a normalized image vector `x ∈ R^784`, each selected detector is modeled as a rank-two positive semidefinite quadratic form:

```text
12544 y_j = (xᵀ a_j)^2 + (xᵀ b_j)^2.
```

The two real vectors `a_j` and `b_j` are the real and imaginary parts of a complex transmission coefficient. The model is fitted detector-wise from paired training data after image-energy normalization.

## Calibration and inversion

1. Use a fixed seed (`20260929`) to split the first 50,000 paired samples into 40,000 training, 5,000 validation and 5,000 holdout samples.
2. Fit 4,096 randomly selected detector channels from 10,000 training pairs. Select channels with validation relative MSE below `1e-11`; 3,439 channels satisfy the threshold.
3. Reserve 512 of those channels for an independent measurement check. Only the remaining 2,927 channels enter the image objective.
4. Construct the initial image with ridge regression after replacing the reserved measurements by the mean of the non-reserved measurements. The same non-reserved measurements define the normalization energy.
5. Optimize the image against the fitted detector amplitudes with L-BFGS and clip the delivered image to `[0,1]`.

No digit labels, image retrieval, hand-drawn corrections, TV penalty or generative image prior is used in the final reconstruction.

## Baselines

The repository includes a normalized global ridge inverse and a residual CNN postprocessor as reference methods. Their results are evaluated on the same fixed split; model selection uses the validation set only, while the holdout set is evaluated after selection.
