# Validation and limitations

The physical reconstruction was evaluated in two ways.

First, 256 images from a fixed holdout set were reconstructed with no participation in transmission calibration, detector selection, initialization or normalization. The final clipped output has MSE `1.25195e-14`, mean PSNR `139.86857 dB`, mean SSIM `0.999999999973503`, and maximum absolute error `1.78740e-6` against the supplied holdout images.

Second, each test reconstruction was evaluated on 512 detector measurements that were excluded from the image objective. Their relative MSEs are `1.35279e-12`, `7.62009e-13`, `6.25944e-13` and `7.59897e-13`. Replacing only those 512 test measurements by altered values before the full pipeline produced exactly unchanged output pixels, confirming that the withheld measurements do not influence the reconstruction.

The test file has no ground-truth `X`. Consequently, the test results establish consistency with the calibrated forward model, not direct image error against an unavailable reference. The unusually high holdout scores reflect the internally consistent synthetic/paired data and should not be extrapolated to real noisy scattering media without a separate external evaluation.
