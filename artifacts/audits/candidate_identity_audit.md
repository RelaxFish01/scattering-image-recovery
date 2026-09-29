# Candidate identity audit

Used 1,024 randomly selected training pairs as independent reference fingerprints. For each query, measured normalized centered intensity covariances were compared with the squared normalized image inner products predicted by the complex linear scattering model. All 52,000 supplied X images were searched. Indices below are zero based.

| Test | Best correlation candidate | Best correlation | Second correlation | Best direct fingerprint MSE | Second MSE |
|---|---:|---:|---:|---:|---:|
| 1 | 8265 | 0.878358 | 0.819186 | 0.001001659 | 0.001160330 |
| 2 | 42334 | 0.795143 | 0.787795 | 0.001083949 | 0.001111673 |
| 3 | 18847 | 0.870139 | 0.823823 | 0.000891271 | 0.000928306 |
| 4 | 31236 | 0.854263 | 0.822962 | 0.001058926 | 0.001091522 |

As a positive control, all 20 known training queries excluded from the reference set recovered their exact X at rank 1. Their true-match correlation range was 0.978961–0.996102 and true-match MSE range 0.000072427–0.000127603. Thus each test query's smallest residual was at least 6.9 times the worst positive-control true-match residual. No test image has a comparable true-match signature. The extra 2,000 X images performed still worse: maximum correlations were 0.738781, 0.729342, 0.685426, and 0.785881.

Conclusion: this finite reference experiment found no evidence that any of the supplied 52,000 X images is the exact hidden test original. It supports treating the closest candidates as similar handwriting examples, not ground truth. This is empirical evidence, not a proof of absence for every possible physical model. The images retrieved most closely resemble digits 6, 5, 3, 8, consistent with independent reconstruction.

The image and intensity fingerprints are scale invariant, so constant brightness rescaling would not explain the failure to match. Complete candidate rankings and positive controls are in fingerprint_report.json and fingerprint_scores.npz.
