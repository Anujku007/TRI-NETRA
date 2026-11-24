# TRI-Netra Research Phase 2: Experimental Evaluation

This report contains measured results from the configured local evaluation runs. It does not claim improvement over prior work or state-of-the-art methods.

## WIDER FACE validation

Easy, Medium, and Hard use the official WIDER FACE validation keep-index annotations. Predictions overlapping non-kept boxes are ignored. Both detectors use the same image list, inclusive WIDER IoU arithmetic, score normalization, 1,000-point PR thresholds, AP integration, and image-cluster bootstrap samples.

Precision, recall, and F1 are measured at each detector's predeclared native output point. AP and PR curves use all emitted detections ranked by detector-native scores. Haar level weights are rankings, not calibrated probabilities; score magnitudes are not compared between detector families.

| Detector | Subset | Images | GT | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | AP (95% CI) | Mean latency ms (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| humanjs | Easy | 3226 | 7211 | 4305 | 474 | 2906 | 0.9008 [0.8896, 0.9118] | 0.5970 [0.5765, 0.6178] | 0.7181 [0.7026, 0.7345] | 0.5681 [0.5482, 0.5896] | 123.869 [121.068, 126.651] |
| humanjs | Medium | 3226 | 13319 | 5864 | 502 | 7455 | 0.9211 [0.9131, 0.9291] | 0.4403 [0.4182, 0.4607] | 0.5958 [0.5757, 0.6147] | 0.4222 [0.4012, 0.4423] | 123.869 [121.068, 126.651] |
| humanjs | Hard | 3226 | 31958 | 6051 | 504 | 25907 | 0.9231 [0.9154, 0.9306] | 0.1893 [0.1723, 0.2083] | 0.3142 [0.2903, 0.3400] | 0.1816 [0.1650, 0.1999] | 123.869 [121.068, 126.651] |
| haar | Easy | 3226 | 7211 | 3379 | 2694 | 3832 | 0.5564 [0.5393, 0.5733] | 0.4686 [0.4490, 0.4896] | 0.5087 [0.4918, 0.5254] | 0.4252 [0.4041, 0.4465] | 64.975 [63.996, 65.942] |
| haar | Medium | 3226 | 13319 | 5257 | 2694 | 8062 | 0.6612 [0.6435, 0.6769] | 0.3947 [0.3749, 0.4147] | 0.4943 [0.4755, 0.5116] | 0.3668 [0.3464, 0.3861] | 64.975 [63.996, 65.942] |
| haar | Hard | 3226 | 31958 | 6012 | 2694 | 25946 | 0.6906 [0.6740, 0.7062] | 0.1881 [0.1710, 0.2061] | 0.2957 [0.2735, 0.3180] | 0.1754 [0.1590, 0.1927] | 64.975 [63.996, 65.942] |

Intervals are percentile 95% confidence intervals from 1,000 image-cluster bootstrap replicates. PR points are in `results/widerface_pr_curves.csv`.

## Identity score calibration

The runtime score implementation was imported directly from `shared/utils/faceIdentityMatcher.ts::faceEmbeddingSimilarity`. The calibration and evaluation identities are disjoint: **True**.

| Split | Usable identities | Usable images | Adequacy flag | Runtime threshold | Same-identity match (95% CI) | Different-identity false-match (95% CI) |
|---|---:|---:|---|---:|---:|---:|
| calibration | 60 | 535 | True | 0.42 | 0.7521 [0.7028, 0.8002] | 0.0064 [0.0036, 0.0102] |
| evaluation | 40 | 360 | True | 0.42 | 0.7679 [0.7120, 0.8217] | 0.0093 [0.0039, 0.0172] |

Coarse-hash pairs flagged for review: 64667 (64035 cross-identity). Coarse-hash candidates are never discarded. Only same-identity exact-byte duplicates are removed.

Exact-byte duplicate identity groups kept within one split: 0.

Runtime gallery identification uses one deterministic enrollment image per identity and all remaining images as queries. It applies both the runtime authorization threshold and best-versus-runner-up margin; its confidence intervals resample identities.

| Split | Gallery images | Queries | Rank-1 accuracy (95% CI) | Accepted coverage (95% CI) | Accepted accuracy (95% CI) | False accepts/query (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 60 | 475 | 0.8189 [0.7553, 0.8774] | 0.7305 [0.6612, 0.7971] | 0.9280 [0.8933, 0.9591] | 0.0526 [0.0295, 0.0776] |
| evaluation | 40 | 320 | 0.8938 [0.8534, 0.9323] | 0.7625 [0.6982, 0.8211] | 0.9713 [0.9489, 0.9882] | 0.0219 [0.0092, 0.0372] |

## Reproducibility and limitations

- Experiment configuration SHA-256: `a0a293320ac677691187bfddd5db08a01b512c2c11d58c2bc265eff68c6ba640`.
- WIDER validation images scored: 3226.
- WIDER annotation bounding boxes: 39708.
- Human.js uses the unchanged runtime detector service, including its native confidence and maximum detections per image. Haar uses OpenCV's default configured cascade and native level weights.
- Results are conditional on the bundled LFW sample, local model artifacts, and the current CPU/software environment. They are not independent external validation of biometric identity accuracy or of production performance.
- Confidence intervals resample images for detection metrics and identities for identity-pair calibration; they do not account for dataset shift or annotation error.
