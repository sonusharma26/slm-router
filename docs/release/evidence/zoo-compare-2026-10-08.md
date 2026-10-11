# Router comparison

Dataset: **synthetic-model-zoo-7** (synthetic); splits: 42, 43, 44.

Synthetic data exercises code paths; it is not evidence about real models.

| Router | AIQ | Quality at best-single cost | Cost to match best single | Best quality |
|---|---:|---:|---:|---:|
| oracle (reference) | 0.988 ± 0.003 | 0.999 ± 0.001 | 1.2% ± 0.3 | 0.999 ± 0.001 |
| slm-router | 0.883 ± 0.005 | 0.952 ± 0.013 | 100.0% ± 0.0 | 0.952 ± 0.013 |
| knn | 0.878 ± 0.007 | 0.942 ± 0.012 | n/a | 0.942 ± 0.012 |
| random | 0.487 ± 0.024 | 0.543 ± 0.029 | n/a | 0.543 ± 0.029 |
| slm-router-local (previous) | 0.317 ± 0.026 | 0.911 ± 0.017 | n/a | 0.911 ± 0.017 |
| cheapest | 0.144 ± 0.010 | 0.144 ± 0.010 | n/a | 0.144 ± 0.010 |
| threshold | 0.097 ± 0.008 | 0.944 ± 0.012 | n/a | 0.944 ± 0.012 |
| best_single | 0.000 ± 0.000 | 0.952 ± 0.013 | 100.0% ± 0.0 | 0.952 ± 0.013 |
| model:reason-32b | 0.749 ± 0.008 | 0.815 ± 0.009 | n/a | 0.815 ± 0.009 |
| model:frontier | 0.713 ± 0.014 | 0.824 ± 0.016 | n/a | 0.824 ± 0.016 |
| model:large-70b | 0.688 ± 0.017 | 0.705 ± 0.018 | n/a | 0.705 ± 0.018 |
| model:mid-32b | 0.570 ± 0.035 | 0.574 ± 0.035 | n/a | 0.574 ± 0.035 |
| model:writer-24b | 0.476 ± 0.017 | 0.478 ± 0.017 | n/a | 0.478 ± 0.017 |
| model:med-13b | 0.410 ± 0.023 | 0.410 ± 0.023 | n/a | 0.410 ± 0.023 |
| model:legal-13b | 0.395 ± 0.021 | 0.396 ± 0.021 | n/a | 0.396 ± 0.021 |
| model:math-7b | 0.375 ± 0.018 | 0.375 ± 0.018 | n/a | 0.375 ± 0.018 |
| model:coder-14b | 0.369 ± 0.008 | 0.369 ± 0.008 | n/a | 0.369 ± 0.008 |
| model:small-8b | 0.310 ± 0.008 | 0.311 ± 0.008 | n/a | 0.311 ± 0.008 |
| model:tiny-1b | 0.144 ± 0.010 | 0.144 ± 0.010 | n/a | 0.144 ± 0.010 |
| model:frontier-reason | 0.000 ± 0.000 | 0.952 ± 0.013 | 100.0% ± 0.0 | 0.952 ± 0.013 |

## Pooled quality guarantee

21 target/split checks had certified test traffic; 1 fell below target.

| Split | Target | Certified share (test) | Certified mean (test) | Calibration mean | Holds |
|---|---:|---:|---:|---:|---|
| 42 | 0.5 | 100% | 0.539 | 0.560 | yes |
| 42 | 0.6 | 100% | 0.646 | 0.640 | yes |
| 42 | 0.7 | 100% | 0.758 | 0.755 | yes |
| 42 | 0.75 | 100% | 0.820 | 0.797 | yes |
| 42 | 0.8 | 100% | 0.858 | 0.841 | yes |
| 42 | 0.85 | 100% | 0.900 | 0.890 | yes |
| 42 | 0.9 | 100% | 0.924 | 0.932 | yes |
| 42 | 0.95 | 0% | — | — | yes |
| 42 | 1.0 | 0% | — | — | yes |
| 43 | 0.5 | 100% | 0.571 | 0.565 | yes |
| 43 | 0.6 | 100% | 0.625 | 0.639 | yes |
| 43 | 0.7 | 100% | 0.724 | 0.748 | yes |
| 43 | 0.75 | 100% | 0.799 | 0.791 | yes |
| 43 | 0.8 | 100% | 0.849 | 0.842 | yes |
| 43 | 0.85 | 93% | 0.964 | 0.949 | yes |
| 43 | 0.9 | 93% | 0.964 | 0.949 | yes |
| 43 | 0.95 | 0% | — | — | yes |
| 43 | 1.0 | 0% | — | — | yes |
| 44 | 0.5 | 100% | 0.470 | 0.544 | NO |
| 44 | 0.6 | 100% | 0.617 | 0.641 | yes |
| 44 | 0.7 | 100% | 0.729 | 0.748 | yes |
| 44 | 0.75 | 100% | 0.809 | 0.805 | yes |
| 44 | 0.8 | 100% | 0.843 | 0.846 | yes |
| 44 | 0.85 | 100% | 0.890 | 0.880 | yes |
| 44 | 0.9 | 96% | 0.938 | 0.951 | yes |
| 44 | 0.95 | 0% | — | — | yes |
| 44 | 1.0 | 0% | — | — | yes |

## Claim gate

- SYNTHETIC_DATA_IS_NOT_EXTERNAL_BENCHMARK_EVIDENCE
- EXTERNAL_ROUTERS_NOT_RUN_IN_THIS_HARNESS

- AIQ normalizes cost linearly between the cheapest and most expensive single model on each split.
- Quality and cost read from each router's concave hull assume operating points can be mixed.
- The oracle reads test outcomes and is a reference, not a router.
- The pooled guarantee is a 1-quality_risk statement per target and split; occasional violations at about that rate are consistent with it.
