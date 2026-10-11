# Router comparison

Dataset: **synthetic-model-zoo-7** (synthetic); splits: 42, 43, 44.

Synthetic data exercises code paths; it is not evidence about real models.

| Router | AIQ | Quality at best-single cost | Cost to match best single | Best quality |
|---|---:|---:|---:|---:|
| oracle (reference) | 0.988 ± 0.003 | 0.999 ± 0.001 | 1.2% ± 0.3 | 0.999 ± 0.001 |
| slm-cascade | 0.905 ± 0.011 | 0.960 ± 0.006 | 58.5% ± 3.5 | 0.960 ± 0.006 |
| slm-router | 0.886 ± 0.014 | 0.952 ± 0.013 | 99.9% ± 0.2 | 0.952 ± 0.013 |
| knn | 0.878 ± 0.007 | 0.942 ± 0.012 | n/a | 0.942 ± 0.012 |
| random | 0.487 ± 0.024 | 0.543 ± 0.029 | n/a | 0.543 ± 0.029 |
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

42 target/split checks had certified test traffic; 1 fell below target.

| Router | Split | Target | Certified share (test) | Certified mean (test) | Calibration mean | Holds |
|---|---|---:|---:|---:|---:|---|
| slm-router | 42 | 0.5 | 100% | 0.569 | 0.549 | yes |
| slm-router | 42 | 0.6 | 100% | 0.642 | 0.637 | yes |
| slm-router | 42 | 0.7 | 100% | 0.753 | 0.739 | yes |
| slm-router | 42 | 0.75 | 100% | 0.792 | 0.784 | yes |
| slm-router | 42 | 0.8 | 100% | 0.858 | 0.838 | yes |
| slm-router | 42 | 0.85 | 100% | 0.888 | 0.884 | yes |
| slm-router | 42 | 0.9 | 100% | 0.915 | 0.930 | yes |
| slm-router | 42 | 0.95 | 0% | — | — | yes |
| slm-router | 42 | 1.0 | 0% | — | — | yes |
| slm-cascade | 42 | 0.5 | 100% | 0.581 | 0.589 | yes |
| slm-cascade | 42 | 0.6 | 100% | 0.651 | 0.646 | yes |
| slm-cascade | 42 | 0.7 | 100% | 0.753 | 0.756 | yes |
| slm-cascade | 42 | 0.75 | 100% | 0.788 | 0.794 | yes |
| slm-cascade | 42 | 0.8 | 100% | 0.881 | 0.872 | yes |
| slm-cascade | 42 | 0.85 | 100% | 0.888 | 0.887 | yes |
| slm-cascade | 42 | 0.9 | 100% | 0.903 | 0.926 | yes |
| slm-cascade | 42 | 0.95 | 0% | — | — | yes |
| slm-cascade | 42 | 1.0 | 0% | — | — | yes |
| slm-router | 43 | 0.5 | 100% | 0.560 | 0.542 | yes |
| slm-router | 43 | 0.6 | 100% | 0.633 | 0.648 | yes |
| slm-router | 43 | 0.7 | 100% | 0.706 | 0.735 | yes |
| slm-router | 43 | 0.75 | 100% | 0.807 | 0.790 | yes |
| slm-router | 43 | 0.8 | 100% | 0.829 | 0.830 | yes |
| slm-router | 43 | 0.85 | 100% | 0.920 | 0.888 | yes |
| slm-router | 43 | 0.9 | 100% | 0.939 | 0.927 | yes |
| slm-router | 43 | 0.95 | 0% | — | — | yes |
| slm-router | 43 | 1.0 | 0% | — | — | yes |
| slm-cascade | 43 | 0.5 | 100% | 0.620 | 0.624 | yes |
| slm-cascade | 43 | 0.6 | 100% | 0.654 | 0.663 | yes |
| slm-cascade | 43 | 0.7 | 100% | 0.703 | 0.746 | yes |
| slm-cascade | 43 | 0.75 | 100% | 0.809 | 0.791 | yes |
| slm-cascade | 43 | 0.8 | 100% | 0.846 | 0.836 | yes |
| slm-cascade | 43 | 0.85 | 100% | 0.900 | 0.881 | yes |
| slm-cascade | 43 | 0.9 | 100% | 0.958 | 0.930 | yes |
| slm-cascade | 43 | 0.95 | 0% | — | — | yes |
| slm-cascade | 43 | 1.0 | 0% | — | — | yes |
| slm-router | 44 | 0.5 | 100% | 0.497 | 0.548 | NO |
| slm-router | 44 | 0.6 | 100% | 0.602 | 0.644 | yes |
| slm-router | 44 | 0.7 | 100% | 0.734 | 0.734 | yes |
| slm-router | 44 | 0.75 | 100% | 0.791 | 0.796 | yes |
| slm-router | 44 | 0.8 | 100% | 0.816 | 0.832 | yes |
| slm-router | 44 | 0.85 | 100% | 0.883 | 0.882 | yes |
| slm-router | 44 | 0.9 | 100% | 0.941 | 0.951 | yes |
| slm-router | 44 | 0.95 | 0% | — | — | yes |
| slm-router | 44 | 1.0 | 0% | — | — | yes |
| slm-cascade | 44 | 0.5 | 100% | 0.552 | 0.603 | yes |
| slm-cascade | 44 | 0.6 | 100% | 0.600 | 0.645 | yes |
| slm-cascade | 44 | 0.7 | 100% | 0.736 | 0.739 | yes |
| slm-cascade | 44 | 0.75 | 100% | 0.799 | 0.797 | yes |
| slm-cascade | 44 | 0.8 | 100% | 0.823 | 0.841 | yes |
| slm-cascade | 44 | 0.85 | 100% | 0.905 | 0.896 | yes |
| slm-cascade | 44 | 0.9 | 100% | 0.940 | 0.931 | yes |
| slm-cascade | 44 | 0.95 | 0% | — | — | yes |
| slm-cascade | 44 | 1.0 | 0% | — | — | yes |

## Claim gate

- SYNTHETIC_DATA_IS_NOT_EXTERNAL_BENCHMARK_EVIDENCE
- EXTERNAL_ROUTERS_NOT_RUN_IN_THIS_HARNESS

- AIQ normalizes cost linearly between the cheapest and most expensive single model on each split.
- Quality and cost read from each router's concave hull assume operating points can be mixed.
- The oracle reads test outcomes and is a reference, not a router.
- The pooled guarantee is a 1-quality_risk statement per target and split; occasional violations at about that rate are consistent with it.
