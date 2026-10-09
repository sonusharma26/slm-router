# Router comparison

Dataset: **synthetic-model-zoo-7** (synthetic); splits: 42, 43, 44.

Synthetic data exercises code paths; it is not evidence about real models.

| Router | AIQ | Quality at best-single cost | Cost to match best single | Best quality |
|---|---:|---:|---:|---:|
| oracle (reference) | 0.988 ± 0.003 | 0.999 ± 0.001 | 1.2% ± 0.3 | 0.999 ± 0.001 |
| slm-cascade | 0.910 ± 0.008 | 0.963 ± 0.009 | 58.1% ± 14.0 | 0.965 ± 0.007 |
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

43 target/split checks had certified test traffic; 1 fell below target.

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
| slm-cascade | 42 | 0.9 | 100% | 0.946 | 0.925 | yes |
| slm-cascade | 42 | 0.95 | 100% | 0.958 | 0.970 | yes |
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
| slm-cascade | 43 | 0.8 | 100% | 0.860 | 0.840 | yes |
| slm-cascade | 43 | 0.85 | 100% | 0.900 | 0.881 | yes |
| slm-cascade | 43 | 0.9 | 100% | 0.927 | 0.933 | yes |
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
| slm-cascade | 44 | 0.85 | 100% | 0.903 | 0.895 | yes |
| slm-cascade | 44 | 0.9 | 100% | 0.936 | 0.922 | yes |
| slm-cascade | 44 | 0.95 | 0% | — | — | yes |
| slm-cascade | 44 | 1.0 | 0% | — | — | yes |

## Cascade operating points

Kind: plain price-penalty routing, answer-agreement cascade, or majority vote (voters/quorum). Rates are shares of test requests; calls are endpoints invoked per request.

| Split | Target | Kind | Voters/quorum | Penalty | Accept | Calls | Check | Vote | Escalated | Test quality | Test cost | Calibrated |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.5 | plain | — | 2174 | -1 | 1.00 | 0% | 0% | 0% | 0.581 | 0.00014 | yes |
| 42 | 0.6 | plain | — | 1298 | -1 | 1.00 | 0% | 0% | 0% | 0.651 | 0.00016 | yes |
| 42 | 0.7 | plain | — | 30.71 | -1 | 1.00 | 0% | 0% | 0% | 0.753 | 0.00203 | yes |
| 42 | 0.75 | plain | — | 14.78 | -1 | 1.00 | 0% | 0% | 0% | 0.788 | 0.00404 | yes |
| 42 | 0.8 | agree | — | 5.722 | 0.8 | 1.40 | 26% | 0% | 15% | 0.881 | 0.01769 | yes |
| 42 | 0.85 | agree | — | 14.78 | 0.8 | 1.61 | 39% | 0% | 22% | 0.888 | 0.02078 | yes |
| 42 | 0.9 | vote | 3/2 seq | 8.452 | 2 | 2.61 | 100% | 81% | 19% | 0.946 | 0.02408 | yes |
| 42 | 0.95 | vote | 3/3 seq | 8.452 | 2 | 3.12 | 100% | 47% | 53% | 0.958 | 0.04691 | yes |
| 42 | 1.0 | vote | 3/3 seq | 8.452 | 2 | 3.12 | 100% | 47% | 53% | 0.958 | 0.04691 | no |
| 43 | 0.5 | plain | — | 2067 | -1 | 1.00 | 0% | 0% | 0% | 0.620 | 0.00013 | yes |
| 43 | 0.6 | plain | — | 1537 | -1 | 1.00 | 0% | 0% | 0% | 0.654 | 0.00015 | yes |
| 43 | 0.7 | plain | — | 142 | -1 | 1.00 | 0% | 0% | 0% | 0.703 | 0.00050 | yes |
| 43 | 0.75 | plain | — | 30.72 | -1 | 1.00 | 0% | 0% | 0% | 0.809 | 0.00243 | yes |
| 43 | 0.8 | plain | — | 3.409 | -1 | 1.00 | 0% | 0% | 0% | 0.860 | 0.01035 | yes |
| 43 | 0.85 | agree | — | 8.564 | 0.8 | 1.40 | 26% | 0% | 14% | 0.900 | 0.01761 | yes |
| 43 | 0.9 | vote | 3/2 seq | 8.564 | 2 | 2.55 | 100% | 83% | 17% | 0.927 | 0.02359 | yes |
| 43 | 0.95 | vote | 5/4 seq | 8.564 | 2 | 4.61 | 100% | 55% | 45% | 0.975 | 0.04645 | no |
| 43 | 1.0 | vote | 5/4 seq | 8.564 | 2 | 4.61 | 100% | 55% | 45% | 0.975 | 0.04645 | no |
| 44 | 0.5 | plain | — | 2191 | -1 | 1.00 | 0% | 0% | 0% | 0.552 | 0.00014 | yes |
| 44 | 0.6 | plain | — | 1663 | -1 | 1.00 | 0% | 0% | 0% | 0.600 | 0.00016 | yes |
| 44 | 0.7 | plain | — | 44.7 | -1 | 1.00 | 0% | 0% | 0% | 0.736 | 0.00142 | yes |
| 44 | 0.75 | plain | — | 16.23 | -1 | 1.00 | 0% | 0% | 0% | 0.799 | 0.00436 | yes |
| 44 | 0.8 | plain | — | 3.313 | -1 | 1.00 | 0% | 0% | 0% | 0.823 | 0.01114 | yes |
| 44 | 0.85 | vote | 3/2 seq | 44.7 | 2 | 2.77 | 100% | 73% | 27% | 0.903 | 0.02349 | yes |
| 44 | 0.9 | vote | 3/2 seq | 8.626 | 2 | 2.70 | 100% | 75% | 25% | 0.936 | 0.02926 | yes |
| 44 | 0.95 | vote | 5/3 seq | 0 | 2 | 3.62 | 100% | 77% | 23% | 0.963 | 0.07559 | no |
| 44 | 1.0 | vote | 5/3 seq | 0 | 2 | 3.62 | 100% | 77% | 23% | 0.963 | 0.07559 | no |

## Claim gate

- SYNTHETIC_DATA_IS_NOT_EXTERNAL_BENCHMARK_EVIDENCE
- EXTERNAL_ROUTERS_NOT_RUN_IN_THIS_HARNESS

- AIQ normalizes cost linearly between the cheapest and most expensive single model on each split.
- Quality and cost read from each router's concave hull assume operating points can be mixed.
- The oracle reads test outcomes and is a reference, not a router.
- The pooled guarantee is a 1-quality_risk statement per target and split; occasional violations at about that rate are consistent with it.
