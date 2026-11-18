Random forest (100 trees) trained on each data source, tested on the real Adult test split (9,769 rows; positive class income >50K, 24%). Mean ± SD over 5 generator seeds (real: 5 forest seeds).

| Training data for the forest | Accuracy | Macro F1 | AUROC | Avg. precision | Recall of >50K |
|---|---|---|---|---|---|
| Real training rows (reference) | 0.861 ± 0.003 | 0.797 ± 0.003 | 0.909 ± 0.001 | 0.792 ± 0.001 | 0.627 ± 0.004 |
| QTabGAN (best searched) | 0.807 ± 0.024 | 0.756 ± 0.015 | 0.875 ± 0.008 | 0.688 ± 0.027 | 0.727 ± 0.077 |
| Gaussian control (best searched) | 0.829 ± 0.010 | 0.756 ± 0.010 | 0.871 ± 0.014 | 0.702 ± 0.016 | 0.594 ± 0.059 |
| CTGAN (classical baseline) | 0.834 (1 seed) | 0.750 (1 seed) | 0.877 (1 seed) | 0.698 (1 seed) | 0.532 (1 seed) |
| QTabGAN, first-pass settings | 0.753 (1 seed) | 0.481 (1 seed) | 0.574 (1 seed) | 0.275 (1 seed) | 0.061 (1 seed) |
| QTabGAN, literal paper settings | 0.740 (1 seed) | 0.526 (1 seed) | 0.704 (1 seed) | 0.376 (1 seed) | 0.142 (1 seed) |
| Majority class (always <=50K) | 0.761 | 0.432 | 0.500 | 0.239 | 0.000 |
