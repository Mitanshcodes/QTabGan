Random forest (100 trees) trained on each data source, tested on the real Adult test split (9,769 rows; positive class income >50K, 24%). Mean ± SD over 5 generator seeds (real: 5 forest seeds).

| Training data for the forest | Accuracy | Macro F1 | AUROC | Avg. precision | Recall of >50K |
|---|---|---|---|---|---|
| Real training rows (reference) | 0.861 ± 0.003 | 0.797 ± 0.003 | 0.909 ± 0.001 | 0.792 ± 0.001 | 0.627 ± 0.004 |
| QTabGAN (best searched) | 0.807 ± 0.024 | 0.756 ± 0.015 | 0.875 ± 0.008 | 0.688 ± 0.027 | 0.727 ± 0.077 |
| Gaussian control (best searched) | 0.829 ± 0.010 | 0.756 ± 0.010 | 0.871 ± 0.014 | 0.702 ± 0.016 | 0.594 ± 0.059 |
| CTGAN (classical baseline) | 0.829 ± 0.005 | 0.759 ± 0.008 | 0.880 ± 0.004 | 0.693 ± 0.006 | 0.612 ± 0.081 |
| QTabGAN, first-pass settings | 0.709 ± 0.102 | 0.534 ± 0.087 | 0.653 ± 0.113 | 0.363 ± 0.112 | 0.345 ± 0.390 |
| QTabGAN, literal paper settings | 0.721 ± 0.123 | 0.572 ± 0.053 | 0.697 ± 0.039 | 0.441 ± 0.050 | 0.357 ± 0.319 |
| Majority class (always <=50K) | 0.761 | 0.432 | 0.500 | 0.239 | 0.000 |
