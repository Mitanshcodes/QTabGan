Mean +- sd over seeds (n per arm in brackets). ML-utility rows are real - synthetic, averaged over lr/dt/rf/mlp/svm; lower is better everywhere.

| metric | gaussian_paper1 [5] | gaussian_score1 [5] | gaussian_score2 [5] | quantum_paper1 [5] | quantum_score1 [5] | quantum_score2 [5] | floor: marginals | ceiling: real copy |
|---|---|---|---|---|---|---|---|---|
| acc | 3.72 +- 0.57 | 4.17 +- 0.61 | 4.25 +- 0.66 | 4.58 +- 1.42 | 3.60 +- 0.81 | 4.32 +- 0.75 | 9.34 | 0.18 |
| f1 | 0.122 +- 0.030 | 0.128 +- 0.011 | 0.123 +- 0.015 | 0.106 +- 0.020 | 0.115 +- 0.023 | 0.102 +- 0.015 | 0.276 | 0.001 |
| auc | 0.045 +- 0.010 | 0.047 +- 0.004 | 0.056 +- 0.005 | 0.065 +- 0.017 | 0.041 +- 0.009 | 0.044 +- 0.004 | 0.337 | 0.001 |
| avg_jsd | 0.048 +- 0.004 | 0.025 +- 0.002 | 0.034 +- 0.001 | 0.097 +- 0.007 | 0.027 +- 0.002 | 0.028 +- 0.002 | 0.006 | 0.002 |
| avg_wd | 0.021 +- 0.008 | 0.015 +- 0.002 | 0.011 +- 0.002 | 0.023 +- 0.004 | 0.012 +- 0.005 | 0.014 +- 0.005 | 0.001 | 0.000 |
| diff_corr | 0.48 +- 0.03 | 0.44 +- 0.04 | 0.35 +- 0.04 | 1.23 +- 0.19 | 0.40 +- 0.03 | 0.43 +- 0.05 | 2.71 | 0.03 |
| js_div (paper def) | 0.0043 +- 0.0010 | 0.0008 +- 0.0001 | 0.0025 +- 0.0004 | 0.0129 +- 0.0019 | 0.0011 +- 0.0002 | 0.0012 +- 0.0002 | 0.0001 | 0.0000 |
| avg abs dPearson (paper def) | 0.017 +- 0.002 | 0.019 +- 0.002 | 0.014 +- 0.001 | 0.038 +- 0.009 | 0.015 +- 0.002 | 0.016 +- 0.002 | 0.076 | 0.002 |

Paired t-tests (same seed index), Holm-corrected over all comparisons below. Negative delta = quantum arm better. n=5 seeds per arm: a pilot with very low power.
  quantum_paper1 - gaussian_paper1 f1         delta -0.0156   p=0.148  p_holm=1.000
  quantum_paper1 - gaussian_paper1 diff_corr  delta +0.7447   p=0.001  p_holm=0.009
  quantum_paper1 - gaussian_score1 f1         delta -0.0213   p=0.086  p_holm=0.944
  quantum_paper1 - gaussian_score1 diff_corr  delta +0.7850   p=0.001  p_holm=0.018
  quantum_paper1 - gaussian_score2 f1         delta -0.0166   p=0.200  p_holm=1.000
  quantum_paper1 - gaussian_score2 diff_corr  delta +0.8774   p=0.001  p_holm=0.009
  quantum_score1 - gaussian_paper1 f1         delta -0.0074   p=0.672  p_holm=1.000
  quantum_score1 - gaussian_paper1 diff_corr  delta -0.0779   p=0.000  p_holm=0.005
  quantum_score1 - gaussian_score1 f1         delta -0.0131   p=0.375  p_holm=1.000
  quantum_score1 - gaussian_score1 diff_corr  delta -0.0376   p=0.197  p_holm=1.000
  quantum_score1 - gaussian_score2 f1         delta -0.0085   p=0.541  p_holm=1.000
  quantum_score1 - gaussian_score2 diff_corr  delta +0.0548   p=0.067  p_holm=0.802
  quantum_score2 - gaussian_paper1 f1         delta -0.0198   p=0.234  p_holm=1.000
  quantum_score2 - gaussian_paper1 diff_corr  delta -0.0475   p=0.153  p_holm=1.000
  quantum_score2 - gaussian_score1 f1         delta -0.0255   p=0.018  p_holm=0.231
  quantum_score2 - gaussian_score1 diff_corr  delta -0.0072   p=0.840  p_holm=1.000
  quantum_score2 - gaussian_score2 f1         delta -0.0208   p=0.173  p_holm=1.000
  quantum_score2 - gaussian_score2 diff_corr  delta +0.0853   p=0.014  p_holm=0.196
