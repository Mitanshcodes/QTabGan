Mean +- sd over seeds (n per arm in brackets). ML-utility rows are real - synthetic, averaged over lr/dt/rf/mlp/svm; lower is better everywhere.

| metric | ctgan [2] | frozen_quantum [2] | gaussian [2] | quantum_angle [2] | quantum [2] | floor: marginals | ceiling: real copy |
|---|---|---|---|---|---|---|---|
| acc | 4.40 +- 0.82 | 7.98 +- 2.84 | 4.29 +- 0.09 | 9.86 +- 5.49 | 11.91 +- 2.54 | 9.34 | 0.18 |
| f1 | 0.101 +- 0.061 | 0.227 +- 0.054 | 0.125 +- 0.046 | 0.202 +- 0.090 | 0.069 +- 0.012 | 0.276 | 0.001 |
| auc | 0.057 +- 0.002 | 0.248 +- 0.113 | 0.107 +- 0.015 | 0.184 +- 0.002 | 0.091 +- 0.016 | 0.337 | 0.001 |
| avg_jsd | 0.106 +- 0.003 | 0.361 +- 0.128 | 0.170 +- 0.010 | 0.367 +- 0.027 | 0.417 +- 0.013 | 0.006 | 0.002 |
| avg_wd | 0.012 +- 0.001 | 0.072 +- 0.019 | 0.036 +- 0.007 | 0.098 +- 0.027 | 0.088 +- 0.039 | 0.001 | 0.000 |
| diff_corr | 0.67 +- 0.06 | 4.57 +- 1.14 | 1.69 +- 0.19 | 3.99 +- 0.54 | 3.05 +- 0.01 | 2.71 | 0.03 |
| js_div (paper def) | 0.0133 +- 0.0003 | 0.1799 +- 0.1090 | 0.0437 +- 0.0094 | 0.1709 +- 0.0209 | 0.2093 +- 0.0191 | 0.0001 | 0.0000 |
| avg|dPearson| (paper def) | 0.036 +- 0.001 | 0.207 +- 0.069 | 0.070 +- 0.028 | 0.179 +- 0.076 | 0.146 +- 0.018 | 0.076 | 0.002 |

Paired t-tests (same seed index), Holm-corrected over all comparisons below. Negative delta = quantum arm better. n=2 seeds per arm: a pilot with very low power (df=1, p-values are close to meaningless).
  quantum_angle - ctgan            f1         delta +0.1010   p=0.516  p_holm=1.000
  quantum_angle - ctgan            diff_corr  delta +3.3210   p=0.065  p_holm=0.717
  quantum_angle - frozen_quantum   f1         delta -0.0250   p=0.508  p_holm=1.000
  quantum_angle - frozen_quantum   diff_corr  delta -0.5775   p=0.712  p_holm=1.000
  quantum_angle - gaussian         f1         delta +0.0769   p=0.570  p_holm=1.000
  quantum_angle - gaussian         diff_corr  delta +2.2997   p=0.068  p_holm=0.717
  quantum - ctgan                  f1         delta -0.0318   p=0.522  p_holm=1.000
  quantum - ctgan                  diff_corr  delta +2.3835   p=0.013  p_holm=0.152
  quantum - frozen_quantum         f1         delta -0.1578   p=0.184  p_holm=1.000
  quantum - frozen_quantum         diff_corr  delta -1.5150   p=0.309  p_holm=1.000
  quantum - gaussian               f1         delta -0.0560   p=0.255  p_holm=1.000
  quantum - gaussian               diff_corr  delta +1.3622   p=0.066  p_holm=0.717
