Mean +- sd over seeds (n per arm in brackets). ML-utility rows are real - synthetic, averaged over lr/dt/rf/mlp/svm; lower is better everywhere.

| metric | best_gaussian [2] | best_quantum [2] | paper_literal [2] | floor: marginals | ceiling: real copy |
|---|---|---|---|---|---|
| acc | 5.34 +- 1.53 | 9.72 +- 2.52 | 7.77 +- 0.96 | 9.34 | 0.18 |
| f1 | 0.160 +- 0.036 | 0.213 +- 0.011 | 0.237 +- 0.011 | 0.276 | 0.001 |
| auc | 0.090 +- 0.010 | 0.241 +- 0.051 | 0.231 +- 0.055 | 0.337 | 0.001 |
| avg_jsd | 0.048 +- 0.005 | 0.376 +- 0.003 | 0.580 +- 0.192 | 0.006 | 0.002 |
| avg_wd | 0.037 +- 0.025 | 0.036 +- 0.009 | 0.124 +- 0.059 | 0.001 | 0.000 |
| diff_corr | 0.82 +- 0.09 | 2.23 +- 0.47 | 5.15 +- 0.18 | 2.71 | 0.03 |
| js_div (paper def) | 0.0040 +- 0.0021 | 0.1756 +- 0.0047 | 0.4277 +- 0.2435 | 0.0001 | 0.0000 |
| avg abs dPearson (paper def) | 0.036 +- 0.002 | 0.087 +- 0.001 | 0.359 +- 0.056 | 0.076 | 0.002 |

Paired t-tests (same seed index), Holm-corrected over all comparisons below. Negative delta = quantum arm better. n=2 seeds per arm: a pilot with very low power (df=1, p-values are close to meaningless).
  best_quantum - best_gaussian     f1         delta +0.0536   p=0.199  p_holm=0.399
  best_quantum - best_gaussian     diff_corr  delta +1.4130   p=0.122  p_holm=0.366
  paper_literal - best_gaussian    f1         delta +0.0769   p=0.258  p_holm=0.399
  paper_literal - best_gaussian    diff_corr  delta +4.3376   p=0.028  p_holm=0.111
