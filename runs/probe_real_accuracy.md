Target (QTabGAN Table 3, Adult, real data): accuracy 0.776, F1 0.7619. Majority-class accuracy: 0.7607.

| classifier | scaling | accuracy | F1 macro | F1 weighted | F1 positive class | distance to target |
|---|---|---|---|---|---|---|
| KNN (k=5) | none | 0.7779 | 0.6435 | 0.7576 | 0.4247 | 0.0061 |
| LogisticRegression | none | 0.7895 | 0.6423 | 0.7620 | 0.4129 | 0.0136 |
| GaussianNB | none | 0.7967 | 0.6492 | 0.7678 | 0.4217 | 0.0266 |
| GaussianNB | minmax | 0.8001 | 0.6519 | 0.7703 | 0.4247 | 0.0325 |
| GaussianNB | standard | 0.8001 | 0.6519 | 0.7703 | 0.4247 | 0.0325 |
| LinearSVC | none | 0.8062 | 0.6400 | 0.7675 | 0.3954 | 0.0359 |
| LogisticRegression | minmax | 0.8047 | 0.6527 | 0.7725 | 0.4229 | 0.0393 |
| LinearSVC | minmax | 0.8068 | 0.6486 | 0.7715 | 0.4127 | 0.0405 |
| LinearSVC | standard | 0.8068 | 0.6496 | 0.7720 | 0.4149 | 0.0409 |
| MLP | none | 0.7936 | 0.5854 | 0.7386 | 0.2916 | 0.0409 |
| LogisticRegression | standard | 0.8050 | 0.6579 | 0.7748 | 0.4335 | 0.0419 |
| DecisionTree | none | 0.8128 | 0.7456 | 0.8138 | 0.6149 | 0.0531 |
| DecisionTree | minmax | 0.8129 | 0.7456 | 0.8138 | 0.6147 | 0.0532 |
| DecisionTree | standard | 0.8125 | 0.7450 | 0.8134 | 0.6138 | 0.0534 |
| KNN (k=5) | standard | 0.8245 | 0.7479 | 0.8204 | 0.6089 | 0.0626 |
| KNN (k=5) | minmax | 0.8259 | 0.7467 | 0.8205 | 0.6051 | 0.0651 |
| MLP | minmax | 0.8433 | 0.7783 | 0.8409 | 0.6583 | 0.0837 |
| MLP | standard | 0.8418 | 0.7852 | 0.8427 | 0.6748 | 0.0891 |
| RandomForest | standard | 0.8623 | 0.7979 | 0.8574 | 0.6838 | 0.1223 |
| RandomForest | none | 0.8623 | 0.7981 | 0.8575 | 0.6842 | 0.1225 |
| RandomForest | minmax | 0.8624 | 0.7980 | 0.8575 | 0.6839 | 0.1225 |
| GradientBoosting | minmax | 0.8716 | 0.8065 | 0.8650 | 0.6943 | 0.1403 |
| GradientBoosting | none | 0.8716 | 0.8065 | 0.8650 | 0.6943 | 0.1403 |
| GradientBoosting | standard | 0.8716 | 0.8065 | 0.8650 | 0.6943 | 0.1403 |
