
### quantum: 35 configs complete (x2 seeds), 70 runs

base (2-seed mean): score 0.510  f1 0.109  avg_jsd 0.136  diff_corr 1.91.  Seed noise sigma(score) = 0.095 (from 35 seed pairs); a setting matters if max |d score| > 2 sigma = 0.190.

| setting | max abs d score | max abs d F1 gap | max abs d avg JSD | max abs d Diff.Corr | level with best score (d score) | matters? |
|---|---|---|---|---|---|---|
| cat-output | 1.068 | 0.162 | 0.403 | 3.76 | linear (+0.752) | yes |
| noise-encoding | 0.844 | 0.135 | 0.287 | 3.17 | none (+0.844) | yes |
| prob-scale | 0.758 | 0.110 | 0.192 | 3.56 | none (+0.758) | yes |
| d-steps | 0.566 | 0.124 | 0.298 | 0.90 | 2 (+0.452) | yes |
| disc-dims | 0.515 | 0.133 | 0.161 | 1.44 | 512-512 (+0.226) | yes |
| lr-classical | 0.474 | 0.104 | 0.198 | 1.12 | 0.001 (+0.266) | yes |
| batch-size | 0.391 | 0.101 | 0.210 | 0.97 | 500 (+0.290) | yes |
| loss | 0.375 | 0.018 | 0.147 | 1.75 | wgan-gp (+0.375) | yes |
| betas | 0.321 | 0.038 | 0.130 | 2.12 | 0.5-0.999 (+0.321) | yes |
| mapper-dims | 0.267 | 0.094 | 0.119 | 0.28 | 256-256 (+0.089) | yes |
| angle-init | 0.256 | 0.068 | 0.070 | 0.78 | uniform-0-2pi (+0.256) | yes |
| lr-quantum | 0.244 | 0.089 | 0.031 | 0.77 | 0.0001 (+0.072) | yes |
| noise-scale | 0.179 | 0.109 | 0.069 | 0.39 | 0.5 (+0.116) | no |
| n-qubits | 0.129 | 0.035 | 0.060 | 1.02 | 6 (+0.081) | no |
| n-shots | 0.096 | 0.080 | 0.029 | 0.16 | 0 (-0.006) | no |
| n-layers | 0.076 | 0.042 | 0.019 | 0.44 | 2 (-0.050) | no |
| input-noise-var | 0.024 | 0.051 | 0.014 | 0.54 | 0.001 (+0.018) | no |

<details><summary>all configs (2-seed means)</summary>

| config | score | f1 | avg_jsd | diff_corr | score s0 | score s1 |
|---|---|---|---|---|---|---|
| n-layers=2 | 0.460 | 0.090 | 0.143 | 1.64 | 0.398 | 0.521 |
| n-layers=1 | 0.486 | 0.114 | 0.155 | 1.48 | 0.513 | 0.460 |
| n-shots=0 | 0.504 | 0.121 | 0.131 | 1.76 | 0.426 | 0.583 |
| base | 0.510 | 0.109 | 0.136 | 1.91 | 0.512 | 0.508 |
| n-layers=8 | 0.516 | 0.105 | 0.147 | 1.92 | 0.525 | 0.508 |
| input-noise-var=0.001 | 0.528 | 0.153 | 0.127 | 1.60 | 0.413 | 0.643 |
| input-noise-var=0.0 | 0.535 | 0.160 | 0.150 | 1.38 | 0.648 | 0.421 |
| lr-quantum=0.0001 | 0.582 | 0.157 | 0.167 | 1.67 | 0.541 | 0.623 |
| n-layers=6 | 0.586 | 0.150 | 0.137 | 2.05 | 0.637 | 0.536 |
| n-qubits=6 | 0.592 | 0.115 | 0.194 | 2.03 | 0.658 | 0.525 |
| mapper-dims=256-256 | 0.600 | 0.141 | 0.211 | 1.63 | 0.602 | 0.597 |
| n-shots=100 | 0.606 | 0.143 | 0.165 | 2.07 | 0.610 | 0.602 |
| n-shots=1024 | 0.606 | 0.189 | 0.131 | 1.81 | 0.685 | 0.528 |
| mapper-dims=1024-1024 | 0.615 | 0.179 | 0.170 | 1.67 | 0.572 | 0.659 |
| noise-scale=0.5 | 0.626 | 0.178 | 0.165 | 1.82 | 0.563 | 0.690 |
| n-qubits=4 | 0.639 | 0.074 | 0.196 | 2.94 | 0.681 | 0.598 |
| noise-scale=2.0 | 0.689 | 0.218 | 0.206 | 1.52 | 0.684 | 0.694 |
| disc-dims=512-512 | 0.736 | 0.176 | 0.289 | 1.71 | 0.623 | 0.849 |
| lr-quantum=0.01 | 0.755 | 0.198 | 0.166 | 2.68 | 0.650 | 0.859 |
| angle-init=uniform-0-2pi | 0.767 | 0.177 | 0.206 | 2.69 | 0.777 | 0.756 |
| lr-classical=0.001 | 0.776 | 0.199 | 0.259 | 2.04 | 0.745 | 0.807 |
| mapper-dims=128-128 | 0.777 | 0.203 | 0.255 | 2.03 | 0.764 | 0.789 |
| batch-size=500 | 0.800 | 0.177 | 0.218 | 2.88 | 0.785 | 0.815 |
| betas=0.5-0.999 | 0.831 | 0.071 | 0.266 | 4.03 | 0.862 | 0.800 |
| disc-dims=1024-1024 | 0.851 | 0.241 | 0.294 | 1.86 | 0.865 | 0.837 |
| loss=wgan-gp | 0.885 | 0.127 | 0.283 | 3.66 | 0.902 | 0.869 |
| batch-size=128 | 0.901 | 0.209 | 0.347 | 2.23 | 0.861 | 0.941 |
| d-steps=2 | 0.962 | 0.233 | 0.368 | 2.29 | 1.024 | 0.901 |
| lr-classical=0.0002 | 0.985 | 0.212 | 0.334 | 3.03 | 1.004 | 0.965 |
| disc-dims=128-128 | 1.026 | 0.241 | 0.297 | 3.36 | 0.906 | 1.145 |
| d-steps=5 | 1.076 | 0.223 | 0.434 | 2.81 | 1.075 | 1.077 |
| cat-output=linear | 1.262 | 0.219 | 0.391 | 4.87 | 1.475 | 1.049 |
| prob-scale=none | 1.268 | 0.219 | 0.328 | 5.47 | 1.288 | 1.248 |
| noise-encoding=none | 1.354 | 0.244 | 0.423 | 5.08 | 1.435 | 1.272 |
| cat-output=softmax | 1.578 | 0.271 | 0.539 | 5.68 | 1.583 | 1.573 |

</details>

### gaussian: 21 configs complete (x2 seeds), 42 runs

base (2-seed mean): score 0.503  f1 0.147  avg_jsd 0.099  diff_corr 1.70.  Seed noise sigma(score) = 0.058 (from 21 seed pairs); a setting matters if max |d score| > 2 sigma = 0.115.

| setting | max abs d score | max abs d F1 gap | max abs d avg JSD | max abs d Diff.Corr | level with best score (d score) | matters? |
|---|---|---|---|---|---|---|
| loss | 0.477 | 0.051 | 0.290 | 0.99 | bce (+0.477) | yes |
| cat-output | 0.317 | 0.072 | 0.099 | 1.30 | linear (-0.166) | yes |
| d-steps | 0.280 | 0.069 | 0.060 | 1.07 | 5 (-0.280) | yes |
| disc-dims | 0.147 | 0.058 | 0.028 | 1.19 | 1024-1024 (-0.147) | yes |
| lr-classical | 0.125 | 0.068 | 0.036 | 0.24 | 0.001 (-0.080) | yes |
| mapper-dims | 0.124 | 0.053 | 0.026 | 0.19 | 128-128 (-0.124) | yes |
| batch-size | 0.102 | 0.015 | 0.024 | 0.49 | 128 (-0.102) | no |
| betas | 0.065 | 0.005 | 0.037 | 0.17 | 0.5-0.999 (-0.065) | no |
| n-qubits | 0.064 | 0.036 | 0.018 | 0.38 | 6 (-0.064) | no |
| input-noise-var | 0.062 | 0.008 | 0.018 | 0.29 | 0.001 (-0.062) | no |

<details><summary>all configs (2-seed means)</summary>

| config | score | f1 | avg_jsd | diff_corr | score s0 | score s1 |
|---|---|---|---|---|---|---|
| d-steps=5 | 0.223 | 0.078 | 0.039 | 0.64 | 0.206 | 0.240 |
| d-steps=2 | 0.293 | 0.107 | 0.047 | 0.82 | 0.305 | 0.280 |
| cat-output=linear | 0.337 | 0.075 | 0.119 | 0.97 | 0.357 | 0.318 |
| disc-dims=1024-1024 | 0.356 | 0.089 | 0.074 | 1.36 | 0.328 | 0.384 |
| mapper-dims=128-128 | 0.379 | 0.094 | 0.073 | 1.51 | 0.381 | 0.377 |
| batch-size=128 | 0.401 | 0.132 | 0.075 | 1.21 | 0.349 | 0.453 |
| lr-classical=0.001 | 0.423 | 0.134 | 0.063 | 1.49 | 0.384 | 0.462 |
| betas=0.5-0.999 | 0.438 | 0.142 | 0.062 | 1.53 | 0.557 | 0.319 |
| n-qubits=6 | 0.439 | 0.145 | 0.081 | 1.33 | 0.415 | 0.463 |
| input-noise-var=0.001 | 0.441 | 0.139 | 0.081 | 1.42 | 0.451 | 0.431 |
| input-noise-var=0.0 | 0.448 | 0.142 | 0.083 | 1.44 | 0.487 | 0.409 |
| n-qubits=4 | 0.450 | 0.111 | 0.091 | 1.75 | 0.398 | 0.501 |
| mapper-dims=512-512 | 0.459 | 0.102 | 0.107 | 1.81 | 0.517 | 0.401 |
| mapper-dims=1024-1024 | 0.471 | 0.107 | 0.103 | 1.88 | 0.492 | 0.449 |
| disc-dims=256-256 | 0.482 | 0.140 | 0.103 | 1.57 | 0.420 | 0.543 |
| base | 0.503 | 0.147 | 0.099 | 1.70 | 0.529 | 0.477 |
| batch-size=500 | 0.537 | 0.153 | 0.106 | 1.87 | 0.564 | 0.511 |
| lr-classical=0.0002 | 0.628 | 0.215 | 0.101 | 1.94 | 0.611 | 0.645 |
| disc-dims=128-128 | 0.638 | 0.126 | 0.127 | 2.89 | 0.643 | 0.632 |
| cat-output=softmax | 0.820 | 0.196 | 0.198 | 3.00 | 0.836 | 0.805 |
| loss=bce | 0.980 | 0.198 | 0.389 | 2.69 | 1.001 | 0.959 |

</details>

**Selected for step 3** (matters in either arm): angle-init, batch-size, betas, cat-output, d-steps, disc-dims, loss, lr-classical, lr-quantum, mapper-dims, noise-encoding, prob-scale; plus noise-encoding, angle-init, noise-scale for the quantum arm (always).
