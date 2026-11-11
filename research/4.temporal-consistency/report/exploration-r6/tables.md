# R6/R6b/R6c fixed-draw mechanism diagnosis

Clipped RGB; means over 24 conditions, not 24 independent scenes. Oracle controls are not deployable.

| Model | Intervention | Mean PSNR | Noise SD (0–255 levels) |
|---|---|---:|---:|
| official | centered_body | 19.30149 | 16.12041 |
| official | centered_head | 20.53087 | 15.51523 |
| official | debias_body | 16.32744 | 32.50692 |
| official | debias_head | 18.54834 | 14.43257 |
| official | linearize_body | 15.53317 | 19.42540 |
| official | linearize_head | 18.34168 | 12.47775 |
| official | raw | 19.39885 | 15.78449 |
| r5_plain | centered_body | 19.17034 | 14.99829 |
| r5_plain | centered_head | 20.21552 | 14.57740 |
| r5_plain | debias_body | 18.28576 | 29.48735 |
| r5_plain | debias_head | 18.19547 | 13.77120 |
| r5_plain | linearize_body | 17.72789 | 19.80159 |
| r5_plain | linearize_head | 17.60730 | 12.22234 |
| r5_plain | raw | 19.33270 | 14.47172 |
| r5_spatial_post | centered_body | 19.24476 | 14.98120 |
| r5_spatial_post | centered_head | 20.28793 | 14.55119 |
| r5_spatial_post | debias_body | 18.46414 | 28.77915 |
| r5_spatial_post | debias_head | 18.30746 | 13.72907 |
| r5_spatial_post | linearize_body | 17.90443 | 19.61852 |
| r5_spatial_post | linearize_head | 17.79591 | 12.22485 |
| r5_spatial_post | raw | 19.41235 | 14.42703 |

## Exact RGB decomposition

Cov(component, output)/Var(output); correlated quantities, may be negative.

| Model | Body linear / mean gate | Body quadratic / mean gate | Gate / mean feature | Gate-feature interaction | Combined quadratic incl interaction |
|---|---:|---:|---:|---:|---:|
| official | 0.893169 | 0.004667 | 0.104044 | 0.001010 | -0.003633 |
| r5_plain | 0.891637 | 0.005815 | 0.103633 | 0.000068 | -0.002863 |
| r5_spatial_post | 0.893345 | 0.005290 | 0.102209 | 0.000215 | -0.003289 |
