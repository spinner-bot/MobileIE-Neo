# Native-scale and matched-field input smoothing

Two-seed means; all frozen candidates, all four project-dev clips.

Full-area and crop have different field of view/content/context; do not interpret their metric gap as pure noise.

## native_crop

| Variant | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---:|---:|---:|---:|
| full_lr1e3 | half01 | 25.17115 | 0.90365 | 3.63861 | 1.60651 |
| full_lr1e3 | raw | 25.11754 | 0.89606 | 4.05308 | 1.68758 |
| full_lr1e4 | half01 | 22.26394 | 0.88836 | 3.66936 | 1.68786 |
| full_lr1e4 | raw | 22.23915 | 0.87941 | 4.07960 | 1.77515 |
| head_lr1e3 | half01 | 19.39375 | 0.86379 | 3.85961 | 1.74491 |
| head_lr1e3 | raw | 19.38644 | 0.85401 | 4.27686 | 1.83316 |
| plain | half01 | 19.05667 | 0.86114 | 3.90593 | 1.77600 |
| plain | raw | 19.04548 | 0.85111 | 4.34354 | 1.86986 |
| pretrained | half01 | 18.50496 | 0.85719 | 3.94670 | 1.71225 |
| pretrained | raw | 18.49637 | 0.84784 | 4.37905 | 1.79693 |
| spatial_copy | half01 | 19.73714 | 0.86292 | 3.89931 | 1.75497 |
| spatial_copy | raw | 19.72764 | 0.85285 | 4.32369 | 1.84565 |

## smooth_crop

| Variant | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---:|---:|---:|---:|
| full_lr1e3 | half01 | 22.07213 | 0.82295 | 2.98448 | 2.20259 |
| full_lr1e3 | raw | 22.07285 | 0.82185 | 3.22595 | 2.21332 |
| full_lr1e4 | half01 | 20.77598 | 0.82611 | 2.85758 | 2.15177 |
| full_lr1e4 | raw | 20.78103 | 0.82527 | 3.06611 | 2.16066 |
| head_lr1e3 | half01 | 17.94183 | 0.80256 | 2.92785 | 2.19050 |
| head_lr1e3 | raw | 17.94967 | 0.80190 | 3.12168 | 2.19810 |
| plain | half01 | 17.31089 | 0.80595 | 2.96468 | 2.16611 |
| plain | raw | 17.31595 | 0.80511 | 3.17804 | 2.17512 |
| pretrained | half01 | 17.07506 | 0.79730 | 3.06568 | 2.17729 |
| pretrained | raw | 17.08028 | 0.79646 | 3.28556 | 2.18458 |
| spatial_copy | half01 | 18.18151 | 0.80361 | 2.94078 | 2.18622 |
| spatial_copy | raw | 18.18890 | 0.80291 | 3.13803 | 2.19402 |

## full_area

| Variant | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---:|---:|---:|---:|
| full_lr1e3 | half01 | 23.51949 | 0.90106 | 2.30478 | 1.89440 |
| full_lr1e3 | raw | 23.53971 | 0.90266 | 2.37088 | 1.87308 |
| full_lr1e4 | half01 | 21.63405 | 0.89381 | 2.24828 | 1.89554 |
| full_lr1e4 | raw | 21.65718 | 0.89528 | 2.31158 | 1.87833 |
| head_lr1e3 | half01 | 20.23276 | 0.87875 | 2.30078 | 1.89249 |
| head_lr1e3 | raw | 20.25611 | 0.88013 | 2.37478 | 1.87648 |
| plain | half01 | 19.31657 | 0.87194 | 2.30340 | 1.91111 |
| plain | raw | 19.33569 | 0.87320 | 2.38078 | 1.89811 |
| pretrained | half01 | 19.19823 | 0.86717 | 2.36503 | 1.88841 |
| pretrained | raw | 19.21963 | 0.86844 | 2.45118 | 1.87199 |
| spatial_copy | half01 | 20.34879 | 0.88029 | 2.30667 | 1.87833 |
| spatial_copy | raw | 20.37166 | 0.88163 | 2.38227 | 1.86304 |

