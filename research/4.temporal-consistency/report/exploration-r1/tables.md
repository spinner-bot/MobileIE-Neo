# Exploration-r1 generated tables

Synthetic pilot; averages over sequences, not independent test-set claims.

## Noisy/compressed controls

| Method | PSNR | SSIM | Core residual | Edge error | GT-relative temporal error |
| --- | ---: | ---: | ---: | ---: | ---: |
| mobileie | 18.1274 | 0.4243 | 16.4729 | 5.8770 | 16.4878 |
| frozen_global | 18.1025 | 0.4383 | 15.3725 | 5.5833 | 15.3886 |
| input_bilateral | 16.5332 | 0.5163 | 13.1624 | 4.1205 | 13.1979 |
| output_gaussian | 18.7070 | 0.5808 | 11.2483 | 3.5115 | 11.2826 |
| ema_07 | 18.3252 | 0.5611 | 6.4440 | 3.5693 | 6.4651 |
| ema_09 | 17.7150 | 0.5403 | 5.4276 | 3.5550 | 5.4479 |
| flow_ema_07 | 18.7811 | 0.6037 | 6.1172 | 3.2771 | 6.2800 |
| flow_ema_09 | 18.3930 | 0.6085 | 4.2882 | 3.0596 | 4.5706 |
| oracle_ema_07 | 19.0589 | 0.6482 | 4.0190 | 3.0940 | 4.0388 |
| oracle_ema_09 | 19.2155 | 0.6584 | 1.4573 | 2.9862 | 1.4613 |
| bilateral_flow_ema_07 | 16.7865 | 0.6196 | 4.9808 | 2.7825 | 5.1316 |
| bilateral_oracle_ema_07 | 17.0658 | 0.6744 | 3.3066 | 2.4509 | 3.3271 |

## Clean controls

| Method | PSNR | SSIM | Core residual | Edge error | GT-relative temporal error |
| --- | ---: | ---: | ---: | ---: | ---: |
| mobileie | 17.4700 | 0.8338 | 0.9908 | 1.3203 | 1.0578 |
| frozen_global | 18.5529 | 0.8456 | 0.0000 | 1.2565 | 0.0842 |
| input_bilateral | 16.9741 | 0.7965 | 1.0990 | 1.5778 | 1.1657 |
| output_gaussian | 17.4882 | 0.8066 | 0.9899 | 1.6224 | 1.0436 |
| ema_07 | 17.2297 | 0.7324 | 1.2829 | 2.0468 | 1.3404 |
| ema_09 | 16.7969 | 0.6551 | 3.1460 | 2.4723 | 3.1841 |
| flow_ema_07 | 17.2330 | 0.7500 | 1.2238 | 1.9216 | 1.3167 |
| flow_ema_09 | 16.7122 | 0.6718 | 2.5642 | 2.3821 | 2.6823 |
| oracle_ema_07 | 17.5070 | 0.8334 | 0.5947 | 1.3172 | 0.6097 |
| oracle_ema_09 | 17.6251 | 0.8308 | 0.3620 | 1.3203 | 0.3635 |
| bilateral_flow_ema_07 | 16.7251 | 0.7120 | 1.2257 | 2.1254 | 1.3158 |
| bilateral_oracle_ema_07 | 17.0225 | 0.7961 | 0.6534 | 1.5766 | 0.6674 |

## state: clean

| Method | PSNR | SSIM | Core residual | Edge error | GT-relative temporal error |
| --- | ---: | ---: | ---: | ---: | ---: |
| mobileie | 17.4700 | 0.8338 | 0.9908 | 1.3203 | 1.0578 |
| frozen_global | 18.5529 | 0.8456 | 0.0000 | 1.2565 | 0.0842 |
| attention_ema_09 | 17.6165 | 0.8392 | 0.3952 | 1.2969 | 0.4708 |
| flow_ema_07 | 17.2330 | 0.7500 | 1.2238 | 1.9216 | 1.3167 |
| frozen_flow_ema_07 | 18.0011 | 0.7582 | 0.7327 | 1.9099 | 0.8323 |
| attention_flow_ema_07 | 17.3969 | 0.7540 | 1.0295 | 1.9149 | 1.1226 |
| oracle_ema_07 | 17.5070 | 0.8334 | 0.5947 | 1.3172 | 0.6097 |
| frozen_oracle_ema_07 | 18.5719 | 0.8437 | 0.0237 | 1.2612 | 0.0525 |
| attention_oracle_ema_07 | 17.6976 | 0.8382 | 0.3543 | 1.2959 | 0.3740 |

## state: noise_jpeg

| Method | PSNR | SSIM | Core residual | Edge error | GT-relative temporal error |
| --- | ---: | ---: | ---: | ---: | ---: |
| mobileie | 18.1274 | 0.4243 | 16.4729 | 5.8770 | 16.4878 |
| frozen_global | 18.1025 | 0.4383 | 15.3725 | 5.5833 | 15.3886 |
| attention_ema_09 | 18.4605 | 0.4294 | 16.0878 | 5.7967 | 16.1022 |
| flow_ema_07 | 18.7811 | 0.6037 | 6.1172 | 3.2771 | 6.2800 |
| frozen_flow_ema_07 | 18.5114 | 0.6101 | 5.6716 | 3.1906 | 5.8263 |
| attention_flow_ema_07 | 18.9874 | 0.6079 | 5.9656 | 3.2489 | 6.1244 |
| oracle_ema_07 | 19.0589 | 0.6482 | 4.0190 | 3.0940 | 4.0388 |
| frozen_oracle_ema_07 | 18.8063 | 0.6538 | 3.6417 | 2.9940 | 3.6626 |
| attention_oracle_ema_07 | 19.3848 | 0.6526 | 3.8676 | 3.0608 | 3.8878 |

## step: clean

| Method | PSNR | SSIM | Core residual | Edge error | GT-relative temporal error |
| --- | ---: | ---: | ---: | ---: | ---: |
| mobileie | 19.3217 | 0.8204 | 2.4431 | 1.3556 | 1.6711 |
| frozen_global | 18.7951 | 0.8215 | 1.7278 | 1.4501 | 0.8949 |
| attention_ema_09 | 18.8782 | 0.8217 | 2.1942 | 1.3952 | 1.3461 |
| flow_ema_07 | 20.0350 | 0.7682 | 2.4486 | 1.7101 | 2.5403 |
| frozen_flow_ema_07 | 19.7196 | 0.7723 | 2.3730 | 1.7608 | 2.3865 |
| attention_flow_ema_07 | 19.7411 | 0.7707 | 2.4964 | 1.7311 | 2.4983 |
| oracle_ema_07 | 20.0960 | 0.8258 | 1.8616 | 1.3342 | 1.8894 |
| frozen_oracle_ema_07 | 19.6921 | 0.8267 | 1.6990 | 1.4079 | 1.6472 |
| attention_oracle_ema_07 | 19.7218 | 0.8273 | 1.8620 | 1.3646 | 1.7968 |

## step: noise_jpeg

| Method | PSNR | SSIM | Core residual | Edge error | GT-relative temporal error |
| --- | ---: | ---: | ---: | ---: | ---: |
| mobileie | 17.0834 | 0.3904 | 17.0157 | 5.4951 | 16.9094 |
| frozen_global | 15.1269 | 0.3717 | 18.0696 | 5.7751 | 17.4120 |
| attention_ema_09 | 16.3628 | 0.3805 | 17.7486 | 5.6550 | 17.1207 |
| flow_ema_07 | 18.0208 | 0.5639 | 6.4753 | 3.0877 | 7.1335 |
| frozen_flow_ema_07 | 16.0909 | 0.5453 | 6.9764 | 3.1772 | 7.3027 |
| attention_flow_ema_07 | 17.2256 | 0.5565 | 6.8004 | 3.1317 | 7.1590 |
| oracle_ema_07 | 18.1782 | 0.5989 | 4.2710 | 2.9586 | 4.8598 |
| frozen_oracle_ema_07 | 16.1371 | 0.5783 | 4.7373 | 3.0695 | 4.9623 |
| attention_oracle_ema_07 | 17.3150 | 0.5905 | 4.5729 | 3.0125 | 4.8357 |
