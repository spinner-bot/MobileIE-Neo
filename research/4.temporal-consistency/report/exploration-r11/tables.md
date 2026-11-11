# Strong-optimizer history and correspondence controls

Final1000, two seeds, all global Adam1e-3/clip1; shared teacher half-flow is not free.

## test

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---|---|---:|---:|---:|---:|
| concat_aligned_high | causal | False | half01 | 20.12716 | 0.83847 | 2.27640 | 1.94351 |
| concat_aligned_high | causal | False | raw | 20.12280 | 0.83869 | 2.40788 | 1.93679 |
| concat_aligned_high | causal | True | half01 | 20.12716 | 0.83847 | 2.27640 | 1.94351 |
| concat_aligned_high | causal | True | raw | 20.12280 | 0.83869 | 2.40788 | 1.93679 |
| concat_aligned_high | current | False | half01 | 20.12742 | 0.83845 | 2.28271 | 1.94373 |
| concat_aligned_high | current | False | raw | 20.12294 | 0.83865 | 2.41648 | 1.93731 |
| concat_aligned_high | current | True | half01 | 20.12742 | 0.83845 | 2.28271 | 1.94373 |
| concat_aligned_high | current | True | raw | 20.12294 | 0.83865 | 2.41648 | 1.93731 |
| concat_high | causal | False | half01 | 20.15614 | 0.83765 | 2.28784 | 1.95286 |
| concat_high | causal | False | raw | 20.15115 | 0.83778 | 2.42395 | 1.94686 |
| concat_high | causal | True | half01 | 20.15614 | 0.83765 | 2.28784 | 1.95286 |
| concat_high | causal | True | raw | 20.15115 | 0.83778 | 2.42395 | 1.94686 |
| concat_high | current | False | half01 | 20.13246 | 0.83729 | 2.28732 | 1.95487 |
| concat_high | current | False | raw | 20.12731 | 0.83741 | 2.42412 | 1.94905 |
| concat_high | current | True | half01 | 20.13246 | 0.83729 | 2.28732 | 1.95487 |
| concat_high | current | True | raw | 20.12731 | 0.83741 | 2.42412 | 1.94905 |
| copy_high | causal | False | half01 | 20.29789 | 0.83822 | 2.24969 | 1.97380 |
| copy_high | causal | False | raw | 20.29341 | 0.83848 | 2.37784 | 1.96697 |
| copy_high | causal | True | half01 | 20.29789 | 0.83822 | 2.24969 | 1.97380 |
| copy_high | causal | True | raw | 20.29341 | 0.83848 | 2.37784 | 1.96697 |
| plain_high | causal | False | half01 | 19.31045 | 0.83250 | 2.36124 | 1.96236 |
| plain_high | causal | False | raw | 19.30433 | 0.83263 | 2.50660 | 1.95704 |
| plain_high | causal | True | half01 | 19.31045 | 0.83250 | 2.36124 | 1.96236 |
| plain_high | causal | True | raw | 19.30433 | 0.83263 | 2.50660 | 1.95704 |

## dev

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---|---|---:|---:|---:|---:|
| concat_aligned_high | causal | False | half01 | 23.96467 | 0.89674 | 2.38588 | 1.95963 |
| concat_aligned_high | causal | False | raw | 23.98574 | 0.89852 | 2.44583 | 1.93736 |
| concat_aligned_high | causal | True | half01 | 23.96467 | 0.89674 | 2.38588 | 1.95963 |
| concat_aligned_high | causal | True | raw | 23.98574 | 0.89852 | 2.44583 | 1.93736 |
| concat_aligned_high | current | False | half01 | 23.96597 | 0.89678 | 2.38693 | 1.95943 |
| concat_aligned_high | current | False | raw | 23.98659 | 0.89853 | 2.44947 | 1.93756 |
| concat_aligned_high | current | True | half01 | 23.96597 | 0.89678 | 2.38693 | 1.95943 |
| concat_aligned_high | current | True | raw | 23.98659 | 0.89853 | 2.44947 | 1.93756 |
| concat_high | causal | False | half01 | 24.17708 | 0.89880 | 2.39662 | 1.95144 |
| concat_high | causal | False | raw | 24.19636 | 0.90053 | 2.46223 | 1.92928 |
| concat_high | causal | True | half01 | 24.17708 | 0.89880 | 2.39662 | 1.95144 |
| concat_high | causal | True | raw | 24.19636 | 0.90053 | 2.46223 | 1.92928 |
| concat_high | current | False | half01 | 24.20918 | 0.89897 | 2.39649 | 1.95206 |
| concat_high | current | False | raw | 24.22869 | 0.90069 | 2.46312 | 1.93013 |
| concat_high | current | True | half01 | 24.20918 | 0.89897 | 2.39649 | 1.95206 |
| concat_high | current | True | raw | 24.22869 | 0.90069 | 2.46312 | 1.93013 |
| copy_high | causal | False | half01 | 23.83171 | 0.89478 | 2.38184 | 1.98622 |
| copy_high | causal | False | raw | 23.85362 | 0.89661 | 2.44131 | 1.96360 |
| copy_high | causal | True | half01 | 23.83171 | 0.89478 | 2.38184 | 1.98622 |
| copy_high | causal | True | raw | 23.85362 | 0.89661 | 2.44131 | 1.96360 |
| plain_high | causal | False | half01 | 23.53687 | 0.89668 | 2.40121 | 1.96012 |
| plain_high | causal | False | raw | 23.55621 | 0.89835 | 2.46665 | 1.93934 |
| plain_high | causal | True | half01 | 23.53687 | 0.89668 | 2.40121 | 1.96012 |
| plain_high | causal | True | raw | 23.55621 | 0.89835 | 2.46665 | 1.93934 |

## dev_native

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---|---|---:|---:|---:|---:|
| concat_aligned_high | causal | False | half01 | 25.12055 | 0.90410 | 3.60308 | 1.60444 |
| concat_aligned_high | causal | False | raw | 25.07659 | 0.89679 | 4.00103 | 1.68166 |
| concat_aligned_high | causal | True | half01 | 25.12055 | 0.90410 | 3.60308 | 1.60444 |
| concat_aligned_high | causal | True | raw | 25.07659 | 0.89679 | 4.00103 | 1.68166 |
| concat_aligned_high | current | False | half01 | 25.12469 | 0.90393 | 3.61243 | 1.60616 |
| concat_aligned_high | current | False | raw | 25.07718 | 0.89650 | 4.01942 | 1.68466 |
| concat_aligned_high | current | True | half01 | 25.12469 | 0.90393 | 3.61243 | 1.60616 |
| concat_aligned_high | current | True | raw | 25.07718 | 0.89650 | 4.01942 | 1.68466 |
| concat_high | causal | False | half01 | 25.51777 | 0.90291 | 3.69143 | 1.60960 |
| concat_high | causal | False | raw | 25.46087 | 0.89524 | 4.10988 | 1.68899 |
| concat_high | causal | True | half01 | 25.51777 | 0.90291 | 3.69143 | 1.60960 |
| concat_high | causal | True | raw | 25.46087 | 0.89524 | 4.10988 | 1.68899 |
| concat_high | current | False | half01 | 25.56492 | 0.90325 | 3.68742 | 1.60799 |
| concat_high | current | False | raw | 25.50696 | 0.89558 | 4.10690 | 1.68744 |
| concat_high | current | True | half01 | 25.56492 | 0.90325 | 3.68742 | 1.60799 |
| concat_high | current | True | raw | 25.50696 | 0.89558 | 4.10690 | 1.68744 |
| copy_high | causal | False | half01 | 25.52364 | 0.90442 | 3.58822 | 1.59254 |
| copy_high | causal | False | raw | 25.47741 | 0.89717 | 3.98661 | 1.66825 |
| copy_high | causal | True | half01 | 25.52364 | 0.90442 | 3.58822 | 1.59254 |
| copy_high | causal | True | raw | 25.47741 | 0.89717 | 3.98661 | 1.66825 |
| plain_high | causal | False | half01 | 25.17115 | 0.90365 | 3.63861 | 1.60651 |
| plain_high | causal | False | raw | 25.11754 | 0.89606 | 4.05308 | 1.68758 |
| plain_high | causal | True | half01 | 25.17115 | 0.90365 | 3.63861 | 1.60651 |
| plain_high | causal | True | raw | 25.11754 | 0.89606 | 4.05308 | 1.68758 |

## legacy_synthetic

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---|---|---:|---:|---:|---:|
| concat_aligned_high | causal | False | half01 | 20.70019 | 0.63324 | 10.60436 | 5.06746 |
| concat_aligned_high | causal | False | raw | 20.50967 | 0.60820 | 11.92146 | 5.39253 |
| concat_aligned_high | causal | True | half01 | 20.70019 | 0.63324 | 10.60436 | 5.06746 |
| concat_aligned_high | causal | True | raw | 20.50967 | 0.60820 | 11.92146 | 5.39253 |
| concat_aligned_high | current | False | half01 | 20.69283 | 0.63252 | 10.66348 | 5.07576 |
| concat_aligned_high | current | False | raw | 20.49970 | 0.60732 | 11.99501 | 5.40325 |
| concat_aligned_high | current | True | half01 | 20.69283 | 0.63252 | 10.66348 | 5.07576 |
| concat_aligned_high | current | True | raw | 20.49970 | 0.60732 | 11.99501 | 5.40325 |
| concat_high | causal | False | half01 | 20.68477 | 0.63606 | 10.55149 | 5.03234 |
| concat_high | causal | False | raw | 20.48846 | 0.61098 | 11.86630 | 5.35280 |
| concat_high | causal | True | half01 | 20.68477 | 0.63606 | 10.55149 | 5.03234 |
| concat_high | causal | True | raw | 20.48846 | 0.61098 | 11.86630 | 5.35280 |
| concat_high | current | False | half01 | 20.67489 | 0.63565 | 10.58932 | 5.03598 |
| concat_high | current | False | raw | 20.47722 | 0.61051 | 11.91219 | 5.35730 |
| concat_high | current | True | half01 | 20.67489 | 0.63565 | 10.58932 | 5.03598 |
| concat_high | current | True | raw | 20.47722 | 0.61051 | 11.91219 | 5.35730 |
| copy_high | causal | False | half01 | 20.25592 | 0.63634 | 10.38425 | 5.01069 |
| copy_high | causal | False | raw | 20.08676 | 0.61154 | 11.67264 | 5.32312 |
| copy_high | causal | True | half01 | 20.25592 | 0.63634 | 10.38425 | 5.01069 |
| copy_high | causal | True | raw | 20.08676 | 0.61154 | 11.67264 | 5.32312 |
| plain_high | causal | False | half01 | 20.66436 | 0.63087 | 10.71417 | 5.09433 |
| plain_high | causal | False | raw | 20.47050 | 0.60558 | 12.05179 | 5.42512 |
| plain_high | causal | True | half01 | 20.66436 | 0.63087 | 10.71417 | 5.09433 |
| plain_high | causal | True | raw | 20.47050 | 0.60558 | 12.05179 | 5.42512 |

## event_cut

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---|---|---:|---:|---:|---:|
| concat_aligned_high | causal | False | half01 | 22.93890 | 0.57580 | 12.09475 | 5.20726 |
| concat_aligned_high | causal | False | raw | 22.64462 | 0.54938 | 13.64686 | 5.58260 |
| concat_aligned_high | causal | True | half01 | 22.96075 | 0.57573 | 12.07879 | 5.21056 |
| concat_aligned_high | causal | True | raw | 22.64662 | 0.54939 | 13.64482 | 5.58230 |
| concat_aligned_high | current | False | half01 | 22.93689 | 0.57517 | 12.14116 | 5.21450 |
| concat_aligned_high | current | False | raw | 22.63900 | 0.54866 | 13.70837 | 5.59201 |
| concat_aligned_high | current | True | half01 | 22.95599 | 0.57509 | 12.13059 | 5.21819 |
| concat_aligned_high | current | True | raw | 22.63900 | 0.54866 | 13.70837 | 5.59201 |
| concat_high | causal | False | half01 | 22.91790 | 0.57047 | 12.28956 | 5.25387 |
| concat_high | causal | False | raw | 22.61070 | 0.54397 | 13.87300 | 5.63395 |
| concat_high | causal | True | half01 | 22.93798 | 0.57037 | 12.27299 | 5.25707 |
| concat_high | causal | True | raw | 22.61215 | 0.54398 | 13.86862 | 5.63314 |
| concat_high | current | False | half01 | 22.89382 | 0.57007 | 12.31966 | 5.25626 |
| concat_high | current | False | raw | 22.58679 | 0.54358 | 13.91196 | 5.63689 |
| concat_high | current | True | half01 | 22.91264 | 0.56998 | 12.30986 | 5.26012 |
| concat_high | current | True | raw | 22.58679 | 0.54358 | 13.91196 | 5.63689 |
| copy_high | causal | False | half01 | 22.69577 | 0.57612 | 11.97860 | 5.19439 |
| copy_high | causal | False | raw | 22.41869 | 0.54951 | 13.52557 | 5.56588 |
| copy_high | causal | True | half01 | 22.71559 | 0.57604 | 11.96896 | 5.19800 |
| copy_high | causal | True | raw | 22.41869 | 0.54951 | 13.52557 | 5.56588 |
| plain_high | causal | False | half01 | 22.85531 | 0.57314 | 12.24364 | 5.24718 |
| plain_high | causal | False | raw | 22.55646 | 0.54661 | 13.82522 | 5.62917 |
| plain_high | causal | True | half01 | 22.87457 | 0.57305 | 12.23384 | 5.25110 |
| plain_high | causal | True | raw | 22.55646 | 0.54661 | 13.82522 | 5.62917 |

## event_light_step

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |
|---|---|---|---|---:|---:|---:|---:|
| concat_aligned_high | causal | False | half01 | 23.89290 | 0.57669 | 11.80896 | 4.78303 |
| concat_aligned_high | causal | False | raw | 23.56747 | 0.54944 | 13.17524 | 5.13710 |
| concat_aligned_high | causal | True | half01 | 23.92785 | 0.57587 | 11.74300 | 4.79638 |
| concat_aligned_high | causal | True | raw | 23.58009 | 0.54952 | 13.14472 | 5.13610 |
| concat_aligned_high | current | False | half01 | 23.90105 | 0.57616 | 11.80836 | 4.78925 |
| concat_aligned_high | current | False | raw | 23.57126 | 0.54879 | 13.19753 | 5.14544 |
| concat_aligned_high | current | True | half01 | 23.92183 | 0.57524 | 11.78586 | 4.80367 |
| concat_aligned_high | current | True | raw | 23.57126 | 0.54879 | 13.19753 | 5.14544 |
| concat_high | causal | False | half01 | 23.73333 | 0.57032 | 12.04735 | 4.84008 |
| concat_high | causal | False | raw | 23.39620 | 0.54307 | 13.44834 | 5.20031 |
| concat_high | causal | True | half01 | 23.76110 | 0.56949 | 11.98681 | 4.85343 |
| concat_high | causal | True | raw | 23.40756 | 0.54316 | 13.41865 | 5.19899 |
| concat_high | current | False | half01 | 23.72768 | 0.57005 | 12.03833 | 4.84237 |
| concat_high | current | False | raw | 23.38891 | 0.54278 | 13.45644 | 5.20329 |
| concat_high | current | True | half01 | 23.74271 | 0.56911 | 12.01920 | 4.85703 |
| concat_high | current | True | raw | 23.38891 | 0.54278 | 13.45644 | 5.20329 |
| copy_high | causal | False | half01 | 23.82718 | 0.57551 | 11.83910 | 4.80761 |
| copy_high | causal | False | raw | 23.49781 | 0.54797 | 13.22495 | 5.16383 |
| copy_high | causal | True | half01 | 23.84697 | 0.57458 | 11.81354 | 4.82186 |
| copy_high | causal | True | raw | 23.49781 | 0.54797 | 13.22495 | 5.16383 |
| plain_high | causal | False | half01 | 23.71099 | 0.57092 | 12.09106 | 4.87108 |
| plain_high | causal | False | raw | 23.37792 | 0.54345 | 13.51084 | 5.23668 |
| plain_high | causal | True | half01 | 23.72908 | 0.56999 | 12.06815 | 4.88581 |
| plain_high | causal | True | raw | 23.37792 | 0.54345 | 13.51084 | 5.23668 |

