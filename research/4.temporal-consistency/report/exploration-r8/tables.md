# Raw-pair head and static optimizer controls

Final1000 updates, two seeds. Spatial copy is foldable; causal-current isolates actual history at the same weights.

## test

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal |
|---|---|---|---|---:|---:|---:|
| plain | causal | False | half01 | 16.50809 | 0.80581 | 2.27994 |
| plain | causal | False | raw | 16.50662 | 0.80617 | 2.42026 |
| plain | causal | True | half01 | 16.50809 | 0.80581 | 2.27994 |
| plain | causal | True | raw | 16.50662 | 0.80617 | 2.42026 |
| spatial_copy | causal | False | half01 | 16.97573 | 0.80381 | 2.29778 |
| spatial_copy | causal | False | raw | 16.97564 | 0.80424 | 2.43523 |
| spatial_copy | causal | True | half01 | 16.97573 | 0.80381 | 2.29778 |
| spatial_copy | causal | True | raw | 16.97564 | 0.80424 | 2.43523 |
| spatial_detail | causal | False | half01 | 16.48573 | 0.80409 | 2.26840 |
| spatial_detail | causal | False | raw | 16.48477 | 0.80448 | 2.40509 |
| spatial_detail | causal | True | half01 | 16.48573 | 0.80409 | 2.26840 |
| spatial_detail | causal | True | raw | 16.48477 | 0.80448 | 2.40509 |
| temporal_concat | causal | False | half01 | 16.96858 | 0.80443 | 2.32629 |
| temporal_concat | causal | False | raw | 16.96781 | 0.80484 | 2.46520 |
| temporal_concat | causal | True | half01 | 16.96858 | 0.80443 | 2.32629 |
| temporal_concat | causal | True | raw | 16.96781 | 0.80484 | 2.46520 |
| temporal_concat | current | False | half01 | 17.00263 | 0.80411 | 2.29847 |
| temporal_concat | current | False | raw | 17.00223 | 0.80454 | 2.43535 |
| temporal_concat | current | True | half01 | 17.00263 | 0.80411 | 2.29847 |
| temporal_concat | current | True | raw | 17.00223 | 0.80454 | 2.43535 |
| temporal_delta | causal | False | half01 | 16.34433 | 0.80428 | 2.26973 |
| temporal_delta | causal | False | raw | 16.34233 | 0.80465 | 2.40823 |
| temporal_delta | causal | True | half01 | 16.34433 | 0.80428 | 2.26973 |
| temporal_delta | causal | True | raw | 16.34233 | 0.80465 | 2.40823 |
| temporal_delta | current | False | half01 | 16.40641 | 0.80472 | 2.27879 |
| temporal_delta | current | False | raw | 16.40493 | 0.80509 | 2.41896 |
| temporal_delta | current | True | half01 | 16.40641 | 0.80472 | 2.27879 |
| temporal_delta | current | True | raw | 16.40493 | 0.80509 | 2.41896 |
| temporal_delta_aug | causal | False | half01 | 16.39750 | 0.80487 | 2.26981 |
| temporal_delta_aug | causal | False | raw | 16.39572 | 0.80523 | 2.40912 |
| temporal_delta_aug | causal | True | half01 | 16.39750 | 0.80487 | 2.26981 |
| temporal_delta_aug | causal | True | raw | 16.39572 | 0.80523 | 2.40912 |
| temporal_delta_aug | current | False | half01 | 16.45114 | 0.80470 | 2.27698 |
| temporal_delta_aug | current | False | raw | 16.44972 | 0.80507 | 2.41666 |
| temporal_delta_aug | current | True | half01 | 16.45114 | 0.80470 | 2.27698 |
| temporal_delta_aug | current | True | raw | 16.44972 | 0.80507 | 2.41666 |

## legacy_synthetic

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal |
|---|---|---|---|---:|---:|---:|
| plain | causal | False | half01 | 17.32660 | 0.52993 | 14.08015 |
| plain | causal | False | raw | 17.11115 | 0.50325 | 15.86217 |
| plain | causal | True | half01 | 17.32660 | 0.52993 | 14.08015 |
| plain | causal | True | raw | 17.11115 | 0.50325 | 15.86217 |
| spatial_copy | causal | False | half01 | 18.80711 | 0.54455 | 14.36738 |
| spatial_copy | causal | False | raw | 18.49855 | 0.51623 | 16.20968 |
| spatial_copy | causal | True | half01 | 18.80711 | 0.54455 | 14.36738 |
| spatial_copy | causal | True | raw | 18.49855 | 0.51623 | 16.20968 |
| spatial_detail | causal | False | half01 | 17.51023 | 0.53363 | 14.01014 |
| spatial_detail | causal | False | raw | 17.28594 | 0.50679 | 15.78488 |
| spatial_detail | causal | True | half01 | 17.51023 | 0.53363 | 14.01014 |
| spatial_detail | causal | True | raw | 17.28594 | 0.50679 | 15.78488 |
| temporal_concat | causal | False | half01 | 18.89867 | 0.54481 | 14.47931 |
| temporal_concat | causal | False | raw | 18.58339 | 0.51626 | 16.34628 |
| temporal_concat | causal | True | half01 | 18.89867 | 0.54481 | 14.47931 |
| temporal_concat | causal | True | raw | 18.58339 | 0.51626 | 16.34628 |
| temporal_concat | current | False | half01 | 18.87845 | 0.54543 | 14.35089 |
| temporal_concat | current | False | raw | 18.56674 | 0.51706 | 16.19133 |
| temporal_concat | current | True | half01 | 18.87845 | 0.54543 | 14.35089 |
| temporal_concat | current | True | raw | 18.56674 | 0.51706 | 16.19133 |
| temporal_delta | causal | False | half01 | 17.42664 | 0.53200 | 14.04711 |
| temporal_delta | causal | False | raw | 17.20983 | 0.50516 | 15.82446 |
| temporal_delta | causal | True | half01 | 17.42664 | 0.53200 | 14.04711 |
| temporal_delta | causal | True | raw | 17.20983 | 0.50516 | 15.82446 |
| temporal_delta | current | False | half01 | 17.33802 | 0.53131 | 14.03215 |
| temporal_delta | current | False | raw | 17.12320 | 0.50460 | 15.80810 |
| temporal_delta | current | True | half01 | 17.33802 | 0.53131 | 14.03215 |
| temporal_delta | current | True | raw | 17.12320 | 0.50460 | 15.80810 |
| temporal_delta_aug | causal | False | half01 | 17.43607 | 0.53168 | 14.09935 |
| temporal_delta_aug | causal | False | raw | 17.21719 | 0.50481 | 15.88707 |
| temporal_delta_aug | causal | True | half01 | 17.43607 | 0.53168 | 14.09935 |
| temporal_delta_aug | causal | True | raw | 17.21719 | 0.50481 | 15.88707 |
| temporal_delta_aug | current | False | half01 | 17.33636 | 0.53129 | 14.02999 |
| temporal_delta_aug | current | False | raw | 17.12188 | 0.50458 | 15.80555 |
| temporal_delta_aug | current | True | half01 | 17.33636 | 0.53129 | 14.02999 |
| temporal_delta_aug | current | True | raw | 17.12188 | 0.50458 | 15.80555 |

## event_light_step

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal |
|---|---|---|---|---:|---:|---:|
| plain | causal | False | half01 | 21.35800 | 0.52004 | 14.78459 |
| plain | causal | False | raw | 21.02588 | 0.49092 | 16.49432 |
| plain | causal | True | half01 | 21.35172 | 0.51900 | 14.79947 |
| plain | causal | True | raw | 21.02588 | 0.49092 | 16.49432 |
| spatial_copy | causal | False | half01 | 21.60287 | 0.51915 | 14.73572 |
| spatial_copy | causal | False | raw | 21.26937 | 0.48949 | 16.45104 |
| spatial_copy | causal | True | half01 | 21.60073 | 0.51809 | 14.74034 |
| spatial_copy | causal | True | raw | 21.26937 | 0.48949 | 16.45104 |
| spatial_detail | causal | False | half01 | 21.22116 | 0.51830 | 14.85925 |
| spatial_detail | causal | False | raw | 20.89866 | 0.48914 | 16.57704 |
| spatial_detail | causal | True | half01 | 21.21445 | 0.51726 | 14.87375 |
| spatial_detail | causal | True | raw | 20.89866 | 0.48914 | 16.57704 |
| temporal_concat | causal | False | half01 | 21.70909 | 0.51795 | 15.05820 |
| temporal_concat | causal | False | raw | 21.36311 | 0.48813 | 16.81504 |
| temporal_concat | causal | True | half01 | 21.73371 | 0.51699 | 14.90734 |
| temporal_concat | causal | True | raw | 21.39061 | 0.48821 | 16.64843 |
| temporal_concat | current | False | half01 | 21.59001 | 0.51939 | 14.72978 |
| temporal_concat | current | False | raw | 21.25790 | 0.48971 | 16.44453 |
| temporal_concat | current | True | half01 | 21.58772 | 0.51833 | 14.73447 |
| temporal_concat | current | True | raw | 21.25790 | 0.48971 | 16.44453 |
| temporal_delta | causal | False | half01 | 21.20751 | 0.51669 | 14.95209 |
| temporal_delta | causal | False | raw | 20.88495 | 0.48761 | 16.67564 |
| temporal_delta | causal | True | half01 | 21.22017 | 0.51575 | 14.88825 |
| temporal_delta | causal | True | raw | 20.90282 | 0.48769 | 16.59717 |
| temporal_delta | current | False | half01 | 21.27925 | 0.51932 | 14.81406 |
| temporal_delta | current | False | raw | 20.95198 | 0.49015 | 16.52725 |
| temporal_delta | current | True | half01 | 21.27232 | 0.51828 | 14.82962 |
| temporal_delta | current | True | raw | 20.95198 | 0.49015 | 16.52725 |
| temporal_delta_aug | causal | False | half01 | 21.14345 | 0.51551 | 14.94038 |
| temporal_delta_aug | causal | False | raw | 20.82154 | 0.48643 | 16.67293 |
| temporal_delta_aug | causal | True | half01 | 21.14465 | 0.51450 | 14.94067 |
| temporal_delta_aug | causal | True | raw | 20.82927 | 0.48646 | 16.65763 |
| temporal_delta_aug | current | False | half01 | 21.28695 | 0.51958 | 14.80027 |
| temporal_delta_aug | current | False | raw | 20.95990 | 0.49042 | 16.51145 |
| temporal_delta_aug | current | True | half01 | 21.28015 | 0.51854 | 14.81556 |
| temporal_delta_aug | current | True | raw | 20.95990 | 0.49042 | 16.51145 |

## event_cut

| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal |
|---|---|---|---|---:|---:|---:|
| plain | causal | False | half01 | 19.77099 | 0.52136 | 14.44493 |
| plain | causal | False | raw | 19.49200 | 0.49414 | 16.31666 |
| plain | causal | True | half01 | 19.77549 | 0.52108 | 14.44410 |
| plain | causal | True | raw | 19.49200 | 0.49414 | 16.31666 |
| spatial_copy | causal | False | half01 | 20.54298 | 0.52678 | 14.51874 |
| spatial_copy | causal | False | raw | 20.24479 | 0.49846 | 16.40490 |
| spatial_copy | causal | True | half01 | 20.55199 | 0.52653 | 14.51847 |
| spatial_copy | causal | True | raw | 20.24479 | 0.49846 | 16.40490 |
| spatial_detail | causal | False | half01 | 19.58605 | 0.51975 | 14.46557 |
| spatial_detail | causal | False | raw | 19.32101 | 0.49252 | 16.34568 |
| spatial_detail | causal | True | half01 | 19.59033 | 0.51946 | 14.46639 |
| spatial_detail | causal | True | raw | 19.32101 | 0.49252 | 16.34568 |
| temporal_concat | causal | False | half01 | 20.79639 | 0.52669 | 14.71825 |
| temporal_concat | causal | False | raw | 20.47343 | 0.49800 | 16.65275 |
| temporal_concat | causal | True | half01 | 20.81531 | 0.52656 | 14.68388 |
| temporal_concat | causal | True | raw | 20.49180 | 0.49827 | 16.60128 |
| temporal_concat | current | False | half01 | 20.57262 | 0.52750 | 14.49710 |
| temporal_concat | current | False | raw | 20.27411 | 0.49914 | 16.38045 |
| temporal_concat | current | True | half01 | 20.58158 | 0.52725 | 14.49662 |
| temporal_concat | current | True | raw | 20.27411 | 0.49914 | 16.38045 |
| temporal_delta | causal | False | half01 | 19.55545 | 0.51728 | 14.51174 |
| temporal_delta | causal | False | raw | 19.28996 | 0.49018 | 16.38925 |
| temporal_delta | causal | True | half01 | 19.56242 | 0.51706 | 14.50750 |
| temporal_delta | causal | True | raw | 19.29270 | 0.49025 | 16.38394 |
| temporal_delta | current | False | half01 | 19.79940 | 0.52210 | 14.42581 |
| temporal_delta | current | False | raw | 19.51999 | 0.49478 | 16.29586 |
| temporal_delta | current | True | half01 | 19.80407 | 0.52181 | 14.42501 |
| temporal_delta | current | True | raw | 19.51999 | 0.49478 | 16.29586 |
| temporal_delta_aug | causal | False | half01 | 19.47530 | 0.51556 | 14.57353 |
| temporal_delta_aug | causal | False | raw | 19.21127 | 0.48849 | 16.46460 |
| temporal_delta_aug | causal | True | half01 | 19.47583 | 0.51527 | 14.57696 |
| temporal_delta_aug | causal | True | raw | 19.20805 | 0.48847 | 16.46435 |
| temporal_delta_aug | current | False | half01 | 19.79555 | 0.52211 | 14.42058 |
| temporal_delta_aug | current | False | raw | 19.51684 | 0.49481 | 16.28990 |
| temporal_delta_aug | current | True | half01 | 19.80028 | 0.52183 | 14.41982 |
| temporal_delta_aug | current | True | raw | 19.51684 | 0.49481 | 16.28990 |

