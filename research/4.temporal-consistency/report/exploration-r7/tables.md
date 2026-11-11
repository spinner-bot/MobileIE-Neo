# Frozen channel/placement and input-reset controls

Untrained official weights; reused exploratory inputs. Low temporal error is better. Static proxy uses train images only.

## test

| Method | Reset | Fusion | PSNR | SSIM | Temporal |
|---|---|---|---:|---:|---:|
| body_post_first3_g05 | False | half01 | 14.80894 | 0.78741 | 2.46961 |
| body_post_first3_g05 | False | raw | 14.80420 | 0.78763 | 2.63889 |
| body_post_first3_g05 | True | half01 | 14.80894 | 0.78741 | 2.46961 |
| body_post_first3_g05 | True | raw | 14.80420 | 0.78763 | 2.63889 |
| body_post_selected_g05 | False | half01 | 14.73411 | 0.74301 | 2.51922 |
| body_post_selected_g05 | False | raw | 14.73105 | 0.74310 | 2.64601 |
| body_post_selected_g05 | True | half01 | 14.73411 | 0.74301 | 2.51922 |
| body_post_selected_g05 | True | raw | 14.73105 | 0.74310 | 2.64601 |
| body_post_selected_g1 | False | half01 | 14.55223 | 0.66731 | 3.09678 |
| body_post_selected_g1 | False | raw | 14.54461 | 0.66515 | 3.27902 |
| body_post_selected_g1 | True | half01 | 14.55223 | 0.66731 | 3.09678 |
| body_post_selected_g1 | True | raw | 14.54461 | 0.66515 | 3.27902 |
| body_post_uniform_g05 | False | half01 | 14.74267 | 0.74996 | 2.55152 |
| body_post_uniform_g05 | False | raw | 14.73958 | 0.74999 | 2.69848 |
| body_post_uniform_g05 | True | half01 | 14.74267 | 0.74996 | 2.55152 |
| body_post_uniform_g05 | True | raw | 14.73958 | 0.74999 | 2.69848 |
| body_pre_first3_g05 | False | half01 | 15.25678 | 0.78824 | 2.72201 |
| body_pre_first3_g05 | False | raw | 15.25348 | 0.78848 | 2.93347 |
| body_pre_first3_g05 | True | half01 | 15.25678 | 0.78824 | 2.72201 |
| body_pre_first3_g05 | True | raw | 15.25348 | 0.78848 | 2.93347 |
| body_pre_selected_g05 | False | half01 | 14.73756 | 0.74298 | 2.51909 |
| body_pre_selected_g05 | False | raw | 14.73451 | 0.74307 | 2.64607 |
| body_pre_selected_g05 | True | half01 | 14.73756 | 0.74298 | 2.51909 |
| body_pre_selected_g05 | True | raw | 14.73451 | 0.74307 | 2.64607 |
| body_pre_selected_g1 | False | half01 | 14.55223 | 0.66731 | 3.09678 |
| body_pre_selected_g1 | False | raw | 14.54461 | 0.66515 | 3.27902 |
| body_pre_selected_g1 | True | half01 | 14.55223 | 0.66731 | 3.09678 |
| body_pre_selected_g1 | True | raw | 14.54461 | 0.66515 | 3.27902 |
| body_pre_uniform_g05 | False | half01 | 14.74611 | 0.75008 | 2.55240 |
| body_pre_uniform_g05 | False | raw | 14.74301 | 0.75011 | 2.69983 |
| body_pre_uniform_g05 | True | half01 | 14.74611 | 0.75008 | 2.55240 |
| body_pre_uniform_g05 | True | raw | 14.74301 | 0.75011 | 2.69983 |
| pretrained | False | half01 | 14.83036 | 0.78741 | 2.37453 |
| pretrained | False | raw | 14.82822 | 0.78768 | 2.53442 |
| pretrained | True | half01 | 14.83036 | 0.78741 | 2.37453 |
| pretrained | True | raw | 14.82822 | 0.78768 | 2.53442 |

## legacy_synthetic

| Method | Reset | Fusion | PSNR | SSIM | Temporal |
|---|---|---|---:|---:|---:|
| body_post_first3_g05 | False | half01 | 17.61054 | 0.53194 | 14.52301 |
| body_post_first3_g05 | False | raw | 17.31973 | 0.50502 | 16.39427 |
| body_post_first3_g05 | True | half01 | 17.61054 | 0.53194 | 14.52301 |
| body_post_first3_g05 | True | raw | 17.31973 | 0.50502 | 16.39427 |
| body_post_selected_g05 | False | half01 | 18.17962 | 0.58356 | 10.29358 |
| body_post_selected_g05 | False | raw | 18.01759 | 0.56328 | 11.39279 |
| body_post_selected_g05 | True | half01 | 18.17962 | 0.58356 | 10.29358 |
| body_post_selected_g05 | True | raw | 18.01759 | 0.56328 | 11.39279 |
| body_post_selected_g1 | False | half01 | 17.15863 | 0.47137 | 13.67944 |
| body_post_selected_g1 | False | raw | 16.92588 | 0.44642 | 15.30397 |
| body_post_selected_g1 | True | half01 | 17.15863 | 0.47137 | 13.67944 |
| body_post_selected_g1 | True | raw | 16.92588 | 0.44642 | 15.30397 |
| body_post_uniform_g05 | False | half01 | 18.19378 | 0.58717 | 10.50025 |
| body_post_uniform_g05 | False | raw | 18.02067 | 0.56675 | 11.66072 |
| body_post_uniform_g05 | True | half01 | 18.19378 | 0.58717 | 10.50025 |
| body_post_uniform_g05 | True | raw | 18.02067 | 0.56675 | 11.66072 |
| body_pre_first3_g05 | False | half01 | 14.15208 | 0.51204 | 13.09588 |
| body_pre_first3_g05 | False | raw | 14.02753 | 0.48922 | 14.74728 |
| body_pre_first3_g05 | True | half01 | 14.15208 | 0.51204 | 13.09588 |
| body_pre_first3_g05 | True | raw | 14.02753 | 0.48922 | 14.74728 |
| body_pre_selected_g05 | False | half01 | 18.07110 | 0.58100 | 10.38379 |
| body_pre_selected_g05 | False | raw | 17.90830 | 0.56055 | 11.49444 |
| body_pre_selected_g05 | True | half01 | 18.07110 | 0.58100 | 10.38379 |
| body_pre_selected_g05 | True | raw | 17.90830 | 0.56055 | 11.49444 |
| body_pre_selected_g1 | False | half01 | 17.15863 | 0.47137 | 13.67944 |
| body_pre_selected_g1 | False | raw | 16.92588 | 0.44642 | 15.30397 |
| body_pre_selected_g1 | True | half01 | 17.15863 | 0.47137 | 13.67944 |
| body_pre_selected_g1 | True | raw | 16.92588 | 0.44642 | 15.30397 |
| body_pre_uniform_g05 | False | half01 | 18.02071 | 0.58164 | 10.66467 |
| body_pre_uniform_g05 | False | raw | 17.84710 | 0.56093 | 11.84641 |
| body_pre_uniform_g05 | True | half01 | 18.02071 | 0.58164 | 10.66467 |
| body_pre_uniform_g05 | True | raw | 17.84710 | 0.56093 | 11.84641 |
| pretrained | False | half01 | 17.62392 | 0.53197 | 14.35815 |
| pretrained | False | raw | 17.32977 | 0.50505 | 16.18218 |
| pretrained | True | half01 | 17.62392 | 0.53197 | 14.35815 |
| pretrained | True | raw | 17.32977 | 0.50505 | 16.18218 |

## event_light_step

| Method | Reset | Fusion | PSNR | SSIM | Temporal |
|---|---|---|---:|---:|---:|
| body_post_first3_g05 | False | half01 | 18.24581 | 0.47826 | 17.18434 |
| body_post_first3_g05 | False | raw | 18.03385 | 0.44933 | 19.21141 |
| body_post_first3_g05 | True | half01 | 18.33642 | 0.47832 | 16.17024 |
| body_post_first3_g05 | True | raw | 18.14407 | 0.45071 | 17.99973 |
| body_post_selected_g05 | False | half01 | 18.30912 | 0.52656 | 12.59382 |
| body_post_selected_g05 | False | raw | 18.20107 | 0.49998 | 13.91466 |
| body_post_selected_g05 | True | half01 | 18.31034 | 0.52329 | 12.34217 |
| body_post_selected_g05 | True | raw | 18.21198 | 0.49753 | 13.59791 |
| body_post_selected_g1 | False | half01 | 17.37302 | 0.40518 | 15.29590 |
| body_post_selected_g1 | False | raw | 17.23626 | 0.37958 | 17.03498 |
| body_post_selected_g1 | True | half01 | 17.46492 | 0.40702 | 14.39898 |
| body_post_selected_g1 | True | raw | 17.33447 | 0.38198 | 15.95367 |
| body_post_uniform_g05 | False | half01 | 18.26945 | 0.52483 | 13.29261 |
| body_post_uniform_g05 | False | raw | 18.14745 | 0.49753 | 14.75024 |
| body_post_uniform_g05 | True | half01 | 18.28129 | 0.52135 | 13.02715 |
| body_post_uniform_g05 | True | raw | 18.17026 | 0.49495 | 14.40945 |
| body_pre_first3_g05 | False | half01 | 18.58819 | 0.48784 | 15.49917 |
| body_pre_first3_g05 | False | raw | 18.37925 | 0.45979 | 17.31055 |
| body_pre_first3_g05 | True | half01 | 18.71992 | 0.48992 | 14.85721 |
| body_pre_first3_g05 | True | raw | 18.51668 | 0.46263 | 16.54891 |
| body_pre_selected_g05 | False | half01 | 18.35751 | 0.52492 | 12.74180 |
| body_pre_selected_g05 | False | raw | 18.24710 | 0.49832 | 14.08419 |
| body_pre_selected_g05 | True | half01 | 18.36974 | 0.52191 | 12.42486 |
| body_pre_selected_g05 | True | raw | 18.26819 | 0.49611 | 13.69350 |
| body_pre_selected_g1 | False | half01 | 17.37302 | 0.40518 | 15.29590 |
| body_pre_selected_g1 | False | raw | 17.23626 | 0.37958 | 17.03498 |
| body_pre_selected_g1 | True | half01 | 17.46492 | 0.40702 | 14.39898 |
| body_pre_selected_g1 | True | raw | 17.33447 | 0.38198 | 15.95367 |
| body_pre_uniform_g05 | False | half01 | 18.30008 | 0.52143 | 13.54259 |
| body_pre_uniform_g05 | False | raw | 18.17479 | 0.49413 | 15.03699 |
| body_pre_uniform_g05 | True | half01 | 18.32758 | 0.51810 | 13.16784 |
| body_pre_uniform_g05 | True | raw | 18.21200 | 0.49166 | 14.57071 |
| pretrained | False | half01 | 18.34145 | 0.47872 | 16.15790 |
| pretrained | False | raw | 18.12699 | 0.44989 | 18.04659 |
| pretrained | True | half01 | 18.31931 | 0.47752 | 16.21226 |
| pretrained | True | raw | 18.12699 | 0.44989 | 18.04659 |

## event_cut

| Method | Reset | Fusion | PSNR | SSIM | Temporal |
|---|---|---|---:|---:|---:|
| body_post_first3_g05 | False | half01 | 18.39059 | 0.50331 | 16.09486 |
| body_post_first3_g05 | False | raw | 18.15065 | 0.47528 | 18.23417 |
| body_post_first3_g05 | True | half01 | 18.46823 | 0.50627 | 15.05764 |
| body_post_first3_g05 | True | raw | 18.24383 | 0.47855 | 17.03522 |
| body_post_selected_g05 | False | half01 | 18.07106 | 0.53257 | 11.53239 |
| body_post_selected_g05 | False | raw | 17.97721 | 0.50713 | 12.94042 |
| body_post_selected_g05 | True | half01 | 18.20497 | 0.53541 | 11.13782 |
| body_post_selected_g05 | True | raw | 18.09734 | 0.50979 | 12.52139 |
| body_post_selected_g1 | False | half01 | 16.69059 | 0.37715 | 15.10760 |
| body_post_selected_g1 | False | raw | 16.55761 | 0.35302 | 17.03483 |
| body_post_selected_g1 | True | half01 | 16.97105 | 0.39036 | 13.99929 |
| body_post_selected_g1 | True | raw | 16.82537 | 0.36557 | 15.79456 |
| body_post_uniform_g05 | False | half01 | 18.06264 | 0.53404 | 12.07831 |
| body_post_uniform_g05 | False | raw | 17.95474 | 0.50789 | 13.60491 |
| body_post_uniform_g05 | True | half01 | 18.18287 | 0.53626 | 11.72030 |
| body_post_uniform_g05 | True | raw | 18.06275 | 0.50998 | 13.22113 |
| body_pre_first3_g05 | False | half01 | 16.56769 | 0.49226 | 13.99490 |
| body_pre_first3_g05 | False | raw | 16.39902 | 0.46652 | 15.85938 |
| body_pre_first3_g05 | True | half01 | 16.63737 | 0.49628 | 13.75590 |
| body_pre_first3_g05 | True | raw | 16.46354 | 0.47037 | 15.54137 |
| body_pre_selected_g05 | False | half01 | 18.10693 | 0.53171 | 11.65356 |
| body_pre_selected_g05 | False | raw | 18.00879 | 0.50622 | 13.08014 |
| body_pre_selected_g05 | True | half01 | 18.24425 | 0.53432 | 11.25927 |
| body_pre_selected_g05 | True | raw | 18.13240 | 0.50866 | 12.65926 |
| body_pre_selected_g1 | False | half01 | 16.69059 | 0.37715 | 15.10760 |
| body_pre_selected_g1 | False | raw | 16.55761 | 0.35302 | 17.03483 |
| body_pre_selected_g1 | True | half01 | 16.97105 | 0.39036 | 13.99929 |
| body_pre_selected_g1 | True | raw | 16.82537 | 0.36557 | 15.79456 |
| body_pre_uniform_g05 | False | half01 | 18.05554 | 0.53034 | 12.29260 |
| body_pre_uniform_g05 | False | raw | 17.94235 | 0.50416 | 13.85084 |
| body_pre_uniform_g05 | True | half01 | 18.18587 | 0.53245 | 11.92397 |
| body_pre_uniform_g05 | True | raw | 18.06017 | 0.50615 | 13.45173 |
| pretrained | False | half01 | 18.50093 | 0.50637 | 15.01355 |
| pretrained | False | raw | 18.27697 | 0.47834 | 16.98249 |
| pretrained | True | half01 | 18.50038 | 0.50604 | 15.01504 |
| pretrained | True | raw | 18.27697 | 0.47834 | 16.98249 |

