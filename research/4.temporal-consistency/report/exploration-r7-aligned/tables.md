# Matched conditional 3-channel-state/motion controls

Same supplied motion and sampler. Reference proxy is not deployable. Recurrent RGB has longer temporal context.

## test

| Method | Gamma | Motion | Reset | PSNR | SSIM | Temporal |
|---|---:|---|---|---:|---:|---:|
| feature_selected_post | 0.1 | reference_proxy | False | 14.82671 | 0.78713 | 2.46852 |
| feature_selected_post | 0.1 | reference_proxy | True | 14.82671 | 0.78713 | 2.46852 |
| feature_selected_post | 0.1 | teacher_half_fb | False | 14.82662 | 0.78706 | 2.47141 |
| feature_selected_post | 0.1 | teacher_half_fb | True | 14.82662 | 0.78706 | 2.47141 |
| feature_selected_post | 0.1 | zero | False | 14.81647 | 0.78270 | 2.50100 |
| feature_selected_post | 0.1 | zero | True | 14.81647 | 0.78270 | 2.50100 |
| feature_selected_post | 0.5 | reference_proxy | False | 14.81800 | 0.78244 | 2.34954 |
| feature_selected_post | 0.5 | reference_proxy | True | 14.81800 | 0.78244 | 2.34954 |
| feature_selected_post | 0.5 | teacher_half_fb | False | 14.81725 | 0.78192 | 2.36560 |
| feature_selected_post | 0.5 | teacher_half_fb | True | 14.81725 | 0.78192 | 2.36560 |
| feature_selected_post | 0.5 | zero | False | 14.73105 | 0.74310 | 2.64601 |
| feature_selected_post | 0.5 | zero | True | 14.73105 | 0.74310 | 2.64601 |
| output_finite | 0.1 | reference_proxy | False | 14.83019 | 0.78759 | 2.36597 |
| output_finite | 0.1 | reference_proxy | True | 14.83019 | 0.78759 | 2.36597 |
| output_finite | 0.1 | teacher_half_fb | False | 14.83018 | 0.78750 | 2.37128 |
| output_finite | 0.1 | teacher_half_fb | True | 14.83018 | 0.78750 | 2.37128 |
| output_finite | 0.1 | zero | False | 14.81682 | 0.78138 | 2.41703 |
| output_finite | 0.1 | zero | True | 14.81682 | 0.78138 | 2.41703 |
| output_finite | 0.5 | reference_proxy | False | 14.83492 | 0.78333 | 2.06616 |
| output_finite | 0.5 | reference_proxy | True | 14.83492 | 0.78333 | 2.06616 |
| output_finite | 0.5 | teacher_half_fb | False | 14.83425 | 0.78259 | 2.09452 |
| output_finite | 0.5 | teacher_half_fb | True | 14.83425 | 0.78259 | 2.09452 |
| output_finite | 0.5 | zero | False | 14.69665 | 0.72403 | 2.51782 |
| output_finite | 0.5 | zero | True | 14.69665 | 0.72403 | 2.51782 |
| output_recursive | 0.1 | reference_proxy | False | 14.83037 | 0.78750 | 2.36947 |
| output_recursive | 0.1 | reference_proxy | True | 14.83037 | 0.78750 | 2.36947 |
| output_recursive | 0.1 | teacher_half_fb | False | 14.83036 | 0.78741 | 2.37452 |
| output_recursive | 0.1 | teacher_half_fb | True | 14.83036 | 0.78741 | 2.37452 |
| output_recursive | 0.1 | zero | False | 14.81658 | 0.78139 | 2.41806 |
| output_recursive | 0.1 | zero | True | 14.81658 | 0.78139 | 2.41806 |
| output_recursive | 0.5 | reference_proxy | False | 14.84284 | 0.77994 | 1.97925 |
| output_recursive | 0.5 | reference_proxy | True | 14.84284 | 0.77994 | 1.97925 |
| output_recursive | 0.5 | teacher_half_fb | False | 14.84191 | 0.77898 | 2.00394 |
| output_recursive | 0.5 | teacher_half_fb | True | 14.84191 | 0.77898 | 2.00394 |
| output_recursive | 0.5 | zero | False | 14.67357 | 0.72300 | 2.43967 |
| output_recursive | 0.5 | zero | True | 14.67357 | 0.72300 | 2.43967 |
| pretrained | None | none | False | 14.82822 | 0.78768 | 2.53442 |
| pretrained | None | none | True | 14.82822 | 0.78768 | 2.53442 |

## legacy_synthetic

| Method | Gamma | Motion | Reset | PSNR | SSIM | Temporal |
|---|---:|---|---|---:|---:|---:|
| feature_selected_post | 0.1 | reference_proxy | False | 17.61285 | 0.53062 | 14.41496 |
| feature_selected_post | 0.1 | reference_proxy | True | 17.61285 | 0.53062 | 14.41496 |
| feature_selected_post | 0.1 | teacher_half_fb | False | 17.60931 | 0.52875 | 14.65026 |
| feature_selected_post | 0.1 | teacher_half_fb | True | 17.60931 | 0.52875 | 14.65026 |
| feature_selected_post | 0.1 | zero | False | 17.59695 | 0.52754 | 14.69820 |
| feature_selected_post | 0.1 | zero | True | 17.59695 | 0.52754 | 14.69820 |
| feature_selected_post | 0.5 | reference_proxy | False | 18.36050 | 0.60474 | 9.51666 |
| feature_selected_post | 0.5 | reference_proxy | True | 18.36050 | 0.60474 | 9.51666 |
| feature_selected_post | 0.5 | teacher_half_fb | False | 18.30368 | 0.59253 | 10.65139 |
| feature_selected_post | 0.5 | teacher_half_fb | True | 18.30368 | 0.59253 | 10.65139 |
| feature_selected_post | 0.5 | zero | False | 18.01759 | 0.56328 | 11.39279 |
| feature_selected_post | 0.5 | zero | True | 18.01759 | 0.56328 | 11.39279 |
| output_finite | 0.1 | reference_proxy | False | 17.62532 | 0.53387 | 14.06274 |
| output_finite | 0.1 | reference_proxy | True | 17.62532 | 0.53387 | 14.06274 |
| output_finite | 0.1 | teacher_half_fb | False | 17.62190 | 0.53193 | 14.33466 |
| output_finite | 0.1 | teacher_half_fb | True | 17.62190 | 0.53193 | 14.33466 |
| output_finite | 0.1 | zero | False | 17.60854 | 0.53062 | 14.37799 |
| output_finite | 0.1 | zero | True | 17.60854 | 0.53062 | 14.37799 |
| output_finite | 0.5 | reference_proxy | False | 18.35048 | 0.61890 | 8.26524 |
| output_finite | 0.5 | reference_proxy | True | 18.35048 | 0.61890 | 8.26524 |
| output_finite | 0.5 | teacher_half_fb | False | 18.29020 | 0.60564 | 9.61249 |
| output_finite | 0.5 | teacher_half_fb | True | 18.29020 | 0.60564 | 9.61249 |
| output_finite | 0.5 | zero | False | 17.98217 | 0.56792 | 10.59124 |
| output_finite | 0.5 | zero | True | 17.98217 | 0.56792 | 10.59124 |
| output_recursive | 0.1 | reference_proxy | False | 17.62730 | 0.53392 | 14.10981 |
| output_recursive | 0.1 | reference_proxy | True | 17.62730 | 0.53392 | 14.10981 |
| output_recursive | 0.1 | teacher_half_fb | False | 17.62392 | 0.53197 | 14.35815 |
| output_recursive | 0.1 | teacher_half_fb | True | 17.62392 | 0.53197 | 14.35815 |
| output_recursive | 0.1 | zero | False | 17.61063 | 0.53062 | 14.39447 |
| output_recursive | 0.1 | zero | True | 17.61063 | 0.53062 | 14.39447 |
| output_recursive | 0.5 | reference_proxy | False | 18.59211 | 0.64706 | 7.53528 |
| output_recursive | 0.5 | reference_proxy | True | 18.59211 | 0.64706 | 7.53528 |
| output_recursive | 0.5 | teacher_half_fb | False | 18.51652 | 0.63301 | 8.52713 |
| output_recursive | 0.5 | teacher_half_fb | True | 18.51652 | 0.63301 | 8.52713 |
| output_recursive | 0.5 | zero | False | 18.22199 | 0.60064 | 8.98559 |
| output_recursive | 0.5 | zero | True | 18.22199 | 0.60064 | 8.98559 |
| pretrained | None | none | False | 17.32977 | 0.50505 | 16.18218 |
| pretrained | None | none | True | 17.32977 | 0.50505 | 16.18218 |

## event_light_step

| Method | Gamma | Motion | Reset | PSNR | SSIM | Temporal |
|---|---:|---|---|---:|---:|---:|
| feature_selected_post | 0.1 | reference_proxy | False | 18.30227 | 0.47500 | 16.27260 |
| feature_selected_post | 0.1 | reference_proxy | True | 18.29378 | 0.47419 | 16.25021 |
| feature_selected_post | 0.1 | teacher_half_fb | False | 18.29468 | 0.47351 | 16.62986 |
| feature_selected_post | 0.1 | teacher_half_fb | True | 18.28648 | 0.47273 | 16.60512 |
| feature_selected_post | 0.1 | zero | False | 18.26995 | 0.47040 | 16.77839 |
| feature_selected_post | 0.1 | zero | True | 18.26258 | 0.46968 | 16.75764 |
| feature_selected_post | 0.5 | reference_proxy | False | 18.66630 | 0.54389 | 11.70549 |
| feature_selected_post | 0.5 | reference_proxy | True | 18.66541 | 0.54032 | 11.22726 |
| feature_selected_post | 0.5 | teacher_half_fb | False | 18.67881 | 0.54999 | 12.89065 |
| feature_selected_post | 0.5 | teacher_half_fb | True | 18.67929 | 0.54648 | 12.45961 |
| feature_selected_post | 0.5 | zero | False | 18.20107 | 0.49998 | 13.91466 |
| feature_selected_post | 0.5 | zero | True | 18.21198 | 0.49753 | 13.59791 |
| output_finite | 0.1 | reference_proxy | False | 18.34762 | 0.48038 | 15.69835 |
| output_finite | 0.1 | reference_proxy | True | 18.32620 | 0.47913 | 15.75680 |
| output_finite | 0.1 | teacher_half_fb | False | 18.33828 | 0.47847 | 16.13592 |
| output_finite | 0.1 | teacher_half_fb | True | 18.31761 | 0.47728 | 16.19115 |
| output_finite | 0.1 | zero | False | 18.31547 | 0.47487 | 16.30107 |
| output_finite | 0.1 | zero | True | 18.29631 | 0.47380 | 16.35911 |
| output_finite | 0.5 | reference_proxy | False | 18.78248 | 0.55354 | 10.07583 |
| output_finite | 0.5 | reference_proxy | True | 18.71068 | 0.54901 | 9.96730 |
| output_finite | 0.5 | teacher_half_fb | False | 18.79963 | 0.56121 | 11.39844 |
| output_finite | 0.5 | teacher_half_fb | True | 18.73054 | 0.55659 | 11.36508 |
| output_finite | 0.5 | zero | False | 18.24076 | 0.49849 | 12.68435 |
| output_finite | 0.5 | zero | True | 18.19920 | 0.49592 | 12.77804 |
| output_recursive | 0.1 | reference_proxy | False | 18.35117 | 0.48076 | 15.75369 |
| output_recursive | 0.1 | reference_proxy | True | 18.32818 | 0.47948 | 15.80945 |
| output_recursive | 0.1 | teacher_half_fb | False | 18.34146 | 0.47872 | 16.15791 |
| output_recursive | 0.1 | teacher_half_fb | True | 18.31932 | 0.47752 | 16.21227 |
| output_recursive | 0.1 | zero | False | 18.31981 | 0.47513 | 16.30366 |
| output_recursive | 0.1 | zero | True | 18.29909 | 0.47403 | 16.36249 |
| output_recursive | 0.5 | reference_proxy | False | 19.07351 | 0.61005 | 8.59198 |
| output_recursive | 0.5 | reference_proxy | True | 18.90691 | 0.59983 | 8.63962 |
| output_recursive | 0.5 | teacher_half_fb | False | 19.02792 | 0.60407 | 10.07392 |
| output_recursive | 0.5 | teacher_half_fb | True | 18.87918 | 0.59558 | 10.20164 |
| output_recursive | 0.5 | zero | False | 18.45325 | 0.54981 | 10.82805 |
| output_recursive | 0.5 | zero | True | 18.33237 | 0.54101 | 11.19809 |
| pretrained | None | none | False | 18.12699 | 0.44989 | 18.04659 |
| pretrained | None | none | True | 18.12699 | 0.44989 | 18.04659 |

## event_cut

| Method | Gamma | Motion | Reset | PSNR | SSIM | Temporal |
|---|---:|---|---|---:|---:|---:|
| feature_selected_post | 0.1 | reference_proxy | False | 18.48156 | 0.50354 | 14.99227 |
| feature_selected_post | 0.1 | reference_proxy | True | 18.48156 | 0.50354 | 14.99227 |
| feature_selected_post | 0.1 | teacher_half_fb | False | 18.47139 | 0.50271 | 15.34348 |
| feature_selected_post | 0.1 | teacher_half_fb | True | 18.47535 | 0.50249 | 15.34140 |
| feature_selected_post | 0.1 | zero | False | 18.40535 | 0.49774 | 15.60736 |
| feature_selected_post | 0.1 | zero | True | 18.41523 | 0.49736 | 15.60886 |
| feature_selected_post | 0.5 | reference_proxy | False | 18.92092 | 0.56922 | 9.72881 |
| feature_selected_post | 0.5 | reference_proxy | True | 18.92092 | 0.56922 | 9.72881 |
| feature_selected_post | 0.5 | teacher_half_fb | False | 18.87343 | 0.57841 | 11.10197 |
| feature_selected_post | 0.5 | teacher_half_fb | True | 18.95092 | 0.58000 | 10.78111 |
| feature_selected_post | 0.5 | zero | False | 17.97721 | 0.50713 | 12.94042 |
| feature_selected_post | 0.5 | zero | True | 18.09734 | 0.50979 | 12.52139 |
| output_finite | 0.1 | reference_proxy | False | 18.50530 | 0.50705 | 14.58882 |
| output_finite | 0.1 | reference_proxy | True | 18.50530 | 0.50705 | 14.58882 |
| output_finite | 0.1 | teacher_half_fb | False | 18.49966 | 0.50619 | 14.98928 |
| output_finite | 0.1 | teacher_half_fb | True | 18.49911 | 0.50585 | 14.98792 |
| output_finite | 0.1 | zero | False | 18.44446 | 0.50101 | 15.28301 |
| output_finite | 0.1 | zero | True | 18.44425 | 0.50037 | 15.28510 |
| output_finite | 0.5 | reference_proxy | False | 18.95206 | 0.57641 | 8.51002 |
| output_finite | 0.5 | reference_proxy | True | 18.95206 | 0.57641 | 8.51002 |
| output_finite | 0.5 | teacher_half_fb | False | 18.92443 | 0.58786 | 9.98782 |
| output_finite | 0.5 | teacher_half_fb | True | 18.99017 | 0.58964 | 9.64882 |
| output_finite | 0.5 | zero | False | 17.99813 | 0.50851 | 12.21233 |
| output_finite | 0.5 | zero | True | 18.10372 | 0.51122 | 11.73637 |
| output_recursive | 0.1 | reference_proxy | False | 18.50712 | 0.50737 | 14.64572 |
| output_recursive | 0.1 | reference_proxy | True | 18.50712 | 0.50737 | 14.64572 |
| output_recursive | 0.1 | teacher_half_fb | False | 18.50093 | 0.50637 | 15.01356 |
| output_recursive | 0.1 | teacher_half_fb | True | 18.50037 | 0.50604 | 15.01506 |
| output_recursive | 0.1 | zero | False | 18.44684 | 0.50124 | 15.27585 |
| output_recursive | 0.1 | zero | True | 18.44638 | 0.50056 | 15.28474 |
| output_recursive | 0.5 | reference_proxy | False | 19.16962 | 0.62512 | 7.15486 |
| output_recursive | 0.5 | reference_proxy | True | 19.16962 | 0.62512 | 7.15486 |
| output_recursive | 0.5 | teacher_half_fb | False | 19.03939 | 0.62406 | 8.66914 |
| output_recursive | 0.5 | teacher_half_fb | True | 19.14403 | 0.62627 | 8.52387 |
| output_recursive | 0.5 | zero | False | 18.09027 | 0.55593 | 10.05749 |
| output_recursive | 0.5 | zero | True | 18.20534 | 0.55316 | 10.08085 |
| pretrained | None | none | False | 18.27697 | 0.47834 | 16.98249 |
| pretrained | None | none | True | 18.27697 | 0.47834 | 16.98249 |

