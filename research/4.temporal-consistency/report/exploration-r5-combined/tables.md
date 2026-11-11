# R5 complete domain means

All train seeds/input repeats averaged; lower temporal/edge is better. Exploratory, no test selection.

## test

| Method | PSNR | SSIM | temporal error | edge MAE |
|---|---:|---:|---:|---:|
| plain__causal__half01 | 16.50809 | 0.80581 | 2.27994 | 1.89131 |
| plain__causal__raw | 16.50662 | 0.80617 | 2.42026 | 1.88283 |
| plain_st__causal__half01 | 17.04145 | 0.79622 | 2.19645 | 2.02392 |
| plain_st__causal__raw | 17.04189 | 0.79692 | 2.31480 | 2.01325 |
| pretrained__causal__half01 | 14.83036 | 0.78741 | 2.37453 | 1.88546 |
| pretrained__causal__raw | 14.82822 | 0.78768 | 2.53442 | 1.87607 |
| r2_task__causal__half01 | 16.11922 | 0.80082 | 2.29457 | 1.91515 |
| r2_task__causal__raw | 16.11734 | 0.80118 | 2.43811 | 1.90649 |
| spatial_bilinear__causal__half01 | 16.60616 | 0.80693 | 2.28006 | 1.88584 |
| spatial_bilinear__causal__raw | 16.60476 | 0.80728 | 2.42099 | 1.87742 |
| spatial_post__causal__half01 | 17.18407 | 0.81458 | 2.25312 | 1.89003 |
| spatial_post__causal__raw | 17.18326 | 0.81497 | 2.38982 | 1.88151 |
| temporal_bilinear__causal__half01 | 16.60530 | 0.80693 | 2.28122 | 1.88612 |
| temporal_bilinear__causal__raw | 16.60389 | 0.80728 | 2.42234 | 1.87770 |
| temporal_bilinear__current__half01 | 16.60556 | 0.80692 | 2.28005 | 1.88590 |
| temporal_bilinear__current__raw | 16.60416 | 0.80727 | 2.42097 | 1.87747 |
| temporal_bilinear__off__half01 | 16.36094 | 0.80488 | 2.28263 | 1.88428 |
| temporal_bilinear__off__raw | 16.35941 | 0.80522 | 2.42398 | 1.87580 |
| temporal_bilinear_st__causal__half01 | 17.12682 | 0.79660 | 2.19474 | 2.02341 |
| temporal_bilinear_st__causal__raw | 17.12735 | 0.79731 | 2.31297 | 2.01267 |
| temporal_bilinear_st__current__half01 | 17.12705 | 0.79659 | 2.19376 | 2.02331 |
| temporal_bilinear_st__current__raw | 17.12760 | 0.79729 | 2.31181 | 2.01257 |
| temporal_bilinear_st__off__half01 | 16.92759 | 0.79528 | 2.19978 | 2.01904 |
| temporal_bilinear_st__off__raw | 16.92797 | 0.79598 | 2.31879 | 2.00836 |
| temporal_post__causal__half01 | 17.18436 | 0.81458 | 2.25855 | 1.89037 |
| temporal_post__causal__raw | 17.18349 | 0.81497 | 2.39685 | 1.88184 |
| temporal_post__current__half01 | 17.18545 | 0.81459 | 2.25297 | 1.89027 |
| temporal_post__current__raw | 17.18464 | 0.81497 | 2.38962 | 1.88174 |
| temporal_post__off__half01 | 16.38341 | 0.80211 | 2.26913 | 1.90432 |
| temporal_post__off__raw | 16.38212 | 0.80252 | 2.40695 | 1.89535 |
| tsm_fixed__causal__half01 | 16.46950 | 0.80442 | 2.69285 | 1.91329 |
| tsm_fixed__causal__raw | 16.46043 | 0.80467 | 2.87397 | 1.90633 |
| tsm_fixed__current__half01 | 16.50505 | 0.80596 | 2.28057 | 1.89036 |
| tsm_fixed__current__raw | 16.50356 | 0.80632 | 2.42108 | 1.88190 |
| tsm_learned__causal__half01 | 16.54920 | 0.80474 | 2.26826 | 1.89639 |
| tsm_learned__causal__raw | 16.54745 | 0.80513 | 2.40915 | 1.88769 |
| tsm_learned__current__half01 | 16.46856 | 0.80465 | 2.27421 | 1.89462 |
| tsm_learned__current__raw | 16.46722 | 0.80503 | 2.41333 | 1.88591 |
| tsm_learned_st__causal__half01 | 17.03305 | 0.79577 | 2.19397 | 2.02479 |
| tsm_learned_st__causal__raw | 17.03348 | 0.79648 | 2.31247 | 2.01405 |
| tsm_learned_st__current__half01 | 17.01662 | 0.79575 | 2.19580 | 2.02442 |
| tsm_learned_st__current__raw | 17.01714 | 0.79646 | 2.31389 | 2.01368 |

## legacy_synthetic

| Method | PSNR | SSIM | temporal error | edge MAE |
|---|---:|---:|---:|---:|
| plain__causal__half01 | 17.32660 | 0.52993 | 14.08015 | 6.30989 |
| plain__causal__raw | 17.11115 | 0.50325 | 15.86217 | 6.83488 |
| plain_st__causal__half01 | 16.24788 | 0.53867 | 12.94807 | 5.91454 |
| plain_st__causal__raw | 16.11942 | 0.51320 | 14.56946 | 6.37964 |
| pretrained__causal__half01 | 17.62392 | 0.53197 | 14.35815 | 6.32530 |
| pretrained__causal__raw | 17.32977 | 0.50505 | 16.18218 | 6.85105 |
| r2_task__causal__half01 | 17.18425 | 0.52516 | 14.21637 | 6.35484 |
| r2_task__causal__raw | 16.96325 | 0.49875 | 16.01773 | 6.88362 |
| spatial_bilinear__causal__half01 | 17.37328 | 0.53128 | 14.08419 | 6.31090 |
| spatial_bilinear__causal__raw | 17.15759 | 0.50446 | 15.86704 | 6.83717 |
| spatial_post__causal__half01 | 17.48230 | 0.53435 | 13.98532 | 6.26450 |
| spatial_post__causal__raw | 17.26396 | 0.50742 | 15.75495 | 6.78543 |
| temporal_bilinear__causal__half01 | 17.37357 | 0.53127 | 14.09189 | 6.31087 |
| temporal_bilinear__causal__raw | 17.15767 | 0.50444 | 15.87655 | 6.83714 |
| temporal_bilinear__current__half01 | 17.37332 | 0.53128 | 14.08400 | 6.31080 |
| temporal_bilinear__current__raw | 17.15765 | 0.50446 | 15.86683 | 6.83705 |
| temporal_bilinear__off__half01 | 17.51009 | 0.53155 | 14.12880 | 6.32087 |
| temporal_bilinear__off__raw | 17.28108 | 0.50472 | 15.91841 | 6.84802 |
| temporal_bilinear_st__causal__half01 | 16.24282 | 0.54018 | 12.92808 | 5.90374 |
| temporal_bilinear_st__causal__raw | 16.11634 | 0.51463 | 14.54757 | 6.36807 |
| temporal_bilinear_st__current__half01 | 16.24203 | 0.54019 | 12.92107 | 5.90367 |
| temporal_bilinear_st__current__raw | 16.11571 | 0.51464 | 14.53893 | 6.36798 |
| temporal_bilinear_st__off__half01 | 16.41256 | 0.54068 | 12.97111 | 5.91620 |
| temporal_bilinear_st__off__raw | 16.27708 | 0.51506 | 14.59672 | 6.38241 |
| temporal_post__causal__half01 | 17.48144 | 0.53438 | 13.98353 | 6.26327 |
| temporal_post__causal__raw | 17.26331 | 0.50746 | 15.75306 | 6.78402 |
| temporal_post__current__half01 | 17.48115 | 0.53437 | 13.98277 | 6.26333 |
| temporal_post__current__raw | 17.26298 | 0.50744 | 15.75206 | 6.78411 |
| temporal_post__off__half01 | 17.38840 | 0.53321 | 13.95620 | 6.25348 |
| temporal_post__off__raw | 17.17344 | 0.50650 | 15.72113 | 6.77165 |
| tsm_fixed__causal__half01 | 17.28699 | 0.52986 | 14.47253 | 6.30582 |
| tsm_fixed__causal__raw | 17.07606 | 0.50317 | 16.35944 | 6.82994 |
| tsm_fixed__current__half01 | 17.32113 | 0.52980 | 14.08543 | 6.31230 |
| tsm_fixed__current__raw | 17.10578 | 0.50312 | 15.86811 | 6.83762 |
| tsm_learned__causal__half01 | 16.82213 | 0.52951 | 13.83568 | 6.21843 |
| tsm_learned__causal__raw | 16.63218 | 0.50329 | 15.58414 | 6.72790 |
| tsm_learned__current__half01 | 17.41761 | 0.53196 | 14.04351 | 6.28794 |
| tsm_learned__current__raw | 17.19878 | 0.50519 | 15.82119 | 6.81093 |
| tsm_learned_st__causal__half01 | 16.28511 | 0.53960 | 12.92949 | 5.90617 |
| tsm_learned_st__causal__raw | 16.15484 | 0.51411 | 14.54856 | 6.36988 |
| tsm_learned_st__current__half01 | 16.41519 | 0.54045 | 12.96976 | 5.91691 |
| tsm_learned_st__current__raw | 16.28061 | 0.51484 | 14.59467 | 6.38310 |

## event_light_step

| Method | PSNR | SSIM | temporal error | edge MAE |
|---|---:|---:|---:|---:|
| plain__causal__half01 | 21.35800 | 0.52004 | 14.78459 | 5.46223 |
| plain__causal__raw | 21.02588 | 0.49092 | 16.49432 | 5.93347 |
| plain_st__causal__half01 | 21.07234 | 0.53806 | 13.70853 | 5.04958 |
| plain_st__causal__raw | 20.80313 | 0.50925 | 15.26154 | 5.46855 |
| pretrained__causal__half01 | 18.34145 | 0.47872 | 16.15790 | 5.73090 |
| pretrained__causal__raw | 18.12699 | 0.44989 | 18.04659 | 6.23558 |
| r2_task__causal__half01 | 21.06156 | 0.51682 | 14.88398 | 5.45258 |
| r2_task__causal__raw | 20.73940 | 0.48766 | 16.60804 | 5.92480 |
| spatial_bilinear__causal__half01 | 21.42878 | 0.51981 | 14.80772 | 5.47533 |
| spatial_bilinear__causal__raw | 21.09004 | 0.49059 | 16.52202 | 5.94824 |
| spatial_post__causal__half01 | 21.55577 | 0.52136 | 14.69288 | 5.46676 |
| spatial_post__causal__raw | 21.21515 | 0.49207 | 16.39289 | 5.93881 |
| temporal_bilinear__causal__half01 | 21.42792 | 0.51979 | 14.80661 | 5.47526 |
| temporal_bilinear__causal__raw | 21.08857 | 0.49057 | 16.52378 | 5.94820 |
| temporal_bilinear__current__half01 | 21.42885 | 0.51982 | 14.80753 | 5.47520 |
| temporal_bilinear__current__raw | 21.09011 | 0.49060 | 16.52182 | 5.94810 |
| temporal_bilinear__off__half01 | 21.25723 | 0.51776 | 14.90231 | 5.49859 |
| temporal_bilinear__off__raw | 20.92879 | 0.48859 | 16.62853 | 5.97432 |
| temporal_bilinear_st__causal__half01 | 21.12630 | 0.53849 | 13.69473 | 5.05125 |
| temporal_bilinear_st__causal__raw | 20.85330 | 0.50960 | 15.24911 | 5.47043 |
| temporal_bilinear_st__current__half01 | 21.12663 | 0.53852 | 13.69696 | 5.05112 |
| temporal_bilinear_st__current__raw | 20.85413 | 0.50964 | 15.24919 | 5.47023 |
| temporal_bilinear_st__off__half01 | 20.99735 | 0.53628 | 13.79299 | 5.07322 |
| temporal_bilinear_st__off__raw | 20.73118 | 0.50742 | 15.35779 | 5.49564 |
| temporal_post__causal__half01 | 21.54984 | 0.52134 | 14.72397 | 5.46587 |
| temporal_post__causal__raw | 21.20932 | 0.49205 | 16.42637 | 5.93786 |
| temporal_post__current__half01 | 21.55587 | 0.52139 | 14.69146 | 5.46612 |
| temporal_post__current__raw | 21.21531 | 0.49210 | 16.39128 | 5.93809 |
| temporal_post__off__half01 | 21.16239 | 0.51993 | 14.74535 | 5.43077 |
| temporal_post__off__raw | 20.84789 | 0.49080 | 16.44780 | 5.89861 |
| tsm_fixed__causal__half01 | 21.27095 | 0.52096 | 15.78981 | 5.41113 |
| tsm_fixed__causal__raw | 20.95400 | 0.49199 | 17.62345 | 5.86900 |
| tsm_fixed__current__half01 | 21.35799 | 0.51978 | 14.79933 | 5.46855 |
| tsm_fixed__current__raw | 21.02506 | 0.49065 | 16.51110 | 5.94054 |
| tsm_learned__causal__half01 | 21.26993 | 0.52044 | 14.77336 | 5.39865 |
| tsm_learned__causal__raw | 20.94224 | 0.49138 | 16.47183 | 5.86038 |
| tsm_learned__current__half01 | 21.27476 | 0.51940 | 14.80799 | 5.46373 |
| tsm_learned__current__raw | 20.94906 | 0.49022 | 16.52073 | 5.93548 |
| tsm_learned_st__causal__half01 | 21.00795 | 0.53748 | 13.72858 | 5.04664 |
| tsm_learned_st__causal__raw | 20.74292 | 0.50867 | 15.28301 | 5.46513 |
| tsm_learned_st__current__half01 | 20.98908 | 0.53704 | 13.73974 | 5.05805 |
| tsm_learned_st__current__raw | 20.72560 | 0.50820 | 15.29756 | 5.47846 |

## event_cut

| Method | PSNR | SSIM | temporal error | edge MAE |
|---|---:|---:|---:|---:|
| plain__causal__half01 | 19.77099 | 0.52136 | 14.44493 | 5.79918 |
| plain__causal__raw | 19.49200 | 0.49414 | 16.31666 | 6.29450 |
| plain_st__causal__half01 | 19.47845 | 0.53928 | 13.13185 | 5.40106 |
| plain_st__causal__raw | 19.25692 | 0.51230 | 14.83437 | 5.84091 |
| pretrained__causal__half01 | 18.50093 | 0.50637 | 15.01355 | 5.86829 |
| pretrained__causal__raw | 18.27697 | 0.47834 | 16.98249 | 6.37786 |
| r2_task__causal__half01 | 19.64536 | 0.51959 | 14.46971 | 5.77883 |
| r2_task__causal__raw | 19.36053 | 0.49228 | 16.34942 | 6.27395 |
| spatial_bilinear__causal__half01 | 19.79419 | 0.52087 | 14.48996 | 5.81544 |
| spatial_bilinear__causal__raw | 19.51259 | 0.49360 | 16.36767 | 6.31266 |
| spatial_post__causal__half01 | 19.92347 | 0.52317 | 14.41126 | 5.79564 |
| spatial_post__causal__raw | 19.64139 | 0.49573 | 16.27756 | 6.29115 |
| temporal_bilinear__causal__half01 | 19.79506 | 0.52086 | 14.49633 | 5.81500 |
| temporal_bilinear__causal__raw | 19.51242 | 0.49358 | 16.37815 | 6.31216 |
| temporal_bilinear__current__half01 | 19.79441 | 0.52087 | 14.48975 | 5.81534 |
| temporal_bilinear__current__raw | 19.51281 | 0.49360 | 16.36744 | 6.31253 |
| temporal_bilinear__off__half01 | 19.85477 | 0.52092 | 14.54226 | 5.82519 |
| temporal_bilinear__off__raw | 19.57044 | 0.49359 | 16.42753 | 6.32427 |
| temporal_bilinear_st__causal__half01 | 19.48201 | 0.53942 | 13.13832 | 5.40387 |
| temporal_bilinear_st__causal__raw | 19.25927 | 0.51238 | 14.84500 | 5.84374 |
| temporal_bilinear_st__current__half01 | 19.48066 | 0.53942 | 13.13209 | 5.40403 |
| temporal_bilinear_st__current__raw | 19.25877 | 0.51240 | 14.83484 | 5.84390 |
| temporal_bilinear_st__off__half01 | 19.57082 | 0.53929 | 13.20245 | 5.41636 |
| temporal_bilinear_st__off__raw | 19.34556 | 0.51219 | 14.91498 | 5.85910 |
| temporal_post__causal__half01 | 19.91944 | 0.52318 | 14.41758 | 5.79508 |
| temporal_post__causal__raw | 19.63758 | 0.49575 | 16.28415 | 6.29066 |
| temporal_post__current__half01 | 19.92349 | 0.52320 | 14.40919 | 5.79485 |
| temporal_post__current__raw | 19.64148 | 0.49576 | 16.27524 | 6.29026 |
| temporal_post__off__half01 | 19.83937 | 0.52358 | 14.34542 | 5.75925 |
| temporal_post__off__raw | 19.56372 | 0.49625 | 16.20539 | 6.24980 |
| tsm_fixed__causal__half01 | 19.54894 | 0.51800 | 16.26691 | 5.80364 |
| tsm_fixed__causal__raw | 19.25867 | 0.49134 | 18.42404 | 6.28932 |
| tsm_fixed__current__half01 | 19.76565 | 0.52121 | 14.45170 | 5.80186 |
| tsm_fixed__current__raw | 19.48645 | 0.49399 | 16.32431 | 6.29757 |
| tsm_learned__causal__half01 | 19.29794 | 0.51953 | 14.31236 | 5.71889 |
| tsm_learned__causal__raw | 19.04187 | 0.49248 | 16.17406 | 6.20213 |
| tsm_learned__current__half01 | 19.83695 | 0.52221 | 14.43937 | 5.79270 |
| tsm_learned__current__raw | 19.55695 | 0.49487 | 16.31114 | 6.28775 |
| tsm_learned_st__causal__half01 | 19.45876 | 0.53909 | 13.13737 | 5.39605 |
| tsm_learned_st__causal__raw | 19.23869 | 0.51209 | 14.84175 | 5.83513 |
| tsm_learned_st__current__half01 | 19.56462 | 0.53956 | 13.16561 | 5.40885 |
| tsm_learned_st__current__raw | 19.34145 | 0.51249 | 14.87283 | 5.85014 |

