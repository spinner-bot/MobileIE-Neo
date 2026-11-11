# Known static temporal information

All fixed conditions; means across the two photon counts/images and model seeds. Noise correlation after JPEG is measured separately.

| Model | rho | JPEG | First − second | PSNR delta | Bias MSE delta | Variance ratio | Noise SD delta |
|---|---:|---|---|---:|---:|---:|---:|
| R14_plain | 0 | False | preaverage_current − current | 0.992053 | -0.00014843 | 0.533649 | -2.405729 |
| R14_plain | 0 | True | preaverage_current − current | 0.865120 | 0.00005981 | 0.530491 | -2.854736 |
| R14_plain | 0.9 | False | preaverage_current − current | 0.097169 | -0.00002029 | 0.950502 | -0.219870 |
| R14_plain | 0.9 | True | preaverage_current − current | 0.211130 | -0.00000441 | 0.881642 | -0.629077 |
| aligned_history | 0 | False | learned_history − current | 0.019708 | 0.00000877 | 0.975216 | -0.101672 |
| aligned_history | 0 | False | preaverage_current − current | 0.783054 | -0.00012604 | 0.542573 | -2.229144 |
| aligned_history | 0 | False | preaverage_current − learned_history | 0.763346 | -0.00013481 | 0.556375 | -2.127472 |
| aligned_history | 0 | True | learned_history − current | 0.035248 | 0.00000386 | 0.974685 | -0.118055 |
| aligned_history | 0 | True | preaverage_current − current | 0.975618 | -0.00012342 | 0.520546 | -2.678520 |
| aligned_history | 0 | True | preaverage_current − learned_history | 0.940370 | -0.00012727 | 0.534137 | -2.560465 |
| aligned_history | 0.9 | False | learned_history − current | 0.000280 | 0.00000246 | 0.997202 | -0.011206 |
| aligned_history | 0.9 | False | preaverage_current − current | 0.076677 | -0.00001885 | 0.951684 | -0.202185 |
| aligned_history | 0.9 | False | preaverage_current − learned_history | 0.076397 | -0.00002131 | 0.954353 | -0.190979 |
| aligned_history | 0.9 | True | learned_history − current | 0.008120 | 0.00000169 | 0.993036 | -0.030952 |
| aligned_history | 0.9 | True | preaverage_current − current | 0.245409 | -0.00005270 | 0.876565 | -0.611770 |
| aligned_history | 0.9 | True | preaverage_current − learned_history | 0.237289 | -0.00005439 | 0.882742 | -0.580818 |
| official | 0 | False | preaverage_current − current | 0.383906 | 0.00375062 | 0.443032 | -6.931932 |
| official | 0 | True | preaverage_current − current | 2.248929 | -0.00150027 | 0.452285 | -6.583150 |
| official | 0.9 | False | preaverage_current − current | -0.084794 | 0.00093188 | 0.942935 | -0.570975 |
| official | 0.9 | True | preaverage_current − current | 0.860171 | -0.00127610 | 0.828400 | -1.763644 |
| raw_history | 0 | False | learned_history − current | 0.013911 | -0.00000083 | 0.991719 | -0.035555 |
| raw_history | 0 | False | preaverage_current − current | 1.000611 | -0.00010232 | 0.520802 | -2.498785 |
| raw_history | 0 | False | preaverage_current − learned_history | 0.986701 | -0.00010149 | 0.525269 | -2.463230 |
| raw_history | 0 | True | learned_history − current | 0.018283 | -0.00000098 | 0.990560 | -0.046099 |
| raw_history | 0 | True | preaverage_current − current | 0.819972 | 0.00007254 | 0.526398 | -2.763741 |
| raw_history | 0 | True | preaverage_current − learned_history | 0.801690 | 0.00007353 | 0.531576 | -2.717643 |
| raw_history | 0.9 | False | learned_history − current | 0.001829 | -0.00000015 | 0.998957 | -0.004612 |
| raw_history | 0.9 | False | preaverage_current − current | 0.093359 | -0.00000765 | 0.945276 | -0.251115 |
| raw_history | 0.9 | False | preaverage_current − learned_history | 0.091530 | -0.00000750 | 0.946261 | -0.246502 |
| raw_history | 0.9 | True | learned_history − current | 0.006750 | -0.00000051 | 0.996640 | -0.016419 |
| raw_history | 0.9 | True | preaverage_current − current | 0.198628 | 0.00001231 | 0.874781 | -0.648875 |
| raw_history | 0.9 | True | preaverage_current − learned_history | 0.191877 | 0.00001281 | 0.877751 | -0.632456 |
| spatial | 0 | False | preaverage_current − current | 1.031651 | -0.00006626 | 0.521915 | -2.539386 |
| spatial | 0 | True | preaverage_current − current | 0.774889 | 0.00017011 | 0.515296 | -2.932320 |
| spatial | 0.9 | False | preaverage_current − current | 0.099498 | -0.00000795 | 0.947806 | -0.238867 |
| spatial | 0.9 | True | preaverage_current − current | 0.183612 | 0.00004760 | 0.869581 | -0.700747 |
