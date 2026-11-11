# 真实低光样例探索

两段作者样例各16帧，480×270；不代表完整标准基准。

| method | PSNR | SSIM | temporal error | edge MAE |
|---|---:|---:|---:|---:|
| pretrained__raw | 16.2016 | 0.7579 | 2.7202 | 2.1256 |
| pretrained__fb | 16.0064 | 0.6470 | 3.1751 | 3.2407 |
| task__raw | 15.4169 | 0.7466 | 2.6658 | 2.1912 |
| task__fb | 15.2214 | 0.6358 | 3.1673 | 3.2771 |
| T1__raw | 15.3961 | 0.7413 | 2.5705 | 2.2725 |
| T1__fb | 15.2037 | 0.6328 | 3.0578 | 3.2954 |
| cross__history__raw | 15.4292 | 0.7420 | 2.5966 | 2.2363 |
| cross__history__fb | 15.2280 | 0.6317 | 3.1953 | 3.3054 |
| cross__history_off__raw | 15.4174 | 0.7466 | 2.6663 | 2.1907 |
| cross__history_off__fb | 15.2213 | 0.6358 | 3.1683 | 3.2771 |
| mean_square__history__raw | 15.4281 | 0.7409 | 2.5935 | 2.2457 |
| mean_square__history__fb | 15.2268 | 0.6309 | 3.2036 | 3.3112 |
| mean_square__history_off__raw | 15.4150 | 0.7467 | 2.6698 | 2.1887 |
| mean_square__history_off__fb | 15.2190 | 0.6358 | 3.1718 | 3.2768 |
| input_mean__history__raw | 15.4564 | 0.7427 | 2.6126 | 2.2325 |
| input_mean__history__fb | 15.2564 | 0.6323 | 3.1954 | 3.3041 |
| input_mean__history_off__raw | 15.4499 | 0.7474 | 2.6649 | 2.1888 |
| input_mean__history_off__fb | 15.2532 | 0.6364 | 3.1671 | 3.2759 |
| frozen_body_cross__raw | 16.2660 | 0.7310 | 2.7024 | 2.3618 |
| frozen_body_cross__fb | 16.0492 | 0.6311 | 3.3043 | 3.3466 |
| frozen_body_mean_square__raw | 16.2556 | 0.7315 | 2.6934 | 2.3616 |
| frozen_body_mean_square__fb | 16.0426 | 0.6312 | 3.2950 | 3.3475 |
| input_pair__raw | 16.2459 | 0.7316 | 2.7214 | 2.3620 |
| input_pair__fb | 16.0367 | 0.6310 | 3.2964 | 3.3498 |
| output_pair__raw | 16.1752 | 0.7308 | 2.5531 | 2.3693 |
| output_pair__fb | 15.9779 | 0.6308 | 3.2748 | 3.3510 |
