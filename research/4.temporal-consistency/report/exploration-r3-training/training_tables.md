# 第三轮受控训练

48合成输入条件，两个训练seed均完整400步。

| method | PSNR | SSIM | temporal error | edge MAE |
|---|---:|---:|---:|---:|
| task__ordinary__raw | 19.7203 | 0.4916 | 16.4648 | 6.1977 |
| task__ordinary__fb | 20.8075 | 0.6887 | 6.1941 | 3.5539 |
| cross__history__raw | 20.3866 | 0.5235 | 14.4114 | 5.7658 |
| cross__history__fb | 21.1574 | 0.6960 | 5.9761 | 3.5359 |
| cross__history_off__raw | 19.7161 | 0.4913 | 16.4805 | 6.2030 |
| cross__history_off__fb | 20.8095 | 0.6885 | 6.1996 | 3.5552 |
| mean_square__history__raw | 20.4004 | 0.5305 | 13.9043 | 5.6479 |
| mean_square__history__fb | 21.1017 | 0.6970 | 5.8875 | 3.5235 |
| mean_square__history_off__raw | 19.7184 | 0.4910 | 16.5096 | 6.2126 |
| mean_square__history_off__fb | 20.8181 | 0.6883 | 6.2098 | 3.5580 |
