# R2 generated evidence

All videos are synthetic; image versions and split are in the manifest.

## Training (mean over exploratory validation sequences and two training seeds)

| Method | PSNR | SSIM | Edge MAE | GT-relative temporal error |
| --- | ---: | ---: | ---: | ---: |
| pretrained | 18.2765 | 0.4718 | 6.3557 | 17.3157 |
| task | 19.7188 | 0.4916 | 6.1975 | 16.4645 |
| task_T0.1 | 19.7154 | 0.4938 | 6.1412 | 16.2932 |
| bounded | 19.5778 | 0.5053 | 5.8493 | 15.3648 |
| bounded_T0.1 | 19.5386 | 0.5080 | 5.7859 | 15.1561 |
| task_T0.5 | 19.6609 | 0.5007 | 5.9658 | 15.7534 |
| task_T1 | 19.5568 | 0.5062 | 5.8222 | 15.3067 |

## State controls: pan

| Method | PSNR | SSIM | Temporal error |
| --- | ---: | ---: | ---: |
| fb_ema | 19.2834 | 0.6943 | 5.6457 |
| flow_ema | 19.2829 | 0.6952 | 5.6220 |
| gate_ema | 18.0292 | 0.4809 | 16.4798 |
| gate_reset | 18.0292 | 0.4809 | 16.4798 |
| gate_reset_fb | 18.9641 | 0.6935 | 5.5058 |
| input_nlm | 18.8001 | 0.4802 | 17.1176 |
| mobileie | 18.3107 | 0.4783 | 16.9693 |
| oracle_ema | 19.4167 | 0.7015 | 4.0446 |

## State controls: light_step

| Method | PSNR | SSIM | Temporal error |
| --- | ---: | ---: | ---: |
| fb_ema | 19.2032 | 0.6573 | 7.5511 |
| flow_ema | 19.2012 | 0.6579 | 7.5332 |
| gate_ema | 18.2292 | 0.4387 | 17.8617 |
| gate_reset | 18.1161 | 0.4502 | 17.8266 |
| gate_reset_fb | 18.9172 | 0.6407 | 7.7514 |
| input_nlm | 18.8774 | 0.4591 | 18.2854 |
| mobileie | 18.1270 | 0.4499 | 18.0466 |
| oracle_ema | 19.3938 | 0.6776 | 5.7919 |

## State controls: occlusion

| Method | PSNR | SSIM | Temporal error |
| --- | ---: | ---: | ---: |
| fb_ema | 19.1896 | 0.6840 | 6.2624 |
| flow_ema | 19.1129 | 0.6860 | 6.0683 |
| gate_ema | 18.1191 | 0.4826 | 16.6888 |
| gate_reset | 18.1191 | 0.4826 | 16.6888 |
| gate_reset_fb | 18.8680 | 0.6831 | 6.0986 |
| input_nlm | 18.8782 | 0.4825 | 17.3967 |
| mobileie | 18.3968 | 0.4804 | 17.2710 |
| oracle_ema | 19.5370 | 0.6985 | 4.1387 |

## State controls: cut

| Method | PSNR | SSIM | Temporal error |
| --- | ---: | ---: | ---: |
| fb_ema | 18.8069 | 0.6682 | 6.2682 |
| flow_ema | 18.7536 | 0.6714 | 6.2923 |
| gate_ema | 19.0953 | 0.4909 | 16.3043 |
| gate_reset | 18.1505 | 0.4797 | 16.6140 |
| gate_reset_fb | 19.0892 | 0.6754 | 5.9169 |
| input_nlm | 18.7639 | 0.4802 | 17.1202 |
| mobileie | 18.2770 | 0.4783 | 16.9825 |
| oracle_ema | 19.3098 | 0.6758 | 4.1571 |
