# R51 周期机制sandbox完整表

0SGD；544CNN=512分数相位+32整数控制。非真实视频质量。

|seed|模式|相位MSE|相位RMS255|clip后相位MSE|相对raw映射MSE|映射RMS255|
|---:|---|---:|---:|---:|---:|---:|
|0|raw|5.51342785e-05|1.893438|5.512681058e-05|0|0.000000|
|0|quadrature|2.689641273e-06|0.418203|2.68964138e-06|3.866952604e-05|1.585713|
|0|preblur|2.343212167e-06|0.390343|2.343212309e-06|3.817712956e-05|1.575585|
|0|postblur|2.019126311e-05|1.145835|2.019126321e-05|4.168143308e-05|1.646310|
|1|raw|5.74860427e-05|1.933399|5.747552892e-05|0|0.000000|
|1|quadrature|3.223317944e-06|0.457817|3.223317944e-06|2.943885054e-05|1.383568|
|1|preblur|2.807833372e-06|0.427293|2.807833372e-06|2.91222059e-05|1.376107|
|1|postblur|2.201482946e-05|1.196459|2.201482946e-05|3.120938275e-05|1.424567|

```json
{
  "0": {
    "nontrivial_raw_phase_error": true,
    "phase_vs_raw_080": true,
    "phase_vs_best_smoothing_090": false,
    "mapping_vs_best_smoothing_105": true,
    "nonworse_crops_at_least_3": true,
    "eligible": false,
    "nonworse_crops": 4
  },
  "1": {
    "nontrivial_raw_phase_error": true,
    "phase_vs_raw_080": true,
    "phase_vs_best_smoothing_090": false,
    "mapping_vs_best_smoothing_105": true,
    "nonworse_crops_at_least_3": true,
    "eligible": false,
    "nonworse_crops": 4
  }
}
```
