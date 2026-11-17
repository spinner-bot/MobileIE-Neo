# Current F0 and strong-rec resident CPU pipeline costs

Both frozen R44 seeds, every registered condition, first16 frames only. Each mode/condition uses two full warmups then three full measured runs, reset at each run. The primary number is the mean across conditions of each three-run median ms/frame. Sizes are native; CPU is shared and nonexclusive. No handset latency or new quality claim.

CPU: Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz; Torch 2.9.1+cu126 intra2/inter1; OpenCV 4.11.0 thread1; FP32; batch1. Includes input/output conversion, actual backbone, post motion/warp/blend and output sink copy. Excludes model loading, file/camera/display I/O, sink allocation and qualification comparison.

|Domain|Native H×W|Seed|Raw ms|Strong-rec ms|Mean per-condition rec/raw|Ratio range across conditions|
|---|---|---:|---:|---:|---:|---|
|SYN|480×854|0|107.218|204.910|1.915|1.834–2.076|
|SYN|480×854|1|112.490|208.734|1.856|1.819–1.916|
|ND|288×512|0|45.272|81.201|1.794|1.754–1.863|
|ND|288×512|1|44.662|82.614|1.850|1.805–1.938|

## All condition medians and repeat extrema

Brackets are minimum/maximum of three measured sequence means, not confidence intervals. Ratios divide matched condition medians; the mean of these ratios need not equal a ratio of domain means.

|Seed|Condition|Raw median [min,max] ms|Rec. median [min,max] ms|Ratio|Rec. RGB state B|Shared output sink B|
|---:|---|---|---|---:|---:|---:|
|0|bear_photons20|109.490 [109.489,113.518]|208.947 [205.173,210.791]|1.908|9838080|78704640|
|0|bear_photons80|111.494 [106.130,112.759]|204.527 [202.691,207.184]|1.834|9838080|78704640|
|0|lucia_photons20|111.554 [108.937,112.225]|204.546 [199.913,209.847]|1.834|9838080|78704640|
|0|lucia_photons80|109.762 [100.968,113.714]|203.993 [203.358,204.664]|1.859|9838080|78704640|
|0|walking_photons20|99.894 [92.907,111.934]|207.385 [207.032,208.655]|2.076|9838080|78704640|
|0|walking_photons80|101.112 [98.841,111.233]|200.059 [198.810,201.219]|1.979|9838080|78704640|
|0|video_15|44.215 [44.105,44.776]|82.384 [79.536,82.884]|1.863|3538944|28311552|
|0|video_47|45.634 [44.796,46.078]|80.055 [79.187,80.305]|1.754|3538944|28311552|
|0|video_91|46.030 [45.934,46.862]|82.435 [79.771,82.948]|1.791|3538944|28311552|
|0|video_121|45.207 [45.081,46.196]|79.930 [79.745,84.752]|1.768|3538944|28311552|
|1|bear_photons20|113.875 [110.277,114.543]|207.584 [206.613,208.278]|1.823|9838080|78704640|
|1|bear_photons80|114.055 [112.638,115.032]|207.463 [206.129,208.009]|1.819|9838080|78704640|
|1|lucia_photons20|112.456 [104.572,114.915]|207.558 [206.892,208.453]|1.846|9838080|78704640|
|1|lucia_photons80|113.418 [111.764,114.432]|208.747 [207.551,210.033]|1.841|9838080|78704640|
|1|walking_photons20|109.954 [109.695,115.607]|210.710 [210.623,211.304]|1.916|9838080|78704640|
|1|walking_photons80|111.181 [110.891,113.866]|210.341 [209.187,211.167]|1.892|9838080|78704640|
|1|video_15|43.894 [34.649,46.576]|79.247 [78.573,81.504]|1.805|3538944|28311552|
|1|video_47|46.179 [45.343,46.649]|84.168 [83.735,87.055]|1.823|3538944|28311552|
|1|video_91|43.725 [42.985,44.973]|84.759 [80.262,86.482]|1.938|3538944|28311552|
|1|video_121|44.850 [43.837,46.186]|82.284 [79.744,82.663]|1.835|3538944|28311552|

Raw has zero temporal state. Rec. stable state is two native FP32 RGB frames; startup aliases the two buffers. This excludes flow/workspace/current images/CNN activations. The common sixteen-frame host sink is reported separately and is not model temporal state. All first-frame/nonfirst-frame samples and host loads remain in original JSONL.

All 40 warmup equivalence checks pass against the exact same frozen-weight raw/recursive prediction prefixes; these checks add no CNN calls. No experimental kernel/thread/backend selection follows timing results. Existing failed-candidate STOP rules remain unchanged.
