# Same-base external development baselines

R26 plain MSE, both seeds, common CPU. Equal means over three source videos × two noise conditions × two training seeds. Custom synthetic lowlight DEV; no official benchmark or phone measurement.

|Method|PSNR|SSIM|Temporal↓|Edge↓|
|---|---:|---:|---:|---:|
|fastdvd_sigma15_five_current|23.853137|0.713621|8.742266|4.268499|
|fastdvd_sigma15_window|23.872768|0.718324|7.720899|4.174024|
|raw|23.661207|0.678138|10.069569|4.437215|
|self_motion_post05|23.817468|0.712322|7.448614|4.123568|
|shared_motion_post05|23.735146|0.707393|7.673995|4.158096|

**Prior-training overlap:** the official FastDVDnet training list includes all three R30 DEV videos, all 60 DAVIS TRAIN IDs and all 30 VAL IDs. These rows are development/transfer evidence, never held-out-video generalization. The list is not a per-weight frame-sampling trace. See the versioned training-overlap metadata.

Self-motion post reuses its own enhancement and needs no extra teacher CNN. Shared-motion post uses the original fixed teacher flow for quality control. FastDVDnet window waits two future frames; five_current removes real temporal input with identical denoiser capacity. CPU stage timings exclude phone/camera behavior and cannot be reported as mobile latency.

|Method|Seed|PSNR|SSIM|Temporal↓|Edge↓|
|---|---:|---:|---:|---:|---:|
|fastdvd_sigma15_five_current|0|24.396084|0.718275|9.132146|4.263363|
|fastdvd_sigma15_five_current|1|23.310190|0.708968|8.352387|4.273634|
|fastdvd_sigma15_window|0|24.448159|0.725599|8.091766|4.143728|
|fastdvd_sigma15_window|1|23.297376|0.711049|7.350032|4.204319|
|raw|0|24.120196|0.681078|10.429372|4.447802|
|raw|1|23.202219|0.675199|9.709766|4.426628|
|self_motion_post05|0|24.304612|0.718266|7.649480|4.091471|
|self_motion_post05|1|23.330325|0.706378|7.247747|4.155665|
|shared_motion_post05|0|24.200568|0.713095|7.884127|4.128194|
|shared_motion_post05|1|23.269723|0.701690|7.463862|4.187999|
