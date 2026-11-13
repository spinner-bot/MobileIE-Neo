# R42 fixed TLC spatial control

Both R41 expanded final4000 seeds, window192 only. 60 new rows / 20 CNN sequences, plus 60 reused same-weight global CPU references. No new training, future input, or raw temporal state. Finite/recursive retain their explicit RGB history and own-raw half-FB.

|Domain|Spatial pooling|Output|PSNR|SSIM|Temporal↓|Edge↓|
|---|---|---|---:|---:|---:|---:|
|davis_train_pilot|R41_expanded_fp32|finite_self05|26.394347|0.751125|7.285598|3.948008|
|davis_train_pilot|R41_expanded_fp32|raw|26.185509|0.711587|10.007045|4.312240|
|davis_train_pilot|R41_expanded_fp32|recursive_self05|26.502948|0.764433|6.520411|3.839379|
|davis_train_pilot|R42_TLC192|finite_self05|26.100293|0.750149|7.335667|3.958061|
|davis_train_pilot|R42_TLC192|raw|25.825225|0.709727|10.105171|4.333441|
|davis_train_pilot|R42_TLC192|recursive_self05|26.218988|0.763867|6.559867|3.844880|
|paired_real_ND_DEV|R41_expanded_fp32|finite_self05|25.734394|0.916057|2.803777|1.449319|
|paired_real_ND_DEV|R41_expanded_fp32|raw|25.846515|0.908162|3.629984|1.549785|
|paired_real_ND_DEV|R41_expanded_fp32|recursive_self05|25.758288|0.919261|2.553407|1.412203|
|paired_real_ND_DEV|R42_TLC192|finite_self05|23.465614|0.911440|2.915793|1.453470|
|paired_real_ND_DEV|R42_TLC192|raw|23.519425|0.903328|3.754367|1.556719|
|paired_real_ND_DEV|R42_TLC192|recursive_self05|23.494847|0.914719|2.663861|1.415607|

|Domain|Seed|TLC−global raw PSNR|Raw temporal ratio|
|---|---:|---:|---:|
|davis_train_pilot|0|-0.305171|1.011444|
|davis_train_pilot|1|-0.415397|1.008228|
|paired_real_ND_DEV|0|-2.311641|1.041637|
|paired_real_ND_DEV|1|-2.342540|1.026838|

Registered fixed-control screen: False. Both seeds must gain .1dB in the same domain, lose no more than .05dB in the other, and have raw temporal ratios ≤1.02 in all domain/seed pairs.

Worst raw per-condition ΔPSNR: -3.716016854023323. All condition pairs and four metrics are retained. Three synthetic source videos are six conditions, not six independent videos; real sources are reused daytime ND DEV. Known TLC is not a new video method. Extra convolution/prefix-sum memory is not zero inference overhead; phone speed remains unmeasured.
