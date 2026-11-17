# Fixed native DIS reference: complete quality and same-run CPU costs

Repeated DEV, both seeds. Known classical preset, not an innovative method, handset or independent final.

|Domain|Seed|Output|PSNR|Etemp|JOD|
|---|---:|---|---:|---:|---:|
|davis_train_pilot|0|MobileIE_raw|26.475237|10.101737|7.040340|
|davis_train_pilot|0|MobileIE_recursive_self05|26.800044|6.545945|7.423528|
|davis_train_pilot|0|DIS_recursive_self05|27.050821|6.520286|7.447420|
|davis_train_pilot|1|MobileIE_raw|26.456334|9.699145|7.110548|
|davis_train_pilot|1|MobileIE_recursive_self05|26.612977|6.393445|7.408986|
|davis_train_pilot|1|DIS_recursive_self05|26.882562|6.349339|7.440672|
|paired_real_ND_DEV|0|MobileIE_raw|27.252453|3.728773|8.414068|
|paired_real_ND_DEV|0|MobileIE_recursive_self05|27.292047|2.605355|8.373159|
|paired_real_ND_DEV|0|DIS_recursive_self05|27.336567|2.821074|8.400475|
|paired_real_ND_DEV|1|MobileIE_raw|27.004095|3.576601|8.552201|
|paired_real_ND_DEV|1|MobileIE_recursive_self05|26.923483|2.534350|8.512007|
|paired_real_ND_DEV|1|DIS_recursive_self05|27.001759|2.719274|8.535978|

|Domain|Seed|DIS−FB PSNR|DIS/FB Etemp|DIS−FB JOD|DIS ms|FB ms|Mean paired cost ratio|Quality/speed/joint|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|davis_train_pilot|0|+0.250778|0.996080|+0.023892|166.500|198.736|0.838125|False/False/False|
|davis_train_pilot|1|+0.269585|0.993101|+0.031686|157.637|194.445|0.810719|False/False/False|
|paired_real_ND_DEV|0|+0.044520|1.082798|+0.027315|50.817|68.326|0.743738|False/True/False|
|paired_real_ND_DEV|1|+0.078276|1.072967|+0.023972|52.176|69.440|0.751373|False/True/False|

Quality uses SYN64/ND16 frames; cost uses the same-source first16, both complete pipelines in the same new run, two warmups and three measured repeats, mean across matched condition medians. Shared CPU, no camera/display/file I/O. All per-condition quality losses and cost repeats remain in raw files; no best-seed selection or cross-run R85 speed ratio.
