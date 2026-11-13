# R36: classic recursive controls on the strong R34 original backbone

Common CPU, fixed two final4000 seeds, three DEV source videos × two noise conditions, 64 frames. No new training or parameter selection. Metrics are equally averaged by video/condition/seed; conditions are not six independent videos.

|Method|PSNR|SSIM|Temporal↓|Edge↓|History MiB|
|---|---:|---:|---:|---:|---:|
|raw|26.234161|0.698872|10.505352|4.446445|0.000|
|finite_self05|26.465757|0.743040|7.542986|4.014079|4.691|
|recursive_self05|26.610354|0.758387|6.734680|3.882416|9.382|
|adaptive_recursive|26.385385|0.708503|9.945699|4.346142|10.946|

|Method|Seed|ΔPSNR vs finite|Temporal ratio|Joint gate|
|---|---:|---:|---:|---|
|recursive_self05|0|0.140171|0.893122|True|
|recursive_self05|1|0.149022|0.892559|True|
|adaptive_recursive|0|-0.075273|1.317461|False|
|adaptive_recursive|1|-0.085471|1.319608|False|

All post controls use the same own-raw half-FB motion and invalid-current fallback. Recursive controls use past fused outputs in addition to the past raw used for motion. Adaptive variance is a Haar-MAD/innovation heuristic; it is not a calibrated covariance or a BNLK/EMVD reproduction. History includes every required raw/fused/variance buffer and excludes offline outputs and transient activations.

No standalone latency is inferred from the shared quality-evaluation stages. Server CPU is not phone latency. Raw/finite reference checks passed for all 12 input/weight pairs within 1e-6 for all four reported metrics. Passing a development gate does not establish novelty or independent generalization.
