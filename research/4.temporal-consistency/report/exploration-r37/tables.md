# R37: preserved spatial anchor and full-resolution projected state

Final4000 only, both seeds, common CPU, three DEV videos × two noise conditions. Four classical rows reuse the exact R36 immutable-anchor weights/input/CPU reference; current explicitly returns this anchor. Two actual candidate routes require their explicit FP16 state and sampling.

|Method|PSNR|SSIM|Temporal↓|Edge↓|Params|History MiB@480×854|
|---|---:|---:|---:|---:|---:|---:|
|raw|26.234161|0.698872|10.505352|4.446445|4075|0.000|
|finite_self05|26.465757|0.743040|7.542986|4.014079|4075|4.691|
|recursive_self05|26.610354|0.758387|6.734680|3.882416|4075|9.382|
|adaptive_recursive|26.385385|0.708503|9.945699|4.346142|4075|10.946|
|zero_alignment|26.254584|0.701275|10.048953|4.385512|4587|2.346|
|learned_alignment|26.281465|0.702914|9.818070|4.339232|4587|2.346|

|Candidate|Reference|Seed|ΔPSNR|Temporal ratio|Joint gate|
|---|---|---:|---:|---:|---|
|learned_alignment|finite_self05|0|-0.173433|1.322318|False|
|learned_alignment|recursive_self05|0|-0.313604|1.480557|False|
|learned_alignment|finite_self05|1|-0.195151|1.280971|False|
|learned_alignment|recursive_self05|1|-0.344174|1.435167|False|
|zero_alignment|finite_self05|0|-0.212683|1.354460|False|
|zero_alignment|recursive_self05|0|-0.352854|1.516545|False|
|zero_alignment|finite_self05|1|-0.209663|1.310051|False|
|zero_alignment|recursive_self05|1|-0.358685|1.467748|False|

Both-seed, both-reference primary screening: {"learned_alignment": false, "zero_alignment": false}. No CPU pipeline, phone latency, energy, peak memory or independent-test claim follows from these quality-stage times.

All conditions and both seeds remain in the paired table. Full-resolution state does not preserve every feature channel; its FP16 storage is three times the R34 low-resolution state and half one RGB FP32 buffer. Zero alignment retains the offset output parameters but they do not influence its prediction, so effective capacity is not asserted identical. Classical adaptive variance is a heuristic. None of projection, residual initialization, gating or warp alone establishes novelty.
