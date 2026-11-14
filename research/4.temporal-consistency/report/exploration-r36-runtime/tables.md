# R36 classic standalone CPU pipelines

Same 480×854 bear_photons20 first16 workload, two Torch/one OpenCV threads, two seeds × three rotated/reversed rounds. Median of six run-average frame costs; min–max is not a confidence interval. Current server load, not phone/energy/peak-memory evidence.

|Method|Median ms/frame|Min–max|History MiB|
|---|---:|---:|---:|
|raw|100.890|80.871–106.492|0.000|
|finite_self05|184.124|168.139–188.478|4.691|
|recursive_self05|186.455|179.614–192.502|9.382|

Raw/finite were remeasured in this same round; do not mix R34 times with this table. Each pipeline includes resident tensor preparation, enhancement/layout, its own required motion/warp/blend and history. Output references remain in the loop; final finite checks/stack/IO/model loading are excluded. Recursive retains two independent RGB allocations; no shared extra teacher CNN.
