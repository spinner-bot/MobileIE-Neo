# R34 final-model CPU pipeline cost

Resident bear_photons20 frames0–15, 480×854, FP32, two Torch/one OpenCV threads. Three rotated/reversed rounds per training seed. Each value below is the median of six run-average frame costs; ranges show all runs. No phone, energy, peak-memory or quality claim.

|Model|History|Fusion|Parameters|Median ms/frame|Min–max|Online history MiB|
|---|---|---|---:|---:|---:|---:|
|causal|causal|raw|3586|131.757|127.259–141.492|0.784|
|causal|current|raw|3586|125.470|105.985–141.445|0.000|
|causal|current|self_motion_post05|3586|237.708|225.497–247.772|4.691|
|original|current|raw|4075|94.528|81.026–104.251|0.000|
|original|current|self_motion_post05|4075|190.884|182.364–201.167|4.691|
|spatial|current|raw|3586|131.030|112.967–141.101|0.000|
|spatial|current|self_motion_post05|3586|235.643|224.985–249.283|4.691|

Self-motion post uses its own raw enhancement, no extra teacher. Actual causal uses its low-resolution state, not external flow. Timing includes tensor preparation, prediction, layout/clipping and required motion/warp/blend; loop output references are retained. Final finite checks, stack, IO and model loading are outside timing.
