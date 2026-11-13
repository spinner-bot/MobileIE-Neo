# Spatial compute allocation before a new video design

Untrained allocation controls only. Separable convolutions change the function class and cannot exactly factor arbitrary pretrained dense kernels.

|Control|Parameters|Conv GMAC@1080p|Saved Conv GMAC|
|---|---:|---:|---:|
|original|4075|7.937741|0.000000|
|separable_body|3043|5.772903|2.164838|
|separable_head_body|2011|3.608064|4.329677|

|Illustrative history tensor|MiB@1080p|
|---|---:|
|12ch full-resolution FP32|94.922|
|8ch quarter-resolution FP32|3.955|
|8ch quarter-resolution FP16|1.978|

Lower MACs do not imply lower phone latency. History sizes are standalone tensor lower bounds; slicing a large backing tensor may retain the full storage. These calculations reserve budget for future causal-state controls, and provide no evidence of enhancement quality.
