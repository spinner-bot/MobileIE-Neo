# R34 final-weight external development comparison

Both training seeds, equal means of six conditions from three source videos. Common CPU, no event reset, no independent final-test or mobile-latency claim.

|Model|History|Output|PSNR|SSIM|Temporal↓|Edge↓|
|---|---|---|---:|---:|---:|---:|
|causal|causal|raw|25.496515|0.696354|9.551315|4.374354|
|causal|current|postwarp05|25.358520|0.727633|7.361671|4.109950|
|causal|current|raw|25.492768|0.696565|9.561273|4.373176|
|causal|current|self_motion_post05|25.587608|0.733710|7.052074|4.065406|
|original|current|postwarp05|26.218021|0.737103|7.833347|4.058940|
|original|current|raw|26.234161|0.698872|10.505352|4.446445|
|original|current|self_motion_post05|26.465757|0.743040|7.542986|4.014079|
|spatial|current|postwarp05|25.341330|0.727188|7.310007|4.112746|
|spatial|current|raw|25.490643|0.696783|9.461922|4.370254|
|spatial|current|self_motion_post05|25.570730|0.733230|7.002414|4.068684|

Causal/current includes the trained low-resolution spatial path; it is not the bare modified backbone. Actual history benefits must survive this same-weight control and the independently trained equal-capacity spatial control. Shared-motion post uses cached teacher motion; self-motion post computes motion from its own enhancement.

|Reference|Seed|ΔPSNR|Temporal ratio|Registered spatial-post gate|
|---|---:|---:|---:|---|
|causal/postwarp05|0|0.109581|1.307959|None|
|causal/raw|0|0.008842|0.999248|None|
|causal/self_motion_post05|0|-0.127294|1.367090|None|
|original/postwarp05|0|-0.550568|1.255106|None|
|original/raw|0|-0.572758|0.936602|None|
|original/self_motion_post05|0|-0.799158|1.303465|None|
|spatial/postwarp05|0|0.123850|1.314890|None|
|spatial/raw|0|0.009516|1.006968|None|
|spatial/self_motion_post05|0|-0.114033|1.374303|False|
|causal/postwarp05|1|0.166409|1.286495|None|
|causal/raw|1|-0.001348|0.998653|None|
|causal/self_motion_post05|1|-0.054893|1.341230|None|
|original/postwarp05|1|-0.892443|1.183620|None|
|original/raw|1|-0.902535|0.881886|None|
|original/self_motion_post05|1|-1.139326|1.229140|None|
|spatial/postwarp05|1|0.186520|1.297962|None|
|spatial/raw|1|0.002228|1.012083|None|
|spatial/self_motion_post05|1|-0.034396|1.353278|False|

A gate against other references is descriptive only; the amended pre-training primary comparator is independent spatial + self-motion post .5. Original shared-motion gates are retained in the JSON. Report per-seed results and condition pairs rather than selecting the best seed.
