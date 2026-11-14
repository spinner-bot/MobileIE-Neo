# R62 原公开StableLLVE clean权重：共同CPU完整参考

同当前6SYN/4ND资料、448观察；公开clean预训练，项目更新0/不虚构seed，非原论文noisy设置完整复现。
与已适配模型训练条件/计算不同，不据迁移结果作一般架构或SOTA排名。经典递归含额外运动/历史成本。

|域|输出|源数|PSNR|SSIM|时间误差|Edge|
|---|---|---:|---:|---:|---:|---:|
|davis_train_pilot|StableLLVE_clean_raw|6|21.384658|0.622391|9.802266|4.667800|
|davis_train_pilot|StableLLVE_clean_recursive_self05|6|21.540874|0.678547|6.667801|4.280807|
|paired_real_ND_DEV|StableLLVE_clean_raw|4|20.586070|0.852835|3.126002|1.863714|
|paired_real_ND_DEV|StableLLVE_clean_recursive_self05|4|20.539086|0.854198|2.475855|1.814323|
