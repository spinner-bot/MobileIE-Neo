# R64 固定图实际检查

创建：2026-11-13T02:24:20.250727+08:00；作者：Codex  (sqs_A100)。实际用view_image看过全部20个紧凑联系表（五组各四图）及comparison.png。全部6个SYN条件、4个ND条件、两个seed均覆盖；未人工播放完整视频，未逐张人工查看原分辨率归档，不作MOS/感知闪烁或无拖影保证。

SYN固定0/32/63，ND固定0/8/15；每图Input/GT、三模型raw/自身raw经典recursive .5及causal同权重current，统一原亮度/全帧。生成时清单actual_reviewed=false保持历史状态，后续实际检查由[独立记录](../../process/exploration-r64/visual-review001.json)证明，不把程序生成当人工查看。

在bear/lucia/walking的photons20图中，三组raw及causal current的颗粒噪声仍明显；后续帧经典递归列更平滑，photons80总体噪声较轻。causal与spatial/current外观很接近，固定紧凑图没有显示出足以推翻数值门槛失败的独立改善。首帧递归与raw一致符合协议。缩小显示弱化细噪声，不能据此把它们称逐像素相同或判断感知闪烁。

四ND源均见某些色调、对比或细节与GT不同；video47横向边附近的颜色/颗粒差异、video121大背景的明度/灰色调差异在普通raw及后处理中仍能看到。seed之间、三组之间的细微差异保留；不能由固定三帧或压缩显示唯一归因BN/ISP/剪裁/时间模块，也不能将所有实拍质量变化归于causal。GT配准及原估计参考流限制继续存在。

原始20图共212939870B已逐字节移入自有.cache/experiments/exploration-r64/full-contact-archive001，原full manifest另保留。紧凑20图共31871584B，图像tile从SYN854×480/ND512×288缩为266×150，文字重新绘制，所有帧/信号/颜色处理不变；原浮点数组/指标完全不改。原图和紧凑图摘要同时见visual_sources.json。这个打包是资产/显示管理，不是新质量实验。

最终结论仍为history/A/B均False，局部网格止损，未做候选成本优化/新手机或夜景扩测/最终测试。经典递归更平滑也不证明其所有区域更正确；实际完整140行和全部失败源/seed继续保留。
