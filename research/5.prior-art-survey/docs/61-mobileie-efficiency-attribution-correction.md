# MobileIE效率引用的硬件与计数口径纠正

2026-11-13；Codex  (sqs_A100)。本次纠正早期复现README/HTML的错误标注，未重跑Windows延迟、手机测试或增强模型。

[作者arXiv v1](https://arxiv.org/html/2507.01838v1)表1分别列600×400、GPU .895ms、SoC6.72ms和FPS1120.584；§4.2注明4090与Snapdragon8Gen3。表中FPS与GPU延迟量级一致，不能标成“手机1100 FPS”。6.72ms倒数约148.81 FPS只是算术推导，不是本项目实测，也不含相机、解码和视频后处理。不同舍入列无需精确互为倒数。CVF正式元数据已定位，但本次完整正式PDF请求403；上述数值只按实际读取的v1引用，未声称全文版本一致。

[代码参数检查](../process/mobileie-efficiency-correction/inspection001.json)直接实例化CPU MobileIELLENetS(12)，无forward或权重读取：注册4075；head/body FSTS各14，合28；排除这28后恰4047。不能在报告中把4047当完整注册总数。差值没有证明历史工具的确切行为，本次未重跑THOP；也不把可相乘参数的函数自由度混成注册参数量。

早期工具0.924GMAC与原文标注0.924GFLOPs不能仅因数值相同就视为完整算子口径复现。现有论文的Conv-only budget仍是4075注册参数/限定卷积预算，手机ms继续缺失，不重建论文或改任何模型/数值结果。

实际修正[早期README](../../2.reproduction/README.md)与[HTML报告](../../2.reproduction/report/reproduction-report.html)的参数/FPS标签和推断边界；历史Windows1.447ms/691FPS保留，未臆测GPU型号就是全部差异原因。全部原内嵌图payload保持，见[检查](../process/mobileie-efficiency-correction/correction-check001.json)。未触碰移动演示支线或其部署工作。

原17/27/37冻结源和四原PID/start保持；没有新视频/训练/DEV/JOD/GPU或方法成功。本修正避免用GPU数字夸大端侧预算，不代替将来实际完整管线的手机时延/状态/能耗测量。
