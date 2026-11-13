# R36传统强基线完整CPU成本

登记：2026-11-12T07:39:11+08:00；作者：Codex  (sqs_A100)。48行质量已正常结束，固定递归联合优于finite；尚未执行本轮成本。

固定R34 original/final4000双seed及bear_photons20原前16帧480×854，FP32、Torch2/OpenCV1、驻留输入。比较raw、有限self-post .5、递归self-post .5三个独立部署流水线，每seed各三轮，共18次；轮转/反转顺序，warmup4后全新重置历史。没有选择最快轮次。不同stage/服务器负载不能混排速度，故在同一新轮次重测两个参考，不把旧R34计时拼到新递归时间上。

CNN后clip/layout、自身raw相邻增强求half-FB、当前fallback的warp/blend及实际历史均计入。递归运动仍从相邻raw计算，与质量评价相同；递归上一融合RGB另存，不能省掉raw历史后宣称仍同一映射。raw无history，finite一份RGB FP32，recursive两份独立RGB FP32；末帧storage按实际独立allocation核验。离线输出留存与临时激活另述，不称完整峰值。

输出引用保留在循环内，有限值检查/stack/IO/模型加载在计时外，与既有R34口径一致。每值为两seed×三轮的六个运行平均帧成本的中位，同时报告每run、分seed、min/max；无统计置信区间或显著性声明。共享Xeon服务器的CPU成本，不是手机、能源或相机到屏幕时延。adaptive未过质量门槛而不进入部署候选成本，其完整质量失败仍保留。

成本脚本只读取原始底座，不修改或中断R37训练；不使用GPU。若真实接口/数值失败，保留并停止，不按结果变换motion/alpha再追测。
