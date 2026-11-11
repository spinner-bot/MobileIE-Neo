# R21b 聚合训练后的静态信息利用

创建：2026-11-11T22:43:00+08:00；作者：Codex  (sqs_A100)；原因：R21动态全评价仍运行，在不读取质量结果选模型的前提下，固定零运动/已知噪声诊断，区分映射适配与对应问题。

只复用R19的16个已缓存静态条件及每条件64对观测，不生成/下载新像素：Kodak训练01/06固定128方ROI、photons20/80、rho0/.9、JPEG off/on。current跨rho完全一致；这里是已知完美对应，不用GT估计流。条件不代表自然视频或新独立验证。

固定R21四模式×双seed4000全部8权重，每个current替代/真实历史两路径，共256统计行。另完整CPU重算R14 native plain双seed4000的current、preaverage、postaverage_clipped、postaverage_thenclip四经典参考，共128行，合计384。step4000为训练终点预定预算，不按dev结果选取。全部CPU FP32/batch8、Torch2/OpenCV1线程；不与GPU或dynamic batch1像素数值直接等价。

保存权重/缓存SHA、输出SHA、FP64总体MSE=bias²+draw variance分解、clipping比例和每draw MSE。current跨rho必须逐像素相同（预计8候选×8条件+2参考×8=80次哈希重放），MSE恒等式绝对残差<=1e-12；不放宽门槛。完整报告所有rho/JPEG/photons/seed，不只选独立噪声收益。

零新训练、不修改动态评价、不选择候选。若零运动下真实历史仍不改善，相应聚合训练路线的意义进一步降低；若静态有用而动态不占优，则将对应/域差异列为后续待证假设，不能直接宣称运动是唯一原因。原有R19训练less参考不混CPU数值；主静态比较使用本轮同CPU参考。普通平均及scalar不是创新。
