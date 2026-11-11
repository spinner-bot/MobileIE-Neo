# R9 基线优化与新增开发片段

创建：2026-11-11T18:25:49.523127+08:00；作者：Codex  (sqs_A100)；原因：R8可折叠分支收益要求排除基线优化不足；先登记再读取新像素或训练。

## 问题与预算

固定R5当前/GT混合流、两个seed、每次1000更新、最终step1000，不在旧test选择checkpoint。增加3种单帧4075参数优化：full_lr1e4（全部1e-4）、full_lr1e3（全部1e-3）、head_lr1e3（首个Conv5权重和bias 1e-3，其余1e-5）。共6次训练6000更新；Adam wd0，Charbonnier .001，主干全参数统一clip1，与plain/R8分支分开clip的区别必须说明。复用R5 plain1e-5及R8spatial_copy结果，不称完整原训练复现。首层分组检验快首层更新，而非精确等价双Adam的copy轨迹。

## 新开发数据（不是最终测试）

沿已保存官方TRAIN顺序，在同名GT/input可用且未参与R4/R4b的19片段中，等距索引0/6/12/18，确定video_15、video_47、video_91、video_121。不看像素选取。每片前16个共同PNG，固定HF OLiVES-Lowlight/OLiVES revision291ae19d061dedcff8e62f9b6d9d33d7ceeedbaf。下载先记录目录元数据，逐文件验证LFS SHA及bytes，检查与所有既有manifest精确hash重叠；不能由hash断言物理场景独立。

这些片段从官方train划作本项目dev，不进入训练。保持原12训练片段不变；未用官方test三片段与Kodak13–24不读取。旧六test已多次探索，后续明确为exploratory。新dev可用于后续训练设置选择，但本轮全部设置固定并完整报告，不宣布独立最终测试。它们仍为白天ND衰减，不代表自然夜景。

## 评价与门槛

固定原38探索条件与4新dev，每权重raw/teacher-half01，保留PSNR、SSIM、edge、GT残差时间误差；新dev实拍resize480270。先报告每片、双seed及全部候选，不按某域PSNR选赢家。稳定训练与基线优化结果决定是否另登记历史对齐训练，不在本协议隐式增加新模块。模型成本及原生尺度诊断另立受控协议。

## 资源与过程

每GPU作业两次空闲预检，最多同组4卡，计划仅2/3；若占用就拒绝并换真正空闲卡重新记录。不设墙钟截止，不覆盖重试。源码、数据清单、训练、评价、过程、报告逐文件小步提交，像素与checkpoint留忽略cache及SHA。任何非有限失败完整保留，不静默调参。
