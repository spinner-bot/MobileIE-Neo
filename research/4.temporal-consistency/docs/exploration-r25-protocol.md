# R25 同接受域的对应选择敏感性

创建：2026-11-11T23:18:00+08:00；作者：Codex  (sqs_A100)；原因：静态信息利用与动态质量有差，R24单一global gate不能统一恢复，固定同接受域区分mask收缩与对应选择，不先增加结构训练。

零新训练/数据，四native开发缓存全部16帧，R21四模式双seed4000全部8权重，另R14 native plain双seed4000的经典output-warp .5。所有noreset/raw。五方法×两seed×四片×四regime=160逻辑行/2560帧：current替代、teacher_default、teacher_common、reference_common。前两regime的80序列/1280帧须与正式共同CPU结果最大指标差<=1e-10；新增80分析配置，不把逻辑配置当独立片段。

先读缓存half_flow/mask与reference_flow/mask，分别按实际gather bounds/FB mask获得valid_teacher、valid_reference，common为二者交集。teacher_common及reference_common都仅接受同common，其他像素均回退当前；前者与teacher_default差包含mask收缩影响，两common间差才是同接受域下的flow选择敏感性。frame0均当前，不使用未来帧。

R21对过去低光RGB做对应gather再映射，R14经典参考对已增强过去输出做gather再.5平均，均沿用原定义/精度。R14 post的alias只能依据实际warped-output数组相等，不能凭低光gather输入相等推断神经输出warp相等。完整序列输入/输出FP32字节哈希相同才复用预测/指标，记录alias和实际计算数，不以近似阈值跳过配置。

四片60帧对全部记录接受比例、两flow差/范数（common内）、aligned低光差、历史array SHA，先核验来源SHA/形状。全部CPU FP32/batch1/Torch2/OpenCV1，不设墙钟截止。首帧最大灰度差<=1e-4；保留全部逐帧/逐片PSNR/SSIM/edge/时间及全帧全局/中心化MSE，MSE恒等式1e-12、正式PSNR一致1e-8dB；失败保留，不改门槛。

reference_flow是由GT图像估计的流，不是真实运动或EPE标签，shared common也接触GT信息，均不可部署。两flow还会改变插值平滑/噪声相关性，不能把差异唯一归因于运动精度。时序评价本来也使用该GT估计flow，因此不能仅凭时间误差降低认定对应更准；同步看PSNR/SSIM/edge。该诊断不是公平成绩排名或新方法。若无质量恢复，不能否定所有对应方法，但应降低把当前失败泛归因于flow的优先级，返回训练/域/信息统计问题。最终保留素材未读，论文目标active。
