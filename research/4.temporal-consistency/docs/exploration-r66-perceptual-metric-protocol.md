# R66｜共同缓存的已发表时空感知质量补评

登记：2026-11-13T03:10:59.424893+08:00；作者：Codex  (sqs_A100)。尚未下载/安装本轮指标包、执行资格或读取新质量分数。R65原作业继续，全部活跃依赖冻结。Etemp为GT变化残差代理，不能当主观闪烁；本轮补独立的已发表感知指标，不训练新模型/修订旧失败门槛/选择checkpoint。

固定原ColorVideoVDP（非ML变体）、官方PyPI cvvdp0.5.7，全部原源码/校准数据不改，记录wheel官方SHA和实际metric/version/info。隔离新增依赖至zrh自有`.cache/tools/exploration-r66/cvvdp057`，通过project-python运行，只在本CPU/指标进程sys.path加入该自有工具目录；不修改活跃`.venv`、不升级Torch/NumPy，不借其它用户内容。必要额外包固定torchvision0.24.1+cu126（官方PyTorch、匹配现有Torch2.9.1+cu126）、huggingface-hub0.33.4、imageio2.37.0、ffmpeg-python0.2.0、future1.0.0、requests2.32.5、urllib3 2.5.0、idna3.11、charset-normalizer3.4.4、certifi2025.11.12；全部公开官方元数据/摘要校验，不安装不需要的OpenEXR/ML权重或升级依赖树。安装失败保留；版本变更需另登记，不能默默换源。

固定`display_name=standard_fhd`、nominal播放30fps、SDR RGB float32[0,1]、THWC数组、replicate temporal padding、heatmap=None；指标内部按原显示EOTF转换。保存完整display geometry/photometry/metric info与校准参数来源。原854×480/512×288完整缓存不缩放、不GT mean/曝光/颜色修正，不补插帧/重采样；按标准FHD每像素展示的小窗口条件，不声称实际手机观看条件或原传感器30fps。短16帧和未实测显示条件限制需披露。JOD高为好，强失真允许负分，不设置人为裁截。

CPU资格先只用人工T16×64²：identity应10±1e-4、固定小/大亮度偏差应有限且大偏差不高于小偏差；同幅静态偏差/交替亮度偏差各保留分数但不预设二者大小。只验证布局/范围/接口/物理参数与时间输入实际生效，不由人工结果宣称真实MOS/闪烁。需要GPU时同一人工输入比对CPU/CUDA绝对JOD差≤0.01，失败保留、不放宽追分。不得触碰其它活跃环境或全局站点包。

阶段A固定已有全部共同参考：R44 square_mse双seed raw/rec .5（40行）、R56 Nano双seed raw/rec（40）、R53 FastDVD双seed window（20）、R62公开clean raw/rec（20），以及每源Input/GT identity（20），合计140行、448唯一观察。R44只缓存raw时从既有源重建原固定递归；R53读取已缓存window输出，不做新CNN。所有10源、两域、全部种子保留，不能只取更好seed。FastDVD未来2帧、各预训练/容量/目标与计算差异明示。

阶段B仅在原R65正常完整8000/评价链后，原协议两seed×两臂×raw/rec另外80行；合计220感知行，不把尚未存在的输出当完成，也不因R65结果调整本轮参数。资料均既有开发缓存，0新增强CNN/SGD、0新手机/Night/DRV/最终测试像素。指标可GPU加速但不是被测方法部署开销；神经指标/CPU数值处理计数与增强CNN分开。

首选CPU资格、再按实际成本决定指标CPU/GPU运行，不按墙钟截断。GPU只在4–7同组最多4卡，R65已占5/6时本轮最多1张4或7；新外部双采样/项目锁及完整CPU首源准备后内部采样，≤64MiB/≤5%/无compute才用，不干扰共享用户。没有自动科学重试，失败保留。

最终保留全部逐源JOD、域/seed均值、与既有PSNR/SSIM/Etemp/edge并列及raw→rec变化。新指标不能回写旧采用门槛/把旧失败改成功，不把与Etemp分歧当人类意见或由单一分数证明增强成功。ColorVideoVDP不是专门LLVE感知模型，GT配准/色彩/假定显示及短片限制仍在；人工完整播放/MOS仍欠缺。

来源：[作者页](https://www.cl.cam.ac.uk/research/rainbow/projects/colorvideovdp/)、[官方实现](https://github.com/gfxdisp/ColorVideoVDP)、[PyPI版本](https://pypi.org/project/cvvdp/0.5.7/)。协议、获取/依赖代码、实际来源、资格/阶段输出、解释、过程和报告分别小步提交；阶段fetch后推送/no-ff/指定作者保持。论文创新/有效方法/手机/独立最终目标仍active。

获取/来源修正：2026-11-13T03:16:27.862495+08:00；在任何真实资料JOD前，确认R53原评价只保存window数组，没有five_current数组。将阶段A原160修正为140、A+B原240修正220，保留所有已有window/source/seed；不为非主表的five_current诊断重新运行增强CNN，旧PSNR等结果不删除。包首尝试在官方PyTorch索引指向download-r2.pytorch.org的主机检查处停止（未安装）；已核官方网站确实列该CDN及SHA后明确package002恢复，十个既有校验wheel复用。版本/指标/门槛不变，原失败单独入库。
