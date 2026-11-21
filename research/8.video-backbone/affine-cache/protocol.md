# A1：固定局部仿射算子缓存的零训练质量筛查

登记：2026-11-21T13:17:02.369782+08:00；Codex  (sqs_A100)。先于A1首次缓存像素读取/质量计算。L1及所有旧STOP保持；[先例80](../../5.prior-art-survey/docs/80-keyframe-affine-operator-and-guided-filter-boundaries.md)已说明一般keyframe/仿射场/guide非创新。本次只检验一个表示及时间调用假设，0新增CNN/优化/GPU，不自动启动学习或成本/DEV。

## 唯一表示与三路线

输入x为原低光RGB[0,1]，拟合目标y为原R44 F0**已clip显示**RGB[0,1]，不是GT或L1学生。cv2 INTER_AREA将x/y降至ceil(H/4)×ceil(W/4)，FP64局部5×5归一box/BORDER_REFLECT_101得μx/μy、Cxx=B(xxᵀ)−μxμxᵀ、Cxy=B(xyᵀ)−μxμyᵀ；求A=(Cxx+1e−4 I)⁻¹Cxy，b=μy−μxᵀA。A/b再同5×5 box平滑，存连续自有FP32 packed12（9个A行优先＋3个b）。逐像素列/行方向在render用y_o=Σ_i x_i A_io+b_o固定。双线性放大packed系数，应用**当前**全分辨率x；最后clip[0,1]显示/评分及进入经典post，原未clip新结果范围另记。

三路线固定且独立状态：current每帧取本帧F0并重拟合；cached只在0/2/4…取F0/更新系数，其他帧仅用该场作用于当前x，无算子warp/照明门/额外guide；hold在同key拟合并保存本key投影图像，非key原样重复该图像。三者首帧投影相同，但不等于原F0。current是每帧空间近似控制，不节省F0调用；hold是同一逻辑key预算下不读新图像的诊断，不预设它较差或按其时间低误称有效细节。

cached persistent场=48ceil(H/4)ceil(W/4)字节，128²49152B、1080p6220800B。加自己FB/DIS两个FP32RGB共24HW后，1080p持久总55987200B；不含当前输入/低分辨率求解/上采样12通道/flow/workspace，不是峰值RAM或手机速度。hold可有不同状态占用，不能声称等RAM。

## 数据、来源与逻辑预算

只读取L1已完成held-seed0/1的全部64 manifest项（32唯一四帧输入窗在两原F0 seed重复；sampling9150/step20000..20007）。npz只访问low/gt/F0/F0_FB/F0_DIS，不读取L1学生输出/权重，不重新加载MixedPool或打开新源/DEV/最终。验证自有路径、原manifest摘要及原L1完成标记，窗口仍来自训练场景，16窗/域/seed；不是独立场景或自然夜间。

本筛查通过严格key provider读取缓存F0：current每窗4次、cached/hold各2次且只能[0,2]，合每64窗256/128/128逻辑F0请求；实际新NN为0，不能把缓存读取速度当部署成本。cached未来部署的主网络逻辑调用为原逐帧的一半，不等于整链减半或已测手机提速。下一成本若获准，必须真运行F0并完整计入fit/render/post/copy。

每窗12输出：原F0/F0_FB/F0_DIS三项，current/cached/hold各raw/FB/DIS三项。后处理原half-FB或固定DIS、.5递归，各自从自己raw算自己的field/上一fused RGB，无GT或未来部署输入。共768窗口输出/3072逐帧/48均值；新经典部署双向对1152，GT metric half-FB按相同32唯一GT窗共96对、跨seed共用；沿原GT共同域/8边界/PSNR120dB上限、SSIM/edge定义，无空域省略。

## 固定检查与继续门

CPU2/BLAS2/OpenCV1、os.nice5、CUDA_VISIBLE_DEVICES=''。先仅人工fixture：固定identity A/b渲染差≤1e−6；constant x/y拟合渲染差≤1e−6；seed8200合成四帧分别执行三route，访问current[0,1,2,3]/cached-hold[0,2]、首帧相等、state连续自有且cached bytes49152、cached与hold后续有非零差。缓存provider禁止cached/hold请求非key，不能预拟合非key后假称未运行CNN。该有限检查不是已训练原函数保持或所有尺寸/运动证明。

完整评分后核原三个F0基线的逐窗指标对L1原条目最大差≤1e−6，防来源/共同域漂移，不重做位级训练回放。全部数据先固定，不按中途好片选子集或参数。

20项必要比较全部按两域×两seed分别：4项表示保护current raw对F0，平均PSNR差≥−.05dB、最差窗≥−.50dB；16项主组合cached_FB同时对F0_FB/current_FB、cached_DIS同时对F0_DIS/current_DIS，均相同质量保护且Etemp≤1.02。所有raw/hold及配对结果保留；hold是诊断，不凭比弱hold好宣称原创时间收益。20项全过只允许另登记实际整链配对成本，且成本需≤原同post.80倍才有继续价值；此时仍未证明创新/外部/手机/最终。

若失败，STOP固定quarter/5×5/RGB仿射/epsilon1e−4/K2/无算子warp/双线性/目标及clip网格，不加warp/guide/扩大阶数/训练或换周期救分，不扩失败成本/DEV/JOD/手机。current都近似不好时只说明本表示/拟合不适合当前F0，不否定HDRnet/所有可学习局部算子；cached额外退化也不隔离唯一原因。全48均值、全部20门、配对窗和原首draw每域首窗末帧四行联系图实际审阅；静态图不当MOS/手机。

源、协议/冻结、原完整结果、分析/图审/决定、过程/报告分别密集提交并阶段SSH推送research/O1，作者/no-ff/自有data、原84/snapshot003/PDF020/最终封存/7号/并行main/其他worktree保持。论文目标active未完成。
