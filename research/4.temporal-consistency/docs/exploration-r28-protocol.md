# R28 同参数body FST时间算子训练适配

创建：2026-11-12T01:12:55.461070+08:00；作者：Codex  (sqs_A100)；原因：R27已给出可比的冻结控制，不能由未训练冲击判定视频适配成败；预先固定有界训练与强参考。

## 前置门槛与方法范围

正式训练开始前，R27 CUDA算子/梯度/stream验证及真实训练fields/copy Adam检查必须全部通过；仍按两次空闲采样/同组最多四卡，等待不持CUDA或本地锁，无墙钟截止。首CPU copy精确失败保留且未定位，不能以之后重复通过抹除；正式copy检查点必须逐tensor回放旧模型/完整Adam。若CUDA门槛或copy回放失败，保留现场并停止该seed，不静默放宽门槛或换后端/stopgrad。

模式为R27全部五种copy/post_transport/post_after_warp/pre_square/cross，每种仍4075参数。所有head/全局/空间读出/tail保持原计算；上一帧未融合特征由同权重编码器编码，反向保留过去分支梯度。固定teacher输入flow/FB/bounds、1/32插值、拒绝fallback、finite一帧历史。仅用训练数据准备对应，部署不使用GT。跨帧乘法、平方/插值矩恒等式及普通平均不是新颖性。

## 固定训练预算

五模式×Charb/MSE两目标×seed0/1×4000更新=80000新更新，20个独立网络/Adam。copy占16000、四时间控制64000；不能把已有R26更新当作新更新。原生R14完全相同四current/GT任务及RAW过去、ROI/合成噪声流；每步与R26 current/GT/context比对，历史RGB SHA也须与旧GPU R26逐项精确一致，新flow/mask SHA另外存。teacher和fields准备可共享，模型/Adam/梯度不得共享。

原尺度Charb sqrt(e²+1e-6).mean或MSE e².mean，不调尺度/加辅助项。Adam lr1e-3、wd0、betas(.9,.999)、eps1e-8，全局clip1，4000更新不按dev早停；FP32、TF32/cudnn设置沿用，torch.use_deterministic_algorithms(True)、进程局部CUBLAS_WORKSPACE_CONFIG=:4096:8。这些设置全组一致，目标尺度仍改变clip/Adam，非纯梯度方向比较。没有跨帧loss或BPTT长序列，不能称完全视频收敛训练。

固定1000/2000/3000/4000保存80检查点、完整Adam和不可变trace前缀；copy两目标/seed每个checkpoint都与R26相同目标plain模型19/Adam57 tensors精确回放。初始每模式current极限归一化误差<=2e-5，同模式Charb/MSE实际初始输出逐像素相同。时间模式初始actual可能不同于原plain（强制非零历史），不得冒充零扰动初始化；其第一步目标/梯度/clip均保留。所有任务/梯度/权重须有限，数据/来源门槛失败即停；不自动重启部分更新实验。

## 固定正式探索评价

主比较只使用固定4000步终点，不用最优里程碑；1000–3000只留训练诊断和精确回放。全部原46探索条件、双seed、reset关闭/开启及逐片逐帧结果。copy输出raw/recursive .1/finite post .1/.5；四时间模式actual causal输出raw/recursive .1，同时同权重current替代输出raw/recursive .1/finite post .1/.5。所有部署current映射必须先核验与backbone原singleframe等值再作精确alias，不把当前替代提升误认为历史利用。

上述10304逻辑行（copy1472、四时间模式每个2208），不是独立视频样本数。另固定R26 pre/MSE 4000双seed backbone单帧四输出/全部reset，共736逻辑行，以已有较强训练映射作外部参考，不能只与本轮弱copy比较。共11040逻辑行；按既有24个有event条件推导7872指标/3168精确alias、2132真实神经序列（含新参考92），完成时据实际keys重新审计，不把预算推导替代结果。

全部共同CUDA后端，源/权重/缓存SHA核验；copy+R26外部参考的历史正式指标另逐帧重放<=1e-10，firstframe输出差<=1e-4灰度级，不混CPU结果。主要成败看真实历史相对同权重post及固定外部强参考的PSNR/SSIM/时序/边缘、跨域/分seed/事件行为；不只挑一个domain均值或flicker降低。部署特征缓存12通道FP32、teacher/CPUflow/warp及带宽开销都须报告，不能用零新参数替代效率验证。

## 研究解释与边界

冻结R27中cross损失更大不构成淘汰理由，全部预注册算子都训练；如果适配仍不胜同权重post，不继续扩大相似算子网格。若出现一致有效的历史收益，再讨论可证伪噪声/对应机理、可靠性选择与缓存压缩，先查相关工作，再决定后续组合。所有研究DEV/旧TEST已反复使用，OLiVES为白天ND模拟；最终保留3个官方TEST与Kodak13–24保持未读，正式独立验证和自然夜间域仍待。论文目标active，本轮不预先宣称方法创新或SOTA。

## 2026-11-12T01:22:30.301544+08:00 评价入口CPU实现检查

使用固定五帧13×17随机输入和已训练官方权重作接口检查，不读新质量数据：所有模式current、首帧及event reset位置与singleframe精确一致；修改未来两帧不改变前缀，reset后子段与独立新序列相同；copy/外部空间映射不依赖历史。按原46个input-events清单独立枚举逻辑/实际model/output键，固定每分组2208行、总11040，验证7872指标/3168alias/2132神经序列预算。这不是正式R28模型结果。
