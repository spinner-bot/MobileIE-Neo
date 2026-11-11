# R7c：匹配三通道状态的运动对齐与输出融合控制

创建：2026-11-11T17:35:36.332666+08:00；作者：Codex  (sqs_A100)；原因：R7b静态和旧合成改善，实拍selected后混合SSIM却由.78768降到.74310，需要确认运动错位而非继续盲目训练。

本轮为看过R7b后的适应性诊断，不训练、不新增视频。固定官方权重与R5同38条件。

## 方法与严格信息对照

- feature_selected_post：仍缓存[6,7,8]三个原始body FST通道，混合发生于平方后，有限一帧原始历史。
- output_finite：缓存前一帧原始clamp后RGB，有限两帧混合。
- output_recursive：缓存前一帧融合RGB，同样三个通道，但信息可累积多帧，明确披露。

三方法各固定gamma .1/.5，皆不在测试搜索。各比较zero（直接无对齐）、teacher_half_fb（R5共享半尺度增强域flow/mask）、reference_proxy（共享GT图像估计flow/mask；合成事件为已知运动）。reference_proxy是不可部署诊断，实拍GT运动仍非真值。

每项比较无reset/沿R2输入事件reset，保持相同决策；reset时使用当前帧，切断方法自己的历史。所有方法使用相同motion/confidence并交同一PyTorch bilinear grid_sample，align_corners=True，当前坐标加当前→前帧flow，边界与mask相交；zero直接取历史，不引入整数grid浮点插值误差。状态空间分辨率始终原图。

3方法×2gamma×3motion×2reset×38=1368行，官方raw基线按两种reset另76行，共1444行。无额外output half组合，避免重复增加信息深度。保存逐帧与事件frame12。

## 代价与重放

“同三通道状态”只指**给定motion条件下的融合状态**，不能当端到端部署成本：估计teacher motion需要历史RGB，特征方案还可能需要额外原始teacher输出/前向，预计算motion不是免费。reference_proxy不可部署。本轮不宣称手机或端到端速度优势。

重放feature_selected_post、zero、gamma .5的R7b结果，要求相同reset下全部主要序列指标差≤1e-8。output_recursive半尺度gamma .1与R7b官方half01作CV2/PyTorch采样差异记录，不要求位相同；所有本轮主比较使用共同采样器。

先核验方向/边界mask、无对齐重放、有限/递归状态行为与首帧等价。GPU2特征+基线532行，GPU3两输出控制912行，每项新guard检查，同组最多四空闲卡；固定算法预算，无墙钟截断。若对齐后仍无相对同状态输出融合优势，优先缩减该结构方向；如果只有reference_proxy有效，优先研究对应可靠性，不将其当可部署成功。
