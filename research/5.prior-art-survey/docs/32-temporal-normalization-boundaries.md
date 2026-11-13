# 时间归一化不等于新的可折叠视频方法

核查：2026-11-12T13:46:29.005916+08:00；作者：Codex  (sqs_A100)。R42固定空间对照结束后的有界先例核查。

[DNR，NeurIPS2021正文](https://proceedings.neurips.cc/paper/2021/file/5bd529d5b07b647a8863cf71e98d651a-Paper.pdf)阅读引言及§3.1–3.3：已研究相邻帧/层之间的归一化参数依赖，使用输入相关的递归结构预测线性尺度与偏移，并在训练与推理都使用动态参数。这是动作识别研究，不据此推断低光增强有效或可以折叠回固定单帧卷积。

FLAIR顺序GN的检索片段提示时间归一化先例，但[补充PDF](https://openaccess.thecvf.com/content/WACV2025/supplemental/Zou_FLAIR_A_Conditional_WACV_2025_supplemental.pdf)本次403，未完成全文核验，不以片段证明精确实现或效果。

本项目推断：仅交换B/T维度或把同视频帧凑成BN组，不足以主张新颖性；还改变了批内分布及running-stat更新语义。动态、依赖输入或历史的归一化不能一般性吸收为输入无关的固定卷积。R41收益只证明已知展开训练在当前受控条件下较强，尚未证明时间BN机制。暂不以“视频BN”命名再开局部打包网格。
