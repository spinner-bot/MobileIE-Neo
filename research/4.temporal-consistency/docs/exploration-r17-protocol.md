# R17 SDSD跨数据源准备与固定探索子集

创建：2026-11-11T20:38:00.629710+08:00；作者：Codex  (sqs_A100)；原因：在取得或解码新数据前，固定跨相机/ISO探索选片、预处理与参考，不占用最终测试。

SDSD官方commit07330bb0ac124d2d56f6e92908d650dde3f7f7c2，train/test YAML blob及全部官方测试ID见research5 process/data-readiness-r16/official-split.json。室内70减12测试，室外80减13测试。每个域按自然数字序排官方训练对，取首尾，固定indoor pair1/pair69、outdoor pair2/pair80；只取同名配对帧自然排序的前16，不据内容、噪声、指标、配对观感换片。它们是新的探索dev，不称完整SDSD官方测试，全部25官方test像素仍不解码。

先尝试官方PNG分卷获取：原生1920×1080来源、publicDrive页面与size已登记。可优先HTTP Range只取ZIP索引/指定成员；若仅支持完整下载，先记录总量、单并发获取并校验归档CRC。下载时落忽略cache，过程、大小、SHA/CRC、错误单独保留。不安装/改写共享Python环境。npy为作者处理过960×512，不能混作原PNG天然尺度；若PNG不可取得而改npy，必须先补协议且单独标域，不能静默替换。

解码后生成两准备：原生固定ROI x704:1216/y396:684与全帧AREA480×270，两者GT/input共同处理、floatRGB评价不经有损保存。缓存每源SHA/shape/配对名称，GT估计flow只是评价参考、非真实flow，配对误差不能忽略。source域为ND+提高ISO的重复机械路径，不称自然夜景。严格不得将GTflow用于部署分支。

固定零SDSD更新迁移候选：原MobileIE、StableLLVE官方clean；R14 native plain双seed4000；R16 spatial/raw_history/aligned_history双seed4000。raw/共同官方MobileIE-teacher halfFB alpha.1，真实历史/current替代和reset对照。另以R14 native plain为增强基底接FastDVDnet sigma5/15的window/five_current，单独注明两未来。不得按该dev挑checkpoint、sigma或组合，全部固定项报告；若任何加载/来源检查失败，报告失败而非换更好模型。

这是跨数据源探索，无新颖性或公平SOTA排名声明。后续独立正式验证还需完整官方测试协议、充分训练、自然夜景/更复杂动态、真实全链路效率。获取可访问性先在CPU推进，GPU评价待当前R16完成后另经空闲预检，不设墙钟截断。
