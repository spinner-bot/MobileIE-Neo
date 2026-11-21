# 18 · 演示分支以no-ff合入main

日期：2026-11-21；作者：Codex  (sqs_A100)。

按用户明确要求，将`demo/org`合入`main`，替代此前演示分支不合回main的约定。独立演示分支保留，后续合并继续使用`--no-ff`，不rebase、squash或强推。

- 原main：`bcc4198a55b91785a4a2473d4532c7017fe2160e`。
- 演示源提交：`055e310308151e6c25781dadbf74128c84309d76`。
- no-ff合并提交：`c8139cf34dfde04be0e1bf84469a554d62ac71a8`，两个父提交依次为原main和演示源提交，无冲突。
- 集成工作区：`/data/Users/zrh/repo/MobileIE-Neo-main-integration`，分支为main。

原项目目录已由其他工作切换到`research/O1`，本次没有在该目录切分支、提交或改文件。survey/pre及demo/org的工作区也保持原分支。

合入Android0.2.0、Python0.2.1、三任务模型与输入契约、两种展示、开发工具脚本、历史模拟/真机/Windows验收证据及会议材料。原`research/6.mobile-deployment/`演示资料迁至`demo/org/`；研究2/4/5/8、原始MobileIE、paper、研究scripts/specs及main原AGENTS均与原main逐路径比较，内容不变。演示App、桌面和脚本则与演示源分支完全一致。

集成后的实际检查：

| 检查 | 结果 |
| --- | --- |
| Android安装包与仪器测试包构建 | assembleDebug、assembleDebugAndroidTest通过，构建38秒 |
| Android Lint | 0错误、9条既有警告（固定依赖及中文国际化提示） |
| Python核心与真实Tk GUI回归 | Linux/Xvfb 8/8通过、0跳过，13.512秒 |
| 三任务数值与界面 | 9个样例、两窗口联动、任务标题、失败保留baseline及小窗口通过 |
| 合并历史与源码保持 | 两个父提交完整；研究路径及演示程序来源核对通过 |

原始日志、数值与界面结果见[集成验收证据](../results/main-integration-20261121/acceptance.json)。本次没有重跑Android26项模拟器仪器测试或真机/Windows原生测试；已有对应证据完整合入，不将其标成本次新执行。重编译产物只用于集成检查，原有发布包不替换。

main根README和演示入口已更新集成状态，并修正桌面验收证据相对链接。历史报告保留原当时约定；本记录与用户最新指令为当前集成依据。完成验收后阶段推送main。
