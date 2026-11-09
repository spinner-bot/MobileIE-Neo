# MobileIE-Neo

面向手机端的轻量图像增强模型研究：在 MobileIE（极轻量实时图像增强 ConvNet）基础上，解决其应用于视频时产生的帧间闪烁问题。

## 目录结构

| 目录 | 用途 |
| --- | --- |
| `MobileIE/` | 上游官方代码（vendored，提交 `7030d12`）与预训练权重，各工作共用 |
| `docs/` | 进展报告（Markdown） |
| `specs/` | 项目规范与需求文档，如 `项目规范/版本管理规范.md` |
| `research/` | 各阶段工作的中间资料与辅助内容（前期准备、复现、视频闪烁） |
| `assets/` | 图片等静态资源 |

## 项目规范

版本管理遵循 [`specs/项目规范/版本管理规范.md`](specs/项目规范/版本管理规范.md)：频繁提交、SSH 提交、AI 提交作者标注 `claude code (local)` / `Codex (local)`、禁用 fast-forward 合并。

## 快速开始

```bash
git clone git@github.com:spinner-bot/MobileIE-Neo.git
cd MobileIE-Neo
```

<!-- 环境安装、代码运行说明待补充 -->
