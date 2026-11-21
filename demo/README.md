# demo/ · 演示工作区

本目录存放所有**演示性质**的工作，与 `research/` 下的核心研究分开。

## 目录约定

| 目录 | 分支 | 内容 |
| --- | --- | --- |
| `demo/org/` | `demo/org` | MobileIE 手机端部署与项目申报现场演示（原 6 号工作）；2026-11-21已集成至main |

后续新增演示按 `demo/<名称>/` 建目录、`demo/<名称>` 建分支。

## 工作约定

1. 本仓库是唯一数据站：日志、结论、截图、素材、脚本全部提交入库；
2. 小步、密集提交，阶段性推送；
3. 与 `main` 同步使用 `git fetch origin` + `git merge --no-ff origin/main`，禁止 `git pull`；
4. 演示在独立分支开发，向`main`集成时显式使用`--no-ff`。2026-11-21用户已要求将`demo/org`合入`main`，见[本次集成记录](org/docs/18-main-integration.md)；其他研究分支及工作区保持原状。
