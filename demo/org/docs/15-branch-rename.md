# 15 · 演示分支改名 `demo.org` → `demo/org`

- 执行：2026-11-14 · claude code (local)，本地 Windows 11 + GitHub + 服务器 `sqs_A100`
- 依据：用户 2026-11-14 要求把演示分支改名，并在本地、GitHub、服务器三端同时完成、保持同步；改名过程按版本管理规范小步提交与推送。
- 结论：**三端分支名已统一为 `demo/org`，指向同一提交 `b6236be7`**；旧名 `demo.org` 在 GitHub 上已删除，服务器旧工作树目录名保留未动。

## 1. 改名前的同步

先 `git fetch origin`，把远端 34 个提交（双端 0.2.0 三任务升级与验收证据）以 `git merge --no-ff` 合入本地，得到 `b6236be7`，避免改名丢提交。合并只为同步，无冲突。

## 2. 三端操作

| 端 | 操作 | 结果 |
| --- | --- | --- |
| 本地 | `git branch -m demo.org demo/org`，并把 `b6236be7` 推到旧名后删除本地旧名 | `demo/org` 跟踪 `origin/demo/org` |
| GitHub | 推送 `b6236be7` 至 `demo.org`（保住历史）→ `git push -u origin demo/org` 建新分支 → `git push origin --delete demo.org` 删旧分支 | 远端只剩 `refs/heads/demo/org` |
| 服务器 | 演示工作树 `/data/Users/zrh/repo/MobileIE-Neo-demo-org` 内 `git branch -m demo.org demo/org`；`git fetch --prune` 后设上游并 `--ff-only` 对齐 `b6236be7` | 服务器 `demo/org` 与远端同提交、工作树干净 |

删除 GitHub 旧分支前已用 `git ls-remote` 核对两个 ref 指向同一提交 `b6236be7`，确认新分支已存在才删旧名。

服务器那一端是**指针对齐**：该工作树没有本地提交（原在 `70b9c736`，是远端祖先），因此使用 `--ff-only` 直接对齐，没有产生新的合并提交；这不违反「合并必须 `--no-ff`」的约定——此处没有合并任何分叉历史。

## 3. 保留未改动

- 服务器工作树**目录名**仍是 `MobileIE-Neo-demo-org`，只是目录名，不参与版本管理；改名需改动服务器上的路径引用，本次未获授权，暂不动。
- `main` 分支未受影响，服务器 `main` 及其研究进展保持原状。
- 历史文档中的日期性记录不改写；仅把指向「当前分支」的名字统一为 `demo/org`（见 `539a5ec0`、`cf05bdf4`）。

## 4. 修订记录

- 2026-11-14 · claude code (local)：本地/GitHub/服务器三端完成 `demo.org` → `demo/org` 改名，核对同提交后删除远端旧名。
