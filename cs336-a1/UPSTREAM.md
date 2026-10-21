# Upstream 溯源与本地改动

## 来源

- 仓库：https://github.com/stanford-cs336/assignment1-basics
- 本地快照：2026-10-16（`cs336_assignment1_basics.pdf` 等文件时间）
- 项目版本：`26.0.0`
- 注意：克隆后已剥离 `.git`，未记录 upstream commit SHA；本目录**不再与上游同步**。

## 相对上游的改动

| 文件 | 改动 | 原因 |
|---|---|---|
| `pyproject.toml` | `requires-python` 由 `">=3.12,<3.14"` 改为 `">=3.14"` | 整个仓库统一使用 Python 3.14 的单一 venv |
| `uv.lock` | 已删除 | 锁文件由仓库根的 `uv.lock` 统一持有（workspace 成员锁会被忽略） |
| `.git/` | 已删除 | 已并入 Transformer-Infra-Lab 单一 git 根 |

## 约束

- `tests/` 是官方契约，不要修改；所有实现通过 `tests/adapters.py` 接入。
- 包布局必须保持平铺（`module-root = ""`）：`cs336-a1/cs336_basics/`。
- 环境操作在仓库根执行：`uv sync --all-packages`；跑测试先 `cd cs336-a1`。
