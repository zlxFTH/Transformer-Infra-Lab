# attn-peak 项目规划（Transformer 全流程 → AI Infra 算子方向）

> 本文整理自一次完整讨论，覆盖：AIInfraGuide 第 3 章解读、CS336 五个作业拆解、项目形态与命名、
> 学习路线、简历写法，以及对 DeepSeek「高性能算子/通信/编译器工程师」JD 的对齐分析。
>
> 整理时间：基于 2026 Spring 的 CS336 课程页面与仓库内容（课程每学期会更新，执行前请复核）。

---

## 0. 结论速览（TL;DR）

| 问题 | 结论 |
|---|---|
| 只看 AIInfraGuide 第 3 章，能满足"自己写一遍 Transformer 全流程"吗？ | ❌ 不够。教程是原理+代码片段，没有测试和验收标准 |
| CS336 A1 能满足吗？ | ✅ 能满足"Transformer 全流程细节"的骨架，是目前最扎实的选择 |
| CS336 A1 能当简历项目吗？ | ⚠️ 它是**入场券**，不是交付物。写进简历的是 A2 那条线 + 自己做增量 |
| 两份资源的关系 | 互补。教程 = 概念补课（1 周），CS336 A1 = 动手主干（3~4 周），项目增量 = 推理侧/性能数据（2~3 周） |
| 目标 JD 的核心要求 | 算子（CUDA/Ascend C/TileLang）、通信（NCCL/DeepEP）、编译器（MLIR/TileLang/Triton）三选一深挖 |
| 项目命名 | **`attn-peak`**（实测可用） |
| 总周期 | 约 15~16 周 ≈ 3.5~4 个月（每周 10~15 小时） |
| 算力预算 | 约 $100~200（Kaggle/Colab 免费额度 + 按小时租卡） |

---

## 1. 目标对齐：DeepSeek「高性能算子/通信/编译器工程师」JD

来源：[DeepSeek 招聘 - 高性能算子/通信/编译器工程师](https://app.mokahr.com/social-recruitment/high-flyer/140576#/job/aed2a32c-189f-4338-b901-9b45d2b7edfd)（杭州/北京，实习/全职）

### 1.1 团队定位（JD 原话）

> 我们构建 AGI 与物理芯片之间的最后一层软件……团队的唯一标准：**硬件的物理极限**。我们不和任何 baseline 或者其他实现比较性能差异，
> **只关心当前设计和实现距离理论上限还有多远**，对每一个 cycle 和每一瓦功耗有着近乎偏执的追求。

### 1.2 三条职责

1. **算子**：针对 GPU/NPU，用 **CUDA / Ascend C / TileLang** 设计实现并长期维护 GEMM、Attention 及其他算子
2. **通信**：设计实现通信算子（DeepEP 中的 EP、CP/DP、PP），调优排障 NCCL / HCCL
3. **编译器**：参与面向深度学习的 DSL 编译器（如 TileLang）研发，扩展 DSL/IR/CodeGen 能力
4. **贴合模型演进**：与框架/研究员共同设计硬件友好的模型结构，对真实训练推理负载做性能分析

### 1.3 核心要求与加分项

- **核心要求**：熟悉 C++、Git；对细节严谨、对性能优化有热情；良好中文沟通
  （JD 明确写：*"我们不在意你是否有相关背景……充分欢迎有热情的白纸同学加入"*）
- **加分项**：
  1. CUDA / Ascend C 算子经验，熟悉 CUTLASS、DeepGEMM，掌握 PTX ISA、Ascend C API、memory model
  2. 熟悉 NCCL/HCCL/NVSHMEM/DeepEP 等通信库底层实现，有调优排障经验
  3. 深入理解 RDMA 原理与 RoCE/InfiniBand 架构，熟悉 IBGDA 等特性
  4. **编译器或 DSL 系统开发经验，熟悉 MLIR / TileLang / Triton 任一编译栈**
  5. **在大型开源软件（LLVM、NCCL、PyTorch）有深度贡献，或维护过其他个人开源软件**
  6. HPC 赛事（SC/ASC/ISC/PAC）或算法竞赛（NOI/IOI/ICPC）优异成绩

### 1.4 对你的三个直接推论

1. **拉开差距的不是模型知识，是 CUDA/C++ 算子能力**。Transformer 架构是"必要背景"，不是差异点。
2. **加分项第 4 条（Triton/TileLang）和第 5 条（个人开源软件）你都能低成本命中**——Triton 写 FlashAttention、公开算子仓库 + 博客。
3. **JD 那句"只关心距离理论上限还有多远"就是你项目的写作纲领**：不写"我实现了 FlashAttention"，写"我达到峰值算力的 X%，瓶颈在哪"。

---

## 2. AIInfraGuide 第 3 章：逐节定位与取舍

**章节目录**（[模块一 · 第3章](https://caomaolufei.github.io/AIInfraGuide/prerequisites)）：
3.1 为什么必须懂 Transformer｜3.2 全貌及代码实现｜3.3 Self-Attention 深入｜3.4 FFN 深入｜3.5 位置编码深入｜
3.6 LayerNorm 与残差深入｜3.7 Decoder Block 完整解析｜3.8 从 Transformer 到 LLM 自回归生成｜3.9 Tokenization 与词嵌入
＋《[Transformer架构：快速入门篇](https://caomaolufei.github.io/AIInfraGuide/guides/模块一-前置知识/transformer/transformer架构快速入门/)》（第 3 节 = Self-Attention 机制）

### 2.1 与 CS336 的对照总表

> **判断：CS336 覆盖约 55~60%，缺的恰好是"工程与系统"那半。**
> 更关键：CS336 覆盖的是"能写出来"，教程覆盖的是"能算出来、能讲清楚"。面试考后者。

| 教程章节 | CS336 覆盖情况 | 判定 |
|---|---|---|
| 3.1 为什么必须懂 Transformer | 无直接对应；Lec 2 资源账本 + Lec 5 GPU/TPU 讲得更硬 | 🔁 概念重叠，教程更直白 |
| 3.2 Transformer 全貌及代码实现 | Lec 3 Architectures + A1 全覆盖 | ✅ **可跳过** |
| 3.3 Self-Attention 深入（12 节） | §1~8、§10、§11 由 Lec 3 + A1 覆盖；§12 FlashAttention 概念由 Lec 6 + A2 覆盖（档次更高） | ⚠️ **只补 §9** |
| 3.4 FFN 与激活函数 | SwiGLU/激活由 Lec 3 + A1 覆盖；MoE 由 Lec 4 覆盖 | ⚠️ 补张量并行切分 + kernel 融合 |
| 3.5 位置编码 | RoPE/Sinusoidal/ALiBi/NoPE/长上下文由 Lec 3 + A1 覆盖 | ⚠️ 补 §7 长上下文扩展 |
| 3.6 LayerNorm 与残差 | RMSNorm/LayerNorm/Pre-Post-Norm 由 Lec 3 + A1 覆盖 | ⚠️ 补 §5 CUDA 优化视角 |
| 3.7 Decoder Block 完整解析 | 数据流 + 参数量/FLOPs 由 Lec 2/3 + A1 覆盖 | ⚠️ 补 §8 显存规划 |
| 3.8 自回归生成（9 节） | Lec 10 Inference 部分覆盖，量化/投机解码更详细 | ⚠️ 补 PagedAttention/Prefix Cache/解耦 |
| 3.9 Tokenization 与词嵌入 | Lec 1 Tokenization + A1 的 BPE 覆盖更深 | ✅ **可跳过** |
| 快速入门篇 §3.3 O(N²) | A2 直接手写 kernel，档次更高 | ✅ 可跳过（**但 §7 Online Softmax 别跳**） |

### 2.2 教程覆盖不到的（CS336 结构性缺口）

CS336 的作业是**代码中心**，讲义是**研究前沿中心**，中间那层"经典推导 + 设计动机"要自己补，教程刚好填在这里：

- "为什么除以 √d_k"的**方差推导**（3.3 §6）
- RoPE 中 $q_m^\top k_n$ 只依赖 $m-n$ 的**高维证明**（3.5 §5.4）
- Attention 的**信息检索直觉**与历史演进（3.3 §1~3）
- Softmax 数值稳定性的原理（3.3 §7）

> **一句话：CS336 给你"手"，教程给你"嘴"。面试两样都要。**

### 2.3 必读清单（优先级排序）

**🔴 P0 — 硬缺口，跳过会直接卡住（2~3 天）**

1. **3.8 §6.3~§8**：KV Cache 数据结构 / 显存计算 / 碎片化 / PagedAttention / Prefix Cache / KV 量化 / Prefill-Decode 解耦
   → A1 **完全不涉及 KV Cache**（A1 的 generate 是每步全量重算），这块是真空
2. **3.7 §8 显存规划详解**：训练态四大组成、推理态权重+KV、LLaMA-2-7B 完整推演（面试白板题标准答案）
3. **3.3 §9**：MHA vs MQA vs GQA vs MLA（含对比表和 MLA 拆解）——想要 GQA 对比实验的唯一系统材料

**🟠 P1 — 工程选题与并行细节（1~2 天）**

4. **3.4 §7（张量并行切分 FFN）+ §9（Kernel 融合）**：列切分+行切分正是 CUTLASS/DeepGEMM 那一层的切分方式；融合是 kernel 选题来源
5. **3.6 §5（CUDA 优化视角）**：LayerNorm kernel 挑战、Residual+LN 融合、FP16/BF16 数值精度 → **LayerNorm 融合是 Reduce/Softmax 后最合适的第三个 kernel 练习**

**🟡 P2 — 数学直觉与面试表达（半天）**

6. 3.3 §2/§6/§7（QKV 直觉、√d_k 推导、数值稳定性）
7. 3.5 §5.4/§5.6（RoPE 相对位置性质证明、与 Sinusoidal 的联系）

**⚪ 跳过**

- 3.9 全章、3.2 §9（代码实现）、3.3 §4/§10/§11、3.5 §3.3
- 快速入门篇 §3.3（O(N²) 复杂度）

**🔑 唯一例外（必须读透）**

- 3.3 深入篇 **§7.2/§7.3**（减最大值技巧、Online Softmax）
- 快速入门篇 **§7.3**（KV Cache 显存账本）
  → A2 写 FlashAttention 全靠这些；不读透写 kernel 时会卡在数学上

### 2.4 第 3 章只是"索引"，深挖在后续模块

| 主题 | 第 3 章 | 深挖位置 |
|---|---|---|
| KV Cache / PagedAttention / Continuous Batching | 3.8 | **模块四 第 2 章**（推理优化） |
| 张量并行 / 流水线并行 | 3.4 §7 | **模块三 第 6/7 章**（分布式训练） |
| FlashAttention / GEMM / Softmax kernel | 3.3 §12 | **模块二 第 3~6 章**（CUDA 编程与算子优化） |
| 通信原语 / NCCL | 3.1（提及） | **模块三 第 2 章** + 模块一第 6 章 |

> **所以第 3 章别贪多，留力气去模块二/三/四。**

---

## 3. CS336 五个 Assignment 拆解

课程：[cs336.stanford.edu](https://cs336.stanford.edu/)（2026 Spring，Hashimoto & Percy Liang）
设计思路：模仿操作系统课"从零写一个 OS"，带你走完 LLM 的**完整生命周期**——数据收集清洗 → Transformer 构建 → 训练 → 评估 → 部署。

### 3.1 总表

| # | 名字 | 一句话 | 你写的核心东西 | 发布 → 截止 |
|---|---|---|---|---|
| A1 | **Basics** | 从零实现标准 Transformer LM 并训练 | BPE 分词器、模型全部组件、优化器、训练循环 | 3/30 → 4/15 |
| A2 | **Systems** | 让 A1 跑得快：profile + 自己的 FlashAttention + 分布式 | Triton 版 FlashAttention2、分布式训练代码 | 4/15 → 4/29 |
| A3 | **Scaling** | 理解每个组件的作用 + 拟合 scaling law 预测 | 查询训练 API、拟合 scaling law、组件消融 | 4/29 → 5/6 |
| A4 | **Data** | 原始 Common Crawl → 可用预训练数据 | 抽取、质量过滤、去重、数据配比 | 5/6 → 5/20 |
| A5 | **Alignment** | SFT + RL，让模型学会做数学题 | SFT、GRPO（+可选 DPO/安全对齐） | 5/20 → 6/3 |

### 3.2 A1 · Basics（[仓库](https://github.com/stanford-cs336/assignment1-basics)）

- **硬性约束**：不许用 `torch.nn.Linear` / `nn.Embedding`，从 `nn.Parameter` 开始写
- **要手写**：
  - 数据层：BPE 训练（并行预分词、特殊 token）、`Tokenizer` encode/decode
  - 模型层：`Linear`(no bias) / `Embedding` / `RMSNorm` / `SiLU` / `SwiGLU` / `RoPE` / 数值稳定 `Softmax` /
    `scaled_dot_product_attention` / `multihead_self_attention`(带 RoPE) / `transformer_block` / `transformer_lm`
  - 训练层：`cross_entropy` / 手写 `AdamW` / 余弦 LR + warmup / 梯度裁剪 / `get_batch` / checkpoint
- **验收**：`pytest` + 快照测试（`tests/_snapshots/*.npz`），接 `tests/adapters.py`，**要求数值一致**
- **训练目标**：TinyStories（小） + OpenWebText（GPT-2 small 规模），含学习率扫描、消融、leaderboard
- **地位**：后面 A2/A4/A5 都复用 A1 的模型代码 → **A1 写得越干净，后面越轻松**
- **参考复盘**：[XLOverflow 的 A1 记录](https://xloverflow.github.io/zh/2026/02/09/CS336/CS336-Assignment-1-Building-a-Transformer-Language-Model-from-Scratch/)
  （含 LR 扫描、批量大小研究、消融：去 RoPE Δloss +0.074 最大，SwiGLU +0.068，RMSNorm +0.043，Post-Norm +0.012）

### 3.3 A2 · Systems（[仓库](https://github.com/stanford-cs336/assignment2-systems)）★ 最重要

- **前半（单卡性能）**：用 `torch.profiler` + Nsight 给 A1 的模型/单层做 profile 与 benchmark，然后**用 Triton 自己写 FlashAttention2**
- **后半（分布式）**：把 A1 的训练代码改成省显存 + 分布式的版本（数据并行、分片优化器状态/梯度/参数）
- **目录结构**：`cs336-basics/` 放 A1 代码当底座，`cs336_systems/` 基本是空的——**你从头搭**
- **对你的价值**：五份作业里**唯一直接对口那份 JD**；Triton ↔ TileLang 同源（加分项第 4 条）

### 3.4 A3 · Scaling（[仓库](https://github.com/stanford-cs336/assignment3-scaling)）

- ① 消融实验理解每个组件作用；② 用一批小规模训练结果**拟合 scaling law**，预测更大模型的 loss 和最优超参
- **工程细节**：官方提供托管训练 API（`hyperturing.stanford.edu:8000`，学生用学号当 API key）
- **自学注意**：非学生没有学号，但仓库有 `--extra server` 分支，可自己用 Modal 下数据、跑 `cs336_scaling/training/run.py` 起服务
  → **自学性价比最低的一份**（要么烧钱要么造轮子）

### 3.5 A4 · Data（[仓库](https://github.com/stanford-cs336/assignment4-data)）

- 把 Common Crawl 的 **WET 文件**变成数据集：抽取、语言识别过滤（`is_english`）、质量过滤、**去重（精确 + 近似/MinHash）**、数据配比
- **硬约束**：仓库带官方训练代码（`cs336_basics/`），**只能改数据部分，不能动训练逻辑**——保证 leaderboard 比的是数据质量
- **对你**：价值中等偏低（非该 JD 核心），但聊到预训练数据处理时不露怯

### 3.6 A5 · Alignment（[仓库](https://github.com/stanford-cs336/assignment5-alignment)）

- **要做**：SFT → RL（**GRPO**），让模型学会解数学题；测试 `tests/test_grpo.py` + `tests/adapters.py`
- **可选**：安全对齐 / 指令微调 / RLHF（DPO 等）
- **环境坑**：先 `uv sync --no-install-package flash-attn` 再 `uv sync`（`flash-attn` 安装方式特殊）
- **对你**：价值低（后训练与算子岗无关）

### 3.7 自学须知

1. **验收方式分两类**：A1/A5 有单测+快照（学习体验最好）；A2 靠 benchmark 数据；A3/A4 靠 leaderboard（**没有标准答案，自学最难自评**）
2. **算力需求递增**：A1 小卡/免费额度够；A2 要现代卡（最好 H100+，否则结论没意义）；A3 官方给 API；A4 吃 CPU+存储；A5 要真实推理+RL
3. **课程节奏**：约 10 周，五份作业各约 2 周，官方 6 个 late day（每份最多 3 天）
4. **Honor code（重要）**：明确要求不参考现成实现、**不公开自己的答案**、不用 AI 直接解题（甚至建议关 IDE 补全）
   → **作业代码放私有仓库；公开项目必须是重新组织的独立实现**
5. **对你的优先级**：A1（必做，3 周）> A2（重投入，4 周+）> A3（有余力）> A4/A5（跳过）

---

## 4. 项目形态：`attn-peak`

### 4.1 CS336 的产物是什么（破除"没有 demo"的困惑）

**CS336 不产出"一个产品"，产出的是四类"证据"**：

| 作业 | 实体产物 | 它的"demo" |
|---|---|---|
| A1 | 词表文件、模型代码、**训练好的 checkpoint**、loss 曲线 | `python generate.py --ckpt ckpt.pt --prompt "Once upon a time"` → 输出连贯小故事 |
| A2 | Triton kernel 文件、benchmark 脚本、**性能表**、Nsight 截图 + roofline 图 | 跑 benchmark 打印一张对比表（延迟/显存/TFLOPS/峰值占比） |
| A3 | 消融 loss 表、**拟合的 scaling law 曲线** | "预测 vs 实测"对齐图 |
| A4 | 自清洗数据集 + 清洗前后统计 | "原始数据 loss vs 我的数据 loss"曲线 |
| A5 | SFT+GRPO 权重、奖励曲线、数学题通过率 | 同一题 base 答错 / 我的模型答对 |

**A1 的 demo 就是终端生成一段小故事**（参考复盘中贴的样例：有开头、对话、人物一致性）。
**A2 的 demo 就是一张性能表 + 一张 roofline 图**——比任何网页动效都值钱。

### 4.2 项目定位

> **`attn-peak`**：手写 attention 算子，从朴素实现到 Triton FlashAttention2，
> 用 roofline 模型回答"离硬件上限还有多远"

**README 纲领**：不写"我实现了 FlashAttention"，写"我达到峰值算力的 X%，瓶颈在哪、下一步动什么"。

### 4.3 五阶段路线

| 阶段 | 做什么 | 产出（面试证据） |
|---|---|---|
| 1. 数据 | BPE 训练 + encode/decode，和 `tiktoken` 的 GPT-2 词表对拍 | "编码结果与参考实现 100% 一致" |
| 2. 模型 | 手写 Linear/Embedding/RMSNorm/RoPE/SwiGLU/MHA/Block/LM Head，Pre/Post-Norm 都支持 | 逐模块单测 + 与 HF 对应层 logits 对齐误差 < 1e-5 |
| 3. 训练 | 手写 AdamW、cosine+warmup、梯度裁剪、bf16、梯度累积、checkpoint 续训 | "bf16 相比 fp32 吞吐 ×1.8、显存 −40%" 实测表 |
| 4. 推理 | **自己写 KV Cache**、prefill/decode 分离、temperature/top-p、batch 生成；**GQA 版本对比 KV 显存** | "KV Cache 让第 N 个 token 延迟降到常数；GQA(8:1) 把 KV Cache 压到 1/4，吞吐 ×2.3" |
| 5. 消融 | 去 RoPE / Post-Norm / SwiGLU→SiLU / 有无 KV Cache / 有无 bf16 对比 | 一张 Markdown 结果表 + 曲线图 |

### 4.4 核心交付物：性能表（这是简历的主体）

| 实现 | 延迟 (ms) | 峰值显存 (GB) | 有效 TFLOPS | 峰值算力占比 | 峰值带宽占比 | ncu 判定瓶颈 |
|---|---|---|---|---|---|---|
| 朴素 PyTorch（materialize N×N） | | | | | | memory-bound |
| 手写 CUDA naive | | | | | | |
| + shared memory tiling | | | | | | |
| + online softmax（无中间矩阵） | | | | | | |
| Triton FlashAttention2 | | | | | | |
| `F.scaled_dot_product_attention` | | | | | | |

- 参数：N ∈ {512, 1024, 2048, 4096}，d_head = 64/128，fp16/bf16，H100 或 B200
- 配 **roofline 图**（横轴 arithmetic intensity，纵轴 TFLOPS，标出你的点与硬件屋顶线）
- 配 Nsight Compute 截图（`sm__throughput` / `dram__throughput` / achieved occupancy / bank conflict）
- **结论段**："在 N=4096 下我的 kernel 达到峰值算力的 X%，卡在 shared memory 带宽 / 寄存器压力 / occupancy=Y，下一步动 Z"

### 4.5 两个仓库的分工

| 仓库 | 内容 | 是否公开 |
|---|---|---|
| `attn-peak` | 独立重写的算子集 + benchmark + roofline + 博客链接 | ✅ **公开（简历主角）** |
| `cs336-assignments` | A1/A2 作业本体 | ❌ **私有**（honor code） |

公开仓库 README 需注明："受 Stanford CS336 课程启发，独立实现"。

### 4.6 想加"能看的 demo"？三样就够

1. **`demo.py` + 终端录屏**（30 分钟）：加载 checkpoint，交互式输入 prompt 实时输出，录 GIF 贴 README 首屏
2. **对比图/表**（1 小时）：与 `nanoGPT` / GPT-2 在相同数据上的 loss 对齐曲线
3. **一个服务**（可选，2 小时）：FastAPI 包 `generate()`，`POST /generate`——对推理/部署岗价值大

> ⚠️ **别为了好看搭前端**。对算子岗，一张 roofline 图 > 一个 Gradio 聊天框。

---

## 5. 命名方案（已实测可用）

### 5.1 命名四条标准

1. **一眼看出是底层算子活**（面试官扫简历 3 秒）
2. **中文念得顺**
3. **不用 `my-` / `mini-` 前缀**——`mini-` 在简历上暗示"教学级"，与"逼近峰值"的目标相反
4. **不和个人站点/知名项目重名**（`minitorch`、`flash-attn`、`nano*` 都已被占）

### 5.2 推荐（GitHub 实测结果）

| 排名 | 名字 | 含义 | 实测 |
|---|---|---|---|
| 🥇 | **`attn-peak`** | 让 attention 逼近硬件峰值 | 仓库名可用，org 名也空 |
| 🥈 | `attnlab` | attention 算子实验室 | 仓库名、org 名都空 |
| 🥉 | `kernel-peak` | 同上，但不局限 attention | 仓库名、org 名都空 |
| 4 | `gpu-ops-lab` | GPU 算子实验室 | 可用 |
| 5 | `attnforge` | 锻造 attention kernel | 可用（`-forge` 略生僻） |

**建议选 `attn-peak`**：把核心命题写进名字里，面试可用来开场——
*"这个项目叫 attn-peak，因为它回答的问题是：我的 attention kernel 离硬件上限还有多远。"*
若打算从 attention 扩到全算子，用 `kernel-peak`。

### 5.3 明确不推荐

| 名字 | 原因 |
|---|---|
| `mini-llm-from-scratch` | `mini-` 自我降级；A1 是背景不是主角 |
| `flash-attn-from-scratch` | `flash-attn` 是 Dao-AILab 官方项目名 |
| `nano-attn` / `nano-transformer` | `nano` 系列已被 nanoGPT/nano-vllm 占据，且暗示"简化版" |
| `minitron` | 41 个同名仓库 + NVIDIA 同名项目 |
| `llm-kernels` | 太泛，无记忆点 |
| `my-transformer` / `transformer-impl` | 一眼课程作业 |
| 中文名/拼音 | URL 难念、难输入 |

### 5.4 配套三件套（名字三处一致）

1. **仓库 description**（GitHub 搜索命门）：
   `Attention kernels from scratch — naive → CUDA tiled → Triton FlashAttention2, with Nsight profiling and roofline analysis of the gap to hardware peak.`
2. **README 首行**：`attn-peak · 让 attention kernel 逼近硬件峰值`
3. **博客 tag**：给 [zlx.moe](https://zlx.moe/) 建 `attn-peak` 标签，踩坑系列都挂上去
   → 简历同时给 `github.com/zlxFTH/attn-peak` + `zlx.moe/tags/attn-peak`，
   正好命中 JD 加分项第 5 条"维护过个人开源软件"

**GitHub topics 建议**：`cuda` `triton` `flash-attention` `gpu-optimization` `roofline` `benchmark`

---

## 6. 学习路线（15~16 周）

### 6.1 基准假设

- 起点：Python / C++ 能写简单程序，会查文档边做边学（**不需要先刷完《C++ Primer》**）
- 节奏：每周 10~15 小时
- 算力：租云 GPU（Modal / RunPod / Lambda / Nebius）

### 6.2 逐阶段安排

| 阶段 | 内容 | 时间 | 在哪做 |
|---|---|---|---|
| 0 | 教程必读清单（见 §2.3）+ 快速入门篇 §3.2/§3.3/§5.3/§6.3/§7.3 | **1 周** | 本机 |
| 1 | CS336 A1 全绿 + TinyStories 训练跑通（**跳过 OWT 和 GQA/KV Cache 增量**） | **3 周** | 本机 + 免费 T4 |
| 2 | C++/CUDA 地基：模块二 1.1→1.4，写 vector add，**马上用 `ncu` 看 DRAM 吞吐** | **2 周** | 租卡（便宜卡即可） |
| 3 | Reduce → Softmax/Online Softmax（模块二 3.1、5.1、5.2）：**第一个能写进简历的 kernel** | **2~3 周** | 租卡 |
| 4 | **Triton 手写 FlashAttention2（A2 前半）+ 朴素对照 + roofline 分析** ← 简历项目主体 | **4 周** | 租卡 |
| 5 | CUDA GEMM（模块二 4.1，CUTLASS 风格 tiling）**或** A2 的分布式部分（二选一） | **3 周** | 租卡 |
| | **合计** | **约 15~16 周** | |

### 6.3 三个关键判断

1. **Triton 放在 CUDA C++ 之后但重点做**：Triton 用 Python 写 kernel，让你先把精力集中在"分块、数据复用、online softmax"这些**真正的优化思想**上，
   而不是先和 `__shared__` / `cudaMalloc` / 指针偏移搏斗。同时它是 TileLang 的同源思路（JD 加分项第 4 条）。
   但纯 Triton 不够——阶段 2/3 的 CUDA 手写经历让你能讲清"为什么 Triton 生成的代码会 bank conflict"
2. **C++ 不单独学**：直接上 CUDA，边做边查是最快路径（JD 只要求"熟悉 C++"）
3. **阶段 3 是心理关键节点**：Reduce 和 Online Softmax 短小、可验证、优化效果立竿见影（O(N²) 显存 → O(N)）。
   **在这个节点就开始写 README 和博客**——16 周后你手上会有 4~5 篇 kernel 优化文章，比项目本身更能命中 JD 加分项

### 6.4 算力与预算

| 用途 | 建议 |
|---|---|
| A1（纯 PyTorch 小模型） | 本机 CPU/MPS 足够；小模型 5k step 很快 |
| 阶段 2 前的 kernel 练习 | Kaggle（每周约 30h 免费 T4/P100）或 Colab |
| 阶段 2~4 开发调试 | 按小时租小卡（T4/4090 级），预估 30~60 小时，**$50~150** |
| 最终 benchmark + profile | H100/B200 跑 2~3 小时，**$15~25** |

**行情参考**（课程官网列出的单卡 B200 价格，2026-03）：Modal $6.25/h（新用户每月送 $30 额度，按实际计算计费）｜
RunPod $4.99/h｜Nebius $5.50/h（抢占式 $3.05/h）｜Lambda $6.69/h｜Together $7.49/h（需 8 卡起）

**三条省钱原则**：
1. **正确性在便宜环境验，性能在现代卡上量**（CUDA/Triton 跑不了 Mac，不存在"本机验 CUDA"）
2. **profile 要在与结论匹配的卡上做**——带宽/算力占比是相对该卡屋顶线算的，T4 的结论投 B200 会被问穿
3. **抢占式 + checkpoint**：kernel benchmark 短，抢占式性价比最高；训练类实验一定落盘 checkpoint

### 6.5 现在这一周该做的三件事

1. **读**教程必读清单（§2.3），一周内读完，然后合上教程去写代码
2. **装** `uv`，clone [assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)，跑 `uv run pytest`
   ——**先看满屏 `NotImplementedError` 失败一遍**，这就是待办清单，然后按 `tests/adapters.py` 顺序逐个补
3. **建**两个仓库，公开那个先把**性能表骨架**写出来（表头 + 空行），每完成一个实现填一行
   ——这个动作会改变后面 4 个月的工作方式

---

## 7. 简历怎么写

### 7.1 原则

**别写"做了 CS336"**（面试官不关心你上过什么课）。写**产出了什么、量化到什么程度**：
每条 bullet = 「我做了什么 → 数字变了多少 → 为什么」。

### 7.2 算子方向（对应目标 JD）

> **手写 Transformer 算子库与性能分析**（[github.com/zlxFTH/attn-peak](https://github.com/zlxFTH)）
> - 从零实现 Transformer LM 全链路（BPE 分词器、RoPE/SwiGLU/RMSNorm/因果注意力、手写 AdamW），
>   不依赖 `nn.Linear`/`nn.Embedding`，实现与官方快照测试数值一致
> - 用 Triton 手写 FlashAttention2：相比朴素 PyTorch 实现，N=4096 时显存从 2.05 GB 降至 0.31 GB，
>   延迟降低 9.8×，达到 H100 峰值算力 47%
> - 用 Nsight Compute + roofline 模型定位瓶颈：shared memory 带宽利用率仅 62%，occupancy 受寄存器压力限制在 35%，
>   据此调整分块策略后吞吐再提升 12%
> - 手写 GEMM（shared memory tiling + double buffering），达到 cuBLAS 的 X%；
>   可复现的 benchmark 脚本与完整实验记录见仓库 README

### 7.3 模型/推理方向

> **从零训练的语言模型（TinyStories / OpenWebText）**
> - 从零实现 BPE 分词器并训练 10k 词表；实现完整 Decoder-only LM（RoPE + SwiGLU + RMSNorm + Pre-Norm）并手写训练栈
> - 在 TinyStories 上训练至验证困惑度 X；自回归生成样例见仓库；LR 扫描显示 lr=1e-2 最优（ppl 3.67）
> - 消融实验：移除 RoPE 掉点最多（Δloss +0.074），SwiGLU→SiLU +0.068，确认位置编码是最大贡献项

### 7.4 自查：站在面试官角度

- [ ] 每条 bullet 都有数字吗？
- [ ] 有没有一条讲"如何定位瓶颈"？（JD 只关心这个）
- [ ] 名字/仓库/博客三处一致吗？
- [ ] 面试时能就其中任意一条聊 30 分钟吗？

---

## 8. 参考链接

**课程与作业**
- [CS336 课程主页（2026 Spring）](https://cs336.stanford.edu/)
- [A1 Basics](https://github.com/stanford-cs336/assignment1-basics)｜[A2 Systems](https://github.com/stanford-cs336/assignment2-systems)｜
  [A3 Scaling](https://github.com/stanford-cs336/assignment3-scaling)｜[A4 Data](https://github.com/stanford-cs336/assignment4-data)｜
  [A5 Alignment](https://github.com/stanford-cs336/assignment5-alignment)
- [CS336 讲义仓库](https://github.com/stanford-cs336/lectures)

**教程**
- [AIInfraGuide 前置知识](https://caomaolufei.github.io/AIInfraGuide/prerequisites)｜
  [Transformer 快速入门篇](https://caomaolufei.github.io/AIInfraGuide/guides/模块一-前置知识/transformer/transformer架构快速入门/)｜
  [3.3 Self-Attention 深入](https://caomaolufei.github.io/AIInfraGuide/guides/模块一-前置知识/transformer/33-self-attention机制深入理解)｜
  [CUDA 编程与算子优化](https://caomaolufei.github.io/AIInfraGuide/cuda)｜
  [分布式训练](https://caomaolufei.github.io/AIInfraGuide/distributed)｜
  [推理优化](https://caomaolufei.github.io/AIInfraGuide/inference)

**外部参考**
- [A1 实现复盘（含消融与性能分析）](https://xloverflow.github.io/zh/2026/02/09/CS336/CS336-Assignment-1-Building-a-Transformer-Language-Model-from-Scratch/)
- [DeepSeek 目标 JD](https://app.mokahr.com/social-recruitment/high-flyer/140576#/job/aed2a32c-189f-4338-b901-9b45d2b7edfd)

---

## 附：待办清单（可直接勾选）

**第 0 周：概念补课**
- [ ] 快速入门篇 §3.2 / §3.3 / §5.3 / §6.3 / §7.3
- [ ] 3.3 深入篇 §6（√d_k 方差）、§7.2/§7.3（Online Softmax）
- [ ] 3.8 §6.3~§8（KV Cache / PagedAttention / Prefix Cache / 解耦）
- [ ] 3.7 §8（显存规划详解）
- [ ] 3.3 §9（MHA/MQA/GQA/MLA）

**第 1~3 周：CS336 A1**
- [ ] `uv` 环境 + clone 仓库 + 跑一遍失败测试
- [ ] BPE 分词器（含特殊 token、并行预分词）
- [ ] 模型层全部组件 + 快照测试全绿
- [ ] 训练栈（AdamW / cosine+warmup / 梯度裁剪 / checkpoint）
- [ ] TinyStories 训练跑通 + 生成样例
- [ ] 建两个仓库（`attn-peak` 公开 / `cs336-assignments` 私有），写性能表骨架

**第 4~5 周：CUDA 地基**
- [ ] 模块二 1.1~1.4；写 vector add 并用 `ncu` 看 DRAM 吞吐
- [ ] 3.4 §7/§9、3.6 §5（并行切分与 kernel 融合选题）

**第 6~8 周：第一个能写的 kernel**
- [ ] Reduce 优化（对比 naive / shared memory / warp shuffle）
- [ ] Softmax + Online Softmax
- [ ] **开始写博客和 README**

**第 9~12 周：FlashAttention（项目主体）**
- [ ] 朴素 PyTorch attention + 显存/延迟基线
- [ ] Triton FlashAttention2
- [ ] benchmark 脚本 + roofline 图 + Nsight 截图
- [ ] 性能表填满 + 瓶颈结论

**第 13~16 周：扩展与收尾**
- [ ] CUDA GEMM（tiling + double buffering）或 A2 分布式部分
- [ ] 消融实验汇总表
- [ ] README 定稿 + 简历 bullet 撰写 + 博客系列成篇
