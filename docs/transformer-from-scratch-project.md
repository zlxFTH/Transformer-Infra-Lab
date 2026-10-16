# Transformer From Scratch + High-Performance Kernels：DeepSeek 高性能算子岗位项目规划

> **基础目标**：不依赖 Hugging Face Transformers、Trainer 或现成 Transformer 模块，亲手实现 Decoder-only 语言模型从原始文本到训练、评估、生成和性能分析的完整流程。
>
> **目标 JD**：[DeepSeek 高性能算子/通信/编译器工程师](https://app.mokahr.com/social-recruitment/high-flyer/140576#/job/aed2a32c-189f-4338-b901-9b45d2b7edfd)。
>
> **投递版主线**：Transformer 全流程 → C++/CUDA GEMM → Fused Attention → Roofline/SOL 与 Nsight 分析 → 通信或编译器方向择一深化。
>
> **参考路线**：AIInfraGuide 第 3 章、Stanford CS336 Assignment 1/2，以及目标 JD 对算子、硬件和理论性能上限的要求。

---

## 0. 目标 JD 对齐结论

仅完成“手写 Transformer + 训练一个小模型”更适合证明模型理解，**不足以精准支撑这个 JD**。该岗位关注的是模型之下的高性能实现，因此项目必须继续向下做到 C++/CUDA Kernel、硬件性能分析和真实负载集成。

| JD 要求 | 对项目的要求 |
|---|---|
| CUDA、Ascend C、TileLang；GEMM、Attention 等算子 | 以 NVIDIA + C++/CUDA 为主线，完成 GEMM 和 Fused Attention；TileLang/Triton 作为 DSL 对照，Ascend 暂不强行展开 |
| 以硬件物理极限为唯一标准 | 不只和 PyTorch/cuBLAS 比快慢，必须计算 Roofline 理论上限、阅读 Nsight SOL，并分析剩余性能差距 |
| 通信算子与 NCCL/HCCL 调优 | 作为深化方向之一：用 NCCL/nccl-tests 测量集合并通信带宽、延迟和拓扑影响；RDMA/DeepEP 放到后续 |
| DSL/IR/CodeGen 与编译优化 | 作为另一深化方向：检查 TileLang/Triton IR、PTX/SASS，理解向量化、流水、调度和内存优化 |
| 紧贴模型结构和真实训练/推理负载 | 将优化后的 Attention 接回自己的 Transformer，在相同 Batch/Seq 下验证端到端收益，而非只跑孤立 Kernel |
| 熟悉 C++ 和 Git；接口稳定、可靠、可维护 | 增加 Modern C++、CMake、PyTorch Extension、单元测试、Sanitizer、CI 和清晰 API |
| 大型开源贡献或维护个人开源软件 | 仓库不仅要能跑，还要有问题模板、版本发布、Benchmark 原始数据和后续维护记录 |

**路线选择：主攻高性能算子，通信与编译器二选一作为加分项。** 不要在同一首版里同时堆 CUDA、Ascend、RDMA、MLIR、DeepEP 和分布式训练；一个可深入答辩的 Attention Kernel 比一串浅尝辄止的技术名词更有价值。

---

## 1. 项目定位

### 1.1 一句话介绍

**从零实现一个小型现代 LLM，再向下实现它的高性能关键算子，并用测试、硬件理论上限和实测数据解释每一个细节。**

这个项目最终不是只有一个 `model.py`，而应包含：

- 一套自己实现的 Tokenizer、模型、优化器、训练和推理代码；
- 一个真实训练出来的 TinyStories 小型语言模型；
- 一套能发现未来信息泄漏、数值溢出、RoPE 位置偏移等问题的测试；
- 一套 C++/CUDA GEMM 与 Fused Attention 实现及 Kernel 级测试；
- 一份包含训练曲线、生成样例、Roofline/SOL、显存与速度对比的实验报告；
- 一个别人 clone 后能够复现并继续维护的 GitHub 仓库。

### 1.2 为什么选择 Decoder-only

AIInfraGuide 同时介绍了原始 Encoder-Decoder、Encoder-only 和 Decoder-only。本项目选择 **Decoder-only** 作为主线，原因是：

- 它是 GPT、LLaMA、Mistral、Qwen 等现代 LLM 的主流架构；
- 它与 CS336 的语言建模路线一致；
- 它能自然覆盖 next-token prediction、自回归生成、KV Cache、Prefill/Decode 等 AI Infra 核心问题；
- 只需维护一种 Block 和一套 KV Cache，便于把项目做深，而不是同时做两套架构却都停留在表面。

原始 Encoder-Decoder 的 Cross-Attention 可作为后续扩展，不列入首版必做范围。

### 1.3 “From Scratch”的边界

基础阶段目标是掌握 **语言模型和 Transformer**，不是重新实现一个 PyTorch；JD 专项阶段则用手写 Kernel 理解硬件执行，不把现成高性能库包装成自己的成果。

| 层面 | 项目约定 |
|---|---|
| 可以使用 | Python、NumPy、PyTorch Tensor、`nn.Module`、`nn.Parameter`、Autograd、基本矩阵运算 |
| 必须自己实现 | BPE、Linear、Embedding、RMSNorm、SiLU/SwiGLU、RoPE、Attention、MHA、Transformer Block、Cross Entropy、AdamW、学习率调度、训练循环、采样、KV Cache |
| JD 算子阶段 | 使用 C++、CUDA、CMake 和 PyTorch Extension；可使用 TileLang/Triton 做 DSL 对照；核心 GEMM 与 Fused Attention 必须有自己的实现 |
| 核心实现中禁止 | 直接以 `nn.Linear`、`nn.Embedding`、`nn.Transformer*`、`nn.MultiheadAttention`、`F.scaled_dot_product_attention` 作为交付实现；HF `transformers` / `tokenizers` / `Trainer`；现成 AdamW；现成生成器 |
| 可以在测试中作为参考实现 | PyTorch 官方算子、NumPy 手算结果、小规模有限差分、cuBLAS/CUTLASS |
| 可以在优化实验中作为对照 | `F.scaled_dot_product_attention`、官方 AdamW、`torch.compile`，但不能替换掉自己的基线实现 |

不要求基础版手写完整自动微分框架。需要能手推关键梯度，并用 Autograd、`torch.autograd.gradcheck` 或有限差分验证实现。

### 1.4 基础版暂不展开的内容

以下主题与目标 JD 的核心要求距离较远，先放到扩展阶段：

- DDP、FSDP、ZeRO、Tensor/Pipeline Parallel；
- 大规模 Common Crawl 清洗；
- Scaling Law 全套实验；
- SFT、DPO、GRPO 等后训练；
- PagedAttention、Continuous Batching、完整推理服务框架；
- Ascend C、RDMA/IBGDA、DeepEP、MLIR 等多条额外技术线。

**CUDA/TileLang Attention 不再只是可选扩展：它是 JD 投递版的核心交付。** 基础版先做到“小而完整、正确、可复现”，随后立即进入高性能算子阶段。

---

## 2. 最终交付物

项目结束时，仓库至少应有以下成果：

- [ ] 可训练的 Byte-level BPE Tokenizer，以及保存好的 vocab、merges 和 special tokens；
- [ ] 现代 Decoder-only Transformer：Pre-Norm、RMSNorm、RoPE、SwiGLU、Causal MHA；
- [ ] 手写 AdamW、Warmup + Cosine LR、Gradient Clipping 和完整训练循环；
- [ ] TinyStories 上训练出的一个小型模型 checkpoint；
- [ ] Greedy、Temperature、Top-k、Top-p 和 EOS 停止的生成程序；
- [ ] 朴素生成与 KV Cache 生成两种实现；
- [ ] 参数量、FLOPs、训练状态和 KV Cache 显存估算工具；
- [ ] C++/CUDA GEMM 的多版本实现，以及一个 Fused Attention Forward Kernel；
- [ ] Kernel 级正确性测试、边界测试和 Compute Sanitizer 检查；
- [ ] Roofline、Nsight Systems/Compute、SOL、IR/PTX/SASS 分析报告；
- [ ] 优化后的 Attention 接回 Transformer 后的端到端对照实验；
- [ ] 通信或编译器方向至少一个可展示的小型深化实验；
- [ ] 单元测试、集成测试、CPU CI 和可复现的 GPU Benchmark 脚本；
- [ ] 训练、推理、Kernel 与消融实验报告，以及原始结果数据；
- [ ] 完整 README、复现命令、结果表格、限制说明和简历描述。

---

## 3. 总体技术路线

### 3.1 端到端数据流

```text
Raw Text
   │
   ├─ 文档级 Train / Validation 划分
   ├─ 数据清洗、统计与去重（基础版）
   ▼
Byte-level BPE Training
   │
   ├─ vocab.json / merges.json / special_tokens.json
   ▼
Encode Corpus → Token IDs → memmap 二进制数据
   │
   ├─ 随机截取连续片段
   ├─ x = tokens[i : i + T]
   └─ y = tokens[i + 1 : i + T + 1]
   ▼
Token Embedding
   ▼
N × Decoder Block
   │
   ├─ RMSNorm → Causal MHA + RoPE → Residual Add
   └─ RMSNorm → SwiGLU FFN → Residual Add
   ▼
Final RMSNorm → LM Head → Logits
   │
   ├─ 训练：Cross Entropy → Backward → AdamW → Checkpoint
   ├─ 评估：Validation Loss / Perplexity
   └─ 推理：Prefill → Sampling → Decode Loop → KV Cache
   ▼
Profiling / Benchmark / Ablation / Report
```

### 3.2 首版模型配置

以 LLaMA 风格的现代 Decoder-only 为基线：

- Pre-Norm；
- RMSNorm；
- RoPE，只作用于 Q、K；
- SwiGLU FFN，无 bias；
- Multi-Head Causal Self-Attention；
- 每个 Block 两条残差连接；
- Final RMSNorm + LM Head；
- Weight Tying 做成可配置项，并单独做实验。

建议保留两档配置：

| 配置 | 用途 | Layers | d_model | Heads | d_ff | Context | BPE Vocab | 大致参数量 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `tiny` | CPU 调试、单元测试、快速过拟合 | 2 | 128 | 4 | 344 | 128 | 4K | 约 1.4M |
| `small` | 正式训练与最终展示 | 6 | 384 | 6 | 1024 | 512 | 10K | 约 18M |

参数量按 Embedding 和 LM Head 不共享权重估算。实际结果以配置和代码统计为准，不为了凑模型规模而修改实验数字。

硬件策略：

- 模型基础组件的正确性测试必须能在 CPU 上运行；
- 正式训练建议使用一张 NVIDIA GPU，先用小数据和小模型跑通，再增加训练 token 数；
- **JD 专项阶段需要 Linux + NVIDIA GPU + CUDA Toolkit + Nsight 工具**；没有 GPU 时可以学习原理和写 CPU 参考实现，但不能声称完成了 Kernel 性能项目；
- 每份报告记录 GPU 型号、SM 数量、显存、时钟、功耗上限、CUDA/Driver 版本和测试输入；
- Ascend NPU 只在 NVIDIA 主线完成后作为硬件迁移扩展，不在首版同时维护两套 Kernel。

---

## 4. 分阶段实施计划

## Milestone 0：仓库、规格与实验规范

### 要做的事

- [ ] 初始化 Python Package、依赖锁文件、格式化、静态检查和 pytest；
- [ ] 建立 `configs/`，所有模型和训练超参数由配置驱动，不散落魔法数字；
- [ ] 写清张量约定：默认使用 `(batch, seq, d_model)`，Attention 内部使用 `(batch, heads, seq, head_dim)`；
- [ ] 写清随机种子、数据划分、checkpoint、日志和实验命名规则；
- [ ] 配置 GitHub Actions，至少在 CPU 上执行单元测试；
- [ ] 在 README 中维护项目进度，未完成的模块不写成已完成。

### 验收标准

- 干净环境下能安装项目并运行 `pytest`；
- 同一个配置和随机种子可以定位到同一次实验；
- 原始数据、生成数据、checkpoint 和日志不混入 Git 历史。

---

## Milestone 1：数据集与 Byte-level BPE Tokenizer

对应 CS336 Assignment 1 的 Tokenizer 部分，也是最容易被“调库”跳过的环节。

### 1. 数据处理

- [ ] 首选 TinyStories，数据量适中且生成结果容易人工检查；也可增加 OpenWebText 小样本作为扩展；
- [ ] 记录数据来源、许可证、下载日期、文件校验值与数据统计；
- [ ] **按文档划分训练集和验证集，再训练 Tokenizer**，避免同一文档的片段同时出现在两侧；
- [ ] 明确文档边界和 EOS 的插入方式；
- [ ] 统计文档数、字符数、字节数、Token 数、平均/分位文档长度；
- [ ] 做基础清洗与完全重复文档去除，保留清洗前后统计；
- [ ] 验证集不参与 BPE 训练、词频统计或数据过滤阈值拟合。

### 2. BPE 训练

- [ ] 从 256 个 Byte Token 开始，而不是只支持 ASCII 或预先按汉字/单词切分；
- [ ] 统计相邻 Token Pair 频率，迭代合并最高频 Pair；
- [ ] 保存合并顺序，因为编码时需要按 rank 重放；
- [ ] 处理并列频率，保证训练结果确定性；
- [ ] 支持特殊 Token，并保证特殊 Token 不会被普通 merge 拆开或跨越；
- [ ] 支持流式读取较大文件，避免一次性把整个语料载入内存。

### 3. BPE 编码与解码

- [ ] 实现 `encode(text) -> list[int]`；
- [ ] 实现 `decode(ids) -> str`；
- [ ] 实现语料级编码并保存为 `np.memmap` 可读取的二进制格式；
- [ ] 根据 vocab 大小选择 `uint16` 或 `uint32`，并记录 dtype；
- [ ] 提供 Tokenizer 的保存、加载和版本信息。

### 4. 必做测试

- [ ] 空字符串、空格、换行、连续换行、emoji、中文、组合 Unicode 字符；
- [ ] 任意 UTF-8 文本的 `decode(encode(text)) == text`；
- [ ] 特殊 Token 出现在文本首尾、中间和相邻出现时结果正确；
- [ ] 构造一个小语料，手算前几轮 merge，与实现结果逐项比较；
- [ ] 同一输入多次训练得到相同 vocab 和 merges；
- [ ] Tokenizer 保存再加载后，编码结果完全一致；
- [ ] 对随机文本做 round-trip 属性测试。

### 验收标准

Tokenizer 独立通过测试；能将完整语料编码为可复用的 Token ID 数据，并输出数据报告和压缩率统计。

---

## Milestone 2：手写模型全部核心组件

每实现一个组件，就先写小测试，不要等整个模型写完后一起 Debug。

### 1. 基础组件

- [ ] `Linear`：支持任意前导维度，明确权重形状和初始化；
- [ ] `Embedding`：查表实现，不构造昂贵的 One-hot 矩阵；
- [ ] `Softmax`：使用减最大值的数值稳定实现；
- [ ] `RMSNorm`：明确 epsilon、计算 dtype 和可学习 scale；
- [ ] `SiLU` 与 `SwiGLU`：区分 gate、up、down 三个投影；
- [ ] 参数初始化：记录每个矩阵的初始化方式和理由。

### 2. RoPE

- [ ] 预计算不同位置与频率对应的 sin/cos；
- [ ] 对 Q、K 的二维子空间执行旋转，不旋转 V；
- [ ] 支持传入 `token_positions`，为后续 KV Cache 的位置偏移做准备；
- [ ] 不显式构造巨大的分块对角旋转矩阵；
- [ ] 用二维小例子手算旋转结果，验证方向和维度配对方式。

### 3. Attention

- [ ] 手写 Scaled Dot-Product Attention：`QK^T / sqrt(head_dim) → mask → softmax → AV`；
- [ ] 缩放因子使用 `head_dim`，不是 `d_model`；
- [ ] 实现 Causal Mask，并明确 mask 中 True/False 的含义；
- [ ] 在低精度路径中避免不安全的 `-inf` 运算，必要时用 FP32 计算 Softmax；
- [ ] 实现 MHA 的 Q/K/V 投影、拆头、合并和输出投影；
- [ ] 首版显式物化 Attention Matrix，便于理解；高性能版本放到优化阶段。

### 4. Transformer Block 与完整模型

- [ ] 实现 `x = x + Attention(RMSNorm(x))`；
- [ ] 实现 `x = x + SwiGLU(RMSNorm(x))`；
- [ ] 堆叠多个 Block；
- [ ] 添加 Final RMSNorm 和 LM Head；
- [ ] 支持可选 Weight Tying；
- [ ] 输出所有位置的 logits，而不是只输出最后一个位置；
- [ ] 打印每个模块的输入输出形状和参数量，形成一张完整数据流表。

### 5. 必做测试

- [ ] 每个模块与 PyTorch 参考实现加载相同权重后比较输出；
- [ ] 使用小矩阵手算 Softmax、RMSNorm、RoPE 和单头 Attention；
- [ ] 检查所有模块输入输出形状和 `d_model % num_heads == 0`；
- [ ] 做梯度检查，确认梯度不是 NaN，关键模块可通过 `gradcheck` 或有限差分抽查；
- [ ] **未来信息泄漏测试**：只修改位置 `t` 之后的 Token，位置 `t` 及其之前的 logits 必须保持不变；
- [ ] 短输入、长度为 1 的输入、不同 batch/seq 组合均能通过；
- [ ] 完整模型做一次 forward + backward，确认所有应训练参数都有梯度。

### 验收标准

完整模型与参考实现误差在事先声明的浮点容差内；因果性测试通过；能从 Token ID 一直跟踪到 `(B, T, V)` logits 的每一步形状。

---

## Milestone 3：训练目标、优化器与训练循环

### 1. Next-token 数据构造

- [ ] 从 Token ID memmap 随机采样连续片段；
- [ ] 构造 `x = tokens[i:i+T]` 与 `y = tokens[i+1:i+T+1]`；
- [ ] 检查最大合法起点，避免越界；
- [ ] 明确 batch、context length、seen tokens 和有效 batch size 的关系；
- [ ] 可选：梯度累积，并写清有效 batch size 的计算方式。

### 2. 损失与优化

- [ ] 手写数值稳定的 Cross Entropy，直接使用 logits 和 Log-Sum-Exp，不要先显式 Softmax 再取 log；
- [ ] 手写 AdamW：一阶/二阶动量、bias correction、epsilon 和解耦 Weight Decay；
- [ ] 对比官方 AdamW 的小规模更新轨迹；
- [ ] 实现 Linear Warmup + Cosine Decay + 最低学习率；
- [ ] 实现全局梯度范数裁剪，并记录裁剪前范数；
- [ ] FP32 基线稳定后，再增加 BF16/AMP 支持，不把混合精度和基础错误混在一起调试。

### 3. 训练循环

- [ ] 不使用 Trainer，显式写出 forward、loss、backward、clip、optimizer step、zero grad；
- [ ] 定期在验证集上计算 loss，验证时切到 eval 模式并禁用梯度；
- [ ] 记录 train/validation loss、perplexity、learning rate、grad norm、seen tokens、tokens/s、step time 和峰值显存；
- [ ] 保存模型、优化器、scheduler、iteration、随机数状态和完整配置；
- [ ] 实现断点恢复，恢复后不重复或跳过训练 step；
- [ ] 保留最佳验证集 checkpoint 和最后一个 checkpoint。

### 4. 调试顺序

1. 单个小 batch 上只跑 forward/backward；
2. 在极小数据上过拟合，训练 loss 应持续接近 0；
3. 用小模型跑完整 train/validation；
4. 通过后再扩大模型和训练 token 数；
5. 最后开启混合精度或其他性能优化。

### 验收标准

- 能过拟合一个小 batch；
- 正式训练时 loss 正常下降，无 NaN；
- Checkpoint 恢复后的下一次更新与未中断运行一致或在合理浮点误差内；
- 任一训练曲线都能对应到配置、代码版本和 checkpoint。

---

## Milestone 4：完成一次正式训练与语言模型评估

### 要做的事

- [ ] 使用 `small` 配置在 TinyStories 上完成一次正式训练；
- [ ] 用 seen tokens 记录训练量，不只写“训练了几个 epoch”；
- [ ] 根据固定验证窗口计算 validation cross entropy 和 `exp(loss)` perplexity；
- [ ] 实现一个简单的训练集 Unigram 基线，证明模型优于朴素统计基线；
- [ ] 绘制训练/验证 loss、学习率和梯度范数曲线；
- [ ] 保存固定 Prompt、固定随机种子下的生成样例；
- [ ] 记录训练所用硬件、耗时、平均 tokens/s 和峰值显存；
- [ ] 分析过拟合、欠拟合、重复生成和词汇多样性，不只展示最好的样例。

### 验收标准

- 验证集 loss 明显低于 Unigram 基线；
- 训练过程、验证指标和生成结果均可复现；
- README 中有完整的实验配置和真实数字；
- 不把 TinyStories 小模型的表现描述成通用 LLM 能力。

---

## Milestone 5：自回归推理与 KV Cache

### 1. 朴素生成

- [ ] 实现 Prompt Tokenization、模型前向、最后一个位置 logits 读取、采样、Token 追加和 Detokenization；
- [ ] 实现 Greedy、Temperature、Top-k、Top-p；
- [ ] 实现 EOS、最大生成长度和停止字符串；
- [ ] 处理 Temperature 为 0、Top-k 越界、Top-p 边界和空 Prompt 等参数情况；
- [ ] 采样前正确重新归一化候选概率。

### 2. KV Cache

- [ ] 将生成拆成 Prefill 和 Decode；
- [ ] Prefill 处理整个 Prompt，并为每层保存 K、V；
- [ ] Decode 每次只输入一个新 Token；
- [ ] 每层缓存形状为 `(B, H_kv, current_seq, head_dim)`；
- [ ] 使用预分配缓存和位置索引，避免每步 `torch.cat` 带来的重复分配；
- [ ] RoPE 使用全局位置 `past_length + current_position`，不能每次从 0 开始；
- [ ] 新 Token 的 Q 与全部历史 K/V 做 Attention，并追加新的 K/V；
- [ ] 支持达到 context 上限时明确报错或执行定义好的截断策略。

### 3. 必做测试与实验

- [ ] 对同一个 checkpoint 和 Prompt，逐步比较有无 KV Cache 的 logits；
- [ ] Greedy 模式下，两种实现生成的 Token 序列必须一致；
- [ ] 检查 Prefill 后缓存长度和每一步 Decode 后的缓存长度；
- [ ] 检查不同 Prompt 长度下 RoPE 位置偏移；
- [ ] 测量 TTFT、TPOT、端到端 tokens/s 和峰值显存；
- [ ] 对比不同 Prompt 长度、生成长度和 batch size；
- [ ] 用公式计算 KV Cache 理论大小，与实际分配做对照。

### 验收标准

KV Cache 不仅“能跑”，还要通过等价性测试，并给出真实性能表格。速度提升写实测值，不预先承诺固定倍数。

---

## Milestone 6：AI Infra 资源核算、Profiling 与消融实验

这是项目区别于普通“手搓 GPT”的关键部分。

### 1. 资源核算工具

实现一个脚本，输入模型配置、batch size、序列长度和 dtype，输出：

- 每层和整模型参数量；
- Embedding、Attention、FFN、LM Head 的参数占比；
- 前向/训练 FLOPs 的近似估算及适用假设；
- 参数、梯度、AdamW 状态、激活值和 KV Cache 的显存估算；
- Attention Score Matrix 的理论大小；
- 给定 GPU 显存时，理论上的配置限制。

至少实现并验证以下公式：

```text
# MHA + SwiGLU、无 bias、未共享 Embedding/LM Head 时
Params ≈ 2·V·d + L·(4·d² + 3·d·d_ff + 2·d) + d

# Weight Tying 时减少一份 V·d

# KV Cache bytes
2 × L × B × N × H_kv × d_head × bytes_per_element

# Attention Scores 元素数
B × H × N × N

# 稠密模型训练的常用粗略估算（需注明近似条件）
Training FLOPs ≈ 6 × 参数量 × 训练 Token 数
```

### 2. Profiling 与 Benchmark

- [ ] 使用 `torch.profiler` 找出训练和推理的主要耗时算子；
- [ ] GPU Benchmark 先 warm-up，并在计时时正确同步；
- [ ] 报告多次运行的中位数和波动，不只挑最快一次；
- [ ] 扫描序列长度，观察朴素 Attention 的 `O(N²)` 计算与显存趋势；
- [ ] 扫描 batch size，观察吞吐和显存变化；
- [ ] 对比自己实现的 Attention 与 PyTorch SDPA，仅作为优化对照；
- [ ] 对比 `torch.compile` 前后性能，记录编译开销和适用条件；
- [ ] 区分数据搬运、Kernel、CPU 调度等瓶颈，有证据再判断 compute-bound 或 memory-bound。

### 3. 至少完成两组消融

一组模型/训练消融，一组系统消融：

- **模型组任选其一**：RoPE vs 可学习绝对位置编码；SwiGLU vs GELU FFN；RMSNorm vs LayerNorm；Weight Tying on/off；
- **系统组必做**：无 KV Cache vs 有 KV Cache；可再加 naive Attention vs SDPA 或 FP32 vs BF16。

实验要求：

- 固定数据划分、训练 Token 数、主要超参数和随机种子；
- 每次只改变一个核心变量；
- 预先写清假设和评价指标；
- 报告结果是否符合预期，不符合时分析原因；
- 小模型实验结论只限定于当前设置，不做过度泛化。

---

## Milestone 7：JD 核心——C++/CUDA 高性能算子

这一阶段是投递该岗位的必要部分，不能再作为“以后有时间再做”的可选项。

### 1. C++ 工程与正确性基础设施

- [ ] 使用 Modern C++ 和 CMake 构建 PyTorch Extension，暴露简洁、稳定且有文档的 API；
- [ ] Kernel 内部计算自己实现，ATen/PyTorch 只用于 Tensor 管理、绑定和参考对照；
- [ ] 明确输入 dtype、device、contiguity、形状和边界约束，不支持的输入要显式报错；
- [ ] 为每个 Kernel 写 CPU/NumPy/PyTorch 参考实现，覆盖随机输入、极端数值和非方形/非整除形状；
- [ ] 使用 `compute-sanitizer` 的 memcheck、racecheck 等工具排查越界和竞争；
- [ ] 保留干净、可读的第一版 Kernel，后续每个优化版本单独提交，确保优化过程可审查。

### 2. 建立硬件与理论上限模型

- [ ] 记录目标 GPU 的 SM 数、CUDA Core/Tensor Core、时钟、HBM 容量与带宽、L2、Shared Memory 和寄存器限制；
- [ ] 复习 Warp 调度、合并访存、Shared Memory Bank Conflict、Occupancy、Register Pressure 和 Kernel Launch 开销；
- [ ] 为每个 Kernel 手算 FLOPs、实际读写 Bytes 和 Arithmetic Intensity；
- [ ] 使用 `min(Peak FLOPs, AI × Memory Bandwidth)` 建立 Roofline 上限，并说明使用的是什么精度和哪一级带宽；
- [ ] 使用 Nsight Systems 看端到端时间线，使用 Nsight Compute 看 Kernel 级 SOL、吞吐、Occupancy、Stall 和内存指标；
- [ ] 每次优化都回答“消除了什么瓶颈、距离理论上限还差多少、下一步为什么值得或不值得做”。

### 3. GEMM：从正确到接近硬件上限

至少保留以下版本，不在最终代码里只留一个无法解释来源的“最优版”：

1. Naive CUDA GEMM：建立线程索引和正确性基线；
2. Shared Memory Tiling：减少全局内存访问并提高数据复用；
3. Coalesced + Vectorized Loads：改善访存效率；
4. Register Tiling、Double Buffering 或 Tensor Core 路径中的至少一种：分析计算与流水瓶颈。

实验要求：

- [ ] 与 CPU/cuBLAS 对照正确性，覆盖矩形矩阵、非整块尺寸和不同 dtype；
- [ ] 选择真实 Transformer 中的 QKV、FFN、LM Head 矩阵形状，而不只测方阵；
- [ ] 报告 TFLOP/s、有效带宽、SOL/Roofline 达成率、Occupancy、寄存器使用和 Bank Conflict；
- [ ] cuBLAS/CUTLASS 可作为参考，但主要结论是距离硬件上限还有多少，而不是只写“达到 cuBLAS 的 X%”；
- [ ] 分析小矩阵时的 Launch Overhead，避免把 host 侧开销误判成 Kernel 瓶颈。

### 4. Fused Attention：从朴素实现到 Online Softmax + Tiling

- [ ] 先写正确但朴素的 CUDA/DSL Attention，验证 QK、Scale、Mask、Softmax、AV 每一步；
- [ ] 实现 FlashAttention 风格的 Forward Kernel：Q/K/V 分块、Online Softmax、Causal Mask 和输出累加；
- [ ] 不把完整 `N × N` Attention Matrix 物化到 HBM，并用显存数据证明这一点；
- [ ] 首版可以固定 `head_dim`、dtype 和 layout，先把一种配置做深，再逐步增加 64/128 等配置；
- [ ] 先覆盖 Training/Prefill Forward；Backward、GQA、Paged KV 和 Decode 特化作为后续独立任务；
- [ ] 与自己的 naive Attention 和 PyTorch SDPA 做正确性对照，覆盖 causal、短序列、长序列和数值极端输入；
- [ ] 使用 CUDA C++ 或 TileLang 完成主实现；再用 TileLang/Triton 复刻一个小算子或其中一个版本，比较开发效率和生成代码；
- [ ] 检查 DSL IR、PTX/SASS 中与向量化、共享内存、流水和指令选择相关的证据，不只看 Python 层源码。

### 5. 接回真实 Transformer 负载

- [ ] 通过 Extension 将优化后的 Attention 接入前面的 Decoder Block；
- [ ] 保持模型权重、输入 Token、Batch 和 Seq 完全一致，比较替换前后的 logits 与训练/推理结果；
- [ ] 分别报告孤立 Kernel 收益和端到端收益，解释 Amdahl 定律以及其他算子为何限制整体提升；
- [ ] 用 Nsight Systems 检查是否出现额外 H2D/D2H、同步、Tensor 拷贝或 Launch 开销。

### 验收标准

- 所有 Kernel 通过正确性和边界测试，Compute Sanitizer 无未解释错误；
- 每个优化版本都有对应 Benchmark、Profiler 原始输出和设计说明；
- 至少完成一份 GEMM Roofline/SOL 报告和一份 Fused Attention Roofline/SOL 报告；
- 优化后的 Attention 在自己的 Transformer 中完成端到端验证；
- 所有性能数字注明 GPU、软件版本、输入形状、dtype、计时方法和统计口径。

---

## Milestone 8：通信或编译器方向择一深化

目标是展示方向感和深入能力，不要求在第一份项目里覆盖 JD 的所有加分项。

### 方向 A：集合通信

- [ ] 使用 `nvidia-smi topo -m` 和硬件文档说明 PCIe/NVLink 拓扑；
- [ ] 用 NCCL/nccl-tests 测量 AllReduce、AllGather、ReduceScatter 的延迟和带宽，扫描消息大小与 GPU 数量；
- [ ] 对照 Ring 等算法的通信量与理论链路带宽，分析 Bus Bandwidth、小消息延迟和大消息带宽之间的差异；
- [ ] 使用 NCCL 日志和 Nsight Systems 定位通信等待、拓扑或同步问题；
- [ ] 可选：用 C++/NCCL 写一个最小自定义集合通信调用或接入自己的训练负载。

### 方向 B：DSL 与编译优化

- [ ] 选择 TileLang 或 Triton，完成一个 Memory-bound 和一个 Compute-bound 小算子；
- [ ] 对比不同 Block/Tile、Num Warps/Stages、Vectorization 和 Layout 配置；
- [ ] 阅读对应 IR、PTX/SASS，指出某项源码配置如何改变生成代码；
- [ ] 用实验区分编译时间、自动调优时间和最终 Kernel 执行时间；
- [ ] 可选：实现一个最小 IR 变换或编译 Pass，并用测试证明语义不变。

### 验收标准

只选一个主方向形成完整报告：问题定义、理论模型、实现/配置、实验数据、瓶颈分析和下一步计划。没有多卡或 RDMA 环境时选择编译器方向，不伪造通信实验。

---

## Milestone 9：工程质量、文档与发布

- [ ] 提供数据准备、训练、评估、生成和 Benchmark 的一条命令入口；
- [ ] 提供最小 Quickstart，让使用者 5–10 分钟内跑通 tiny 示例；
- [ ] 提供 C++/CUDA 构建说明、GPU 环境要求和一键 Kernel Benchmark；
- [ ] CI 覆盖核心单元测试和基本 CLI smoke test，GPU 测试提供可复现脚本；
- [ ] README 包含架构图、完整数据流、项目结构、结果、复现步骤和限制；
- [ ] 大文件通过 GitHub Release 或模型托管平台发布，不提交进 Git 历史；
- [ ] 固定一个公开 Release，记录代码版本、配置、数据版本、GPU 环境和 checkpoint；
- [ ] 为关键技术决策写简短 ADR 或设计说明，例如为什么选 RoPE、为什么某种 Tile 配置更接近理论上限；
- [ ] 保存 Benchmark 原始输出、Profiler 文件和图表生成脚本，而不只上传截图；
- [ ] 增加 Issue/PR 模板和后续维护计划，把调试过程中最有价值的错误写成 troubleshooting。

---

## 5. 推荐仓库结构

```text
transformer-from-scratch/
├── README.md
├── pyproject.toml
├── LICENSE
├── configs/
│   ├── tiny.yaml
│   └── small.yaml
├── src/transformer_lab/
│   ├── tokenizer/
│   │   ├── bpe.py
│   │   ├── train.py
│   │   └── io.py
│   ├── data/
│   │   ├── prepare.py
│   │   ├── dataset.py
│   │   └── stats.py
│   ├── model/
│   │   ├── layers.py
│   │   ├── rope.py
│   │   ├── attention.py
│   │   ├── block.py
│   │   └── transformer.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── optimizer.py
│   │   ├── schedules.py
│   │   ├── checkpoint.py
│   │   └── trainer.py
│   ├── inference/
│   │   ├── sampling.py
│   │   ├── kv_cache.py
│   │   └── generate.py
│   ├── evaluation/
│   │   └── language_modeling.py
│   └── systems/
│       ├── accounting.py
│       ├── profiler.py
│       └── benchmark.py
├── csrc/
│   ├── CMakeLists.txt
│   ├── bindings.cpp
│   ├── gemm_naive.cu
│   ├── gemm_tiled.cu
│   └── attention_fused.cu
├── dsl/
│   ├── README.md
│   └── attention_tilelang.py
├── tests/
│   ├── test_tokenizer.py
│   ├── test_layers.py
│   ├── test_attention.py
│   ├── test_causality.py
│   ├── test_optimizer.py
│   ├── test_training.py
│   ├── test_inference.py
│   └── cuda/
│       ├── test_gemm.py
│       └── test_attention_fused.py
├── scripts/
│   ├── prepare_data.py
│   ├── train_tokenizer.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── benchmark.py
│   └── benchmark_kernels.py
├── reports/
│   ├── architecture.md
│   ├── training.md
│   ├── inference.md
│   ├── gemm.md
│   ├── attention_kernel.md
│   ├── communication_or_compiler.md
│   └── ablations.md
└── .github/workflows/tests.yml
```

---

## 6. 关键测试清单

| 测试 | 能发现的典型问题 |
|---|---|
| 小型语料手算 BPE merge | Pair 统计、rank、并列处理和特殊 Token 错误 |
| BPE 随机文本 round-trip | UTF-8、bytes 与字符串转换错误 |
| 相同权重下对比参考模块 | Linear、RMSNorm、RoPE、Attention 等公式或权重布局错误 |
| 未来 Token 扰动测试 | Causal Mask 失效、训练时偷看未来 |
| 小 batch 过拟合 | 标签错位、Loss、反向传播或优化器错误 |
| 手写 AdamW vs 官方 AdamW | 动量、bias correction、Weight Decay 更新顺序错误 |
| Checkpoint 恢复对照 | 优化器、scheduler、step 或 RNG 状态缺失 |
| 有无 KV Cache 逐 Token 对比 | RoPE 位置偏移、Mask、缓存拼接和 logits 读取错误 |
| 理论参数量 vs 代码统计 | 漏参数、重复计数、Weight Tying 理解错误 |
| 理论 KV Cache vs 实际分配 | Cache shape、层数、头数或 dtype 错误 |
| GEMM 随机/非整除形状对照 | 线程索引、Tile 边界、越界访存和 dtype 错误 |
| Fused vs Naive Attention 对照 | Online Softmax 重缩放、Causal Mask、累加和布局错误 |
| Compute Sanitizer | 越界、未初始化访问和线程竞争 |
| 优化 Kernel 接入模型后的 logits 对照 | Binding、Contiguity、额外拷贝和端到端集成错误 |

---

## 7. 十四周执行安排

按每周约 8–12 小时设计，正式 GPU 训练和自动调优时间另计。每个阶段结束都提交一个可检查的 PR。前七周建立模型底座，后七周形成 JD 所需的高性能算子证据。

| 周次 | 重点 | 周末必须留下的证据 |
|---|---|---|
| 第 1 周 | M0 + 数据处理 + BPE | Tokenizer 测试、vocab/merges、数据统计 |
| 第 2 周 | 基础层、RMSNorm、SwiGLU、RoPE | 单元测试、手算示例、形状表 |
| 第 3 周 | Attention、Block、完整模型 | 参考对照、因果性测试、forward/backward |
| 第 4 周 | Loss、AdamW、训练循环、Checkpoint | 小 batch 过拟合、恢复训练测试 |
| 第 5 周 | 正式训练与评估 | Loss 曲线、PPL、生成样例、训练配置 |
| 第 6 周 | 采样与 KV Cache | 等价性测试、TTFT/TPOT/吞吐表格 |
| 第 7 周 | 资源核算、PyTorch Profiling、消融 | 热点算子清单、资源模型、消融报告 |
| 第 8 周 | C++/CMake/CUDA、Nsight、Roofline | Extension 构建、首个 CUDA Kernel、GPU 理论上限表 |
| 第 9 周 | Naive/Tiled GEMM | 正确性测试、Sanitizer、两版性能数据 |
| 第 10 周 | GEMM 优化与硬件瓶颈分析 | SOL/Roofline、Occupancy、寄存器与访存分析 |
| 第 11 周 | Fused Attention Forward | Online Softmax、Tiling、正确性与显存对照 |
| 第 12 周 | Attention 优化、DSL/IR/PTX/SASS | Kernel 报告、生成代码分析、多版本 Benchmark |
| 第 13 周 | 接回 Transformer + 通信/编译器择一 | 端到端收益、Amdahl 分析、专项报告 |
| 第 14 周 | 补测试、复现、README、Release | 原始性能数据、公开 Release、JD 版简历描述 |

如果只有八周，压缩正式模型训练、数据规模和模型消融，把时间优先留给 M2/M3 的模型理解与 M7 的 Kernel 深度；不要砍掉正确性测试和性能分析。

---

## 8. 后续扩展路线

主线完成后，再按 CS336 Assignment 2–5 和 AIInfraGuide 后续章节扩展。

### P1：最适合优先做的扩展

- [ ] GQA/MQA：减少 KV 头数，实测 KV Cache 与 Decode 性能；
- [ ] BF16 混合精度：对比速度、显存和训练稳定性；
- [ ] Activation Checkpointing：用额外计算换激活显存；
- [ ] `torch.compile`：分析 Graph Break 和真实收益。

### P2：AI Infra 深入方向

- [ ] 实现或剖析 FlashAttention Backward，核对重计算、梯度与显存开销；
- [ ] 为 GEMM 增加 Tensor Core/MMA 路径，并与 CUTLASS/CuTe 的实现和生成代码对照；
- [ ] 为 GQA、Decode 或特定 `head_dim` 增加 Attention 特化 Kernel；
- [ ] 使用 CUDA Graph 减少小 Kernel 的 Launch Overhead，并分析适用边界；
- [ ] DDP：梯度 AllReduce、bucket 与通信/计算重叠；
- [ ] Sharded Optimizer、FSDP：比较模型状态显存和通信代价；
- [ ] 在具备硬件条件后扩展到 RDMA、NVSHMEM、DeepEP 或 Ascend C。

### P3：更大范围的语言模型项目

- [ ] 多组模型规模和数据量实验，拟合 Scaling Law；
- [ ] 小规模 SFT，对比预训练模型与指令微调模型；
- [ ] 更完整的推理服务：Paged KV Cache、Continuous Batching、OpenAI 兼容 API。

这些扩展应单独开 Milestone，不要为了让简历显得“技术多”而在主仓库堆半成品。

---

## 9. 实验结果与简历模板

### 9.1 README 结果表格模板

**训练结果**

| 配置 | 参数量 | 训练 Tokens | 验证 Loss | 验证 PPL | 训练硬件 | 训练时间 | Tokens/s | 峰值显存 |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| small | 待实测 | 待实测 | 待实测 | 待实测 | 待填写 | 待实测 | 待实测 | 待实测 |

**推理结果**

| Prompt 长度 | 生成长度 | 实现 | TTFT | TPOT | Tokens/s | 峰值显存 |
|---:|---:|---|---:|---:|---:|---:|
| 待测 | 待测 | Naive | 待测 | 待测 | 待测 | 待测 |
| 待测 | 待测 | KV Cache | 待测 | 待测 | 待测 | 待测 |

**消融结果**

| 实验 | 唯一变量 | 固定条件 | 核心指标 | 结果 | 结论 |
|---|---|---|---|---|---|
| A1 | 待填写 | 数据、Token 数、Seed 等 | 待填写 | 待实测 | 待分析 |

**Kernel 性能结果**

| Kernel | GPU / Shape / dtype | 版本 | 实测性能 | 理论上限 | 上限达成率 | 主要瓶颈 |
|---|---|---|---:|---:|---:|---|
| GEMM | 待填写 | Naive → Tiled → Optimized | 待实测 | 待计算 | 待计算 | 待分析 |
| Attention | 待填写 | Naive → Fused | 待实测 | 待计算 | 待计算 | 待分析 |

### 9.2 JD 定向简历描述模板

> **Transformer From Scratch + High-Performance Attention**  
> 独立实现 Byte-level BPE、Decoder-only Transformer 和预训练/推理流程，并基于 C++/CUDA 实现多版本 GEMM 与 Fused Causal Attention（Online Softmax + Tiling）；通过参考实现、随机/边界测试和 Compute Sanitizer 验证正确性。在 `[GPU、Shape、dtype]` 下达到 `X TFLOP/s` 或 `X GB/s`，对应 Roofline/SOL 上限达成率 `X%`；使用 Nsight Systems/Compute 分析 Occupancy、寄存器、Bank Conflict 和 Stall，并将优化 Kernel 接回 Transformer，实现端到端 `X%` 性能提升。另完成 `[TileLang/Triton IR 或 NCCL 通信]` 专项实验，代码包含 CMake 构建、CI、可复现 Benchmark 和原始 Profiling 数据。

注意：

- `X` 必须替换成真实结果，并保留对应实验链接；
- Kernel 数字必须写清 GPU、Shape、dtype、Batch/Seq、计时方法和理论上限口径；
- 同时报告孤立 Kernel 与端到端收益，不用前者冒充后者；
- 没做的 Ascend、RDMA、分布式或编译器开发不要写进已完成描述；
- “实测达到理论上限的 X%”比“极致优化”更可信，但必须能解释上限如何计算。

---

## 10. 面试前必须能脱稿回答的问题

1. 为什么 Attention 要除以 `sqrt(head_dim)`，而不是 `sqrt(d_model)`？
2. Q、K、V 分别是什么？为什么 RoPE 只作用于 Q、K？
3. 为什么 Attention 需要 Mask？如何证明你的模型没有未来信息泄漏？
4. 为什么 Softmax 要减最大值？Cross Entropy 为什么用 Log-Sum-Exp？
5. Multi-Head 相比 Single-Head 增加了什么？拼接后为什么还需要输出投影？
6. Pre-Norm 和 Post-Norm 的残差路径有什么不同？
7. RMSNorm 和 LayerNorm 差在哪里？为什么 RMSNorm 更省？
8. SwiGLU 为什么有三个矩阵？为什么常见 `d_ff ≈ 8/3 · d_model`？
9. 残差连接为什么能改善深层网络的梯度传播？
10. 为什么训练可以并行计算整个序列，而生成必须逐 Token？
11. 输入 `x` 和标签 `y` 为什么错开一位？错位会造成什么现象？
12. AdamW 和 Adam + L2 是否完全等价？Bias Correction 在解决什么？
13. 你的模型参数量如何手算？FFN 和 Attention 各占多少？
14. 训练显存由哪些部分组成？AdamW 状态为什么很占显存？
15. KV Cache 存的是什么？它的形状和显存公式是什么？
16. 使用 KV Cache 后，为什么 RoPE 位置不能从 0 重新开始？
17. Prefill 和 Decode 的计算特征有什么不同？TTFT 和 TPOT 分别对应哪个阶段？
18. Top-k 与 Top-p 的区别是什么？Temperature 如何改变分布？
19. 为什么朴素 Attention 的 Score Matrix 会随序列长度平方增长？FlashAttention 在优化什么？
20. 你的每个性能结论是通过什么实验得出的？有哪些不能泛化的限制？
21. Arithmetic Intensity 如何计算？如何用 Roofline 判断一个 Kernel 更接近 Compute-bound 还是 Memory-bound？
22. 什么是合并访存和 Shared Memory Bank Conflict？你的 GEMM 如何减少它们？
23. Occupancy 由哪些资源共同限制？寄存器使用增加为什么可能反而降低性能？
24. GEMM Tiling 如何提高数据复用？Tile 大小改变会如何影响计算、显存带宽和 Occupancy？
25. Online Softmax 在新 Block 最大值变大时，为什么要重缩放之前的累加结果？
26. FlashAttention 为什么仍是精确 Attention？它减少的是哪一类 HBM 访问，而不是简单地减少 FLOPs？
27. 为什么不能只写“比 PyTorch 快 X%”？你的理论上限和 Nsight SOL 指标分别说明什么？
28. 为什么需要检查 IR/PTX/SASS？你观察到的哪条生成代码证据解释了一次性能变化？
29. Kernel 提升 `X%` 为什么端到端只提升 `Y%`？如何用 Amdahl 定律解释？
30. 如果选择通信方向：Ring AllReduce 的数据流和通信量是什么？拓扑、消息大小和延迟如何影响实测带宽？
31. 如果选择编译器方向：Tile/Block、Num Warps/Stages、Layout 和 Vectorization 如何影响生成代码与硬件执行？
32. 你的 Benchmark 如何处理 warm-up、CUDA 同步、时钟波动、编译/自动调优开销和重复测量？

---

## 11. 常见坑与排查顺序

- **Tokenizer 只能处理英文**：检查是否真正从 Byte 级别开始，而不是按字符或空格初始化词表。
- **特殊 Token 被拆开**：检查预分词和 merge 是否跨越特殊 Token 边界。
- **验证集异常好**：检查是否按文档划分、是否存在重复文档和 Tokenizer 数据泄漏。
- **Attention 输出形状随机出错**：逐项打印 reshape/transpose 后的形状，必要时先 `contiguous()`。
- **模型 loss 接近 0 但生成无意义**：优先检查 Causal Mask、标签错位和训练/推理数据路径。
- **低精度出现 NaN**：检查 Mask 填充、Softmax dtype、Loss 和梯度范数，不要先盲目调小学习率。
- **小 batch 都无法过拟合**：按 Tokenizer → Data → Forward → Loss → Backward → Optimizer 的顺序隔离问题。
- **KV Cache 结果与朴素生成不同**：首先检查 RoPE 的全局位置、缓存顺序和 Mask 长度。
- **Checkpoint 恢复后曲线跳变**：检查 optimizer、scheduler、step、数据顺序和 RNG 状态。
- **Benchmark 结果忽快忽慢**：检查 warm-up、CUDA 同步、后台任务、输入尺寸和是否计入编译/Tokenization。
- **理论显存与实测不一致**：先列出临时激活、Allocator 缓存、Fragmentation 和 Profiler 统计口径，不要直接认为公式失效。
- **实验很多但无法比较**：回到单变量原则，固定数据、训练量、配置和种子。

---

## 12. 最终 Definition of Done

### 基础版完成条件

- [ ] 从原始文本到生成文本的整条链路均由自己的代码串起来；
- [ ] 所有核心模块有独立测试，完整模型有因果性和端到端测试；
- [ ] 至少完成一次真实训练，并有验证集指标和失败/局限分析；
- [ ] 朴素推理与 KV Cache 推理结果一致，且有实测性能对比；
- [ ] 能手算参数量、Attention 复杂度和 KV Cache 显存，并与代码/实测对照；
- [ ] 至少完成一组模型消融和一组系统消融；
- [ ] 新环境按照 README 能跑通 tiny 示例和核心测试；
- [ ] 能不依赖代码，在白板上画出完整数据流并解释每个设计选择。

### JD 投递版额外完成条件

- [ ] C++/CMake Extension 能在文档所述环境中干净构建，API 和输入约束清晰；
- [ ] 多版本 GEMM 与 Fused Attention 均通过参考对照、边界测试和 Compute Sanitizer；
- [ ] 能手算 Kernel 的 FLOPs、Bytes、Arithmetic Intensity 和 Roofline 上限；
- [ ] 有完整 Nsight Systems/Compute 与 SOL 分析，能解释每次优化改变了哪个硬件瓶颈；
- [ ] 至少有一项 IR/PTX/SASS 证据支持对生成代码和性能变化的解释；
- [ ] 优化后的 Attention 已接回真实 Transformer，并分别报告 Kernel 与端到端收益；
- [ ] 通信或编译器方向至少完成一个能独立展示的深化实验；
- [ ] README 中每个数字都能追溯到配置、日志、Benchmark 原始输出或 Profiler 文件；
- [ ] 能清楚说明当前实现距离理论上限还差多少，以及下一步为什么不值得或值得继续优化。

---

## 13. 参考资料

- [目标 JD：DeepSeek 高性能算子/通信/编译器工程师](https://app.mokahr.com/social-recruitment/high-flyer/140576#/job/aed2a32c-189f-4338-b901-9b45d2b7edfd)
- [AIInfraGuide：AI Infra 学习路线](https://caomaolufei.github.io/AIInfraGuide/guides/ai-infra%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/)
- [AIInfraGuide：第 3 章 AI Infra 工程师学 Transformer](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%B8%80-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86/%E7%AC%AC3%E7%AB%A0-transformer%E6%9E%B6%E6%9E%84%E8%AF%A6%E8%A7%A3/)
- [CS 自学指南：深度生成模型学习路线](https://csdiy.wiki/%E6%B7%B1%E5%BA%A6%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B/roadmap/)
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [CS336 Assignment 1: Basics](https://github.com/stanford-cs336/assignment1-basics)
- [CS336 Assignment 2: Systems](https://github.com/stanford-cs336/assignment2-systems)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [TileLang](https://github.com/tile-ai/tilelang)
- [Triton](https://github.com/triton-lang/triton)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: RoPE](https://arxiv.org/abs/2104.09864)
- [RMSNorm](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
