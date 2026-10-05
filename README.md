<!-- dev-glm53f report (2026-10-05). English by default — switch to Chinese at the prominent link below. -->

<a id="dev-glm53f"></a>

# GLM-5.3-Flash at 1M context on one 48 GB card — the dev-glm53f line

**English (default)** · **[切换到中文版 →](#cn)**

This branch of the fork runs **GLM-5.3-Flash** — a 180 GB Q4_K_M mixture-of-experts model with 288 experts per layer and
hybrid linear (KDA) + sparse-indexed (DSA) attention — with a **1,048,576-token context**, the **live expert cache** and
the **vision adapter**, on a **single 48 GB GPU** backed by CPU RAM, with output correctness verified up to 250k-token
prompts. Prefill is batched at 65,536 tokens **as a throughput device for long inputs, not as a setting in itself**
(see "Why big prefill batches" below); decode runs its natural 1–16-token batches. At the fork point this configuration did not fit at any
context: selection-scoring memory grew with context x batch (extrapolated ~93 GB at 1M), and 53k+ token batches hit
launch-limit crashes.

## How fast

One machine: **RTX 4090D 48 GB on PCIe 3.0** · EPYC 7642, 8× DDR4-2133 (~100 GB/s) · GLM-5.3-Flash Q4_K_M
(179.7 GB, experts in system RAM) · 1M context reserved with q8_0 KV · 6 expert-cache slots · concurrency 1.
Full ladder, one session, 128 generated tokens per point:

| prompt (tokens) | prefill (ms) | prefill (t/s) | output (tokens) | output (ms) | output (t/s) |
|---:|---:|---:|---:|---:|---:|
| 64 | 1323.16 | 48.37 | 128 | 7960.47 | 16.08 |
| 256 | 4634.11 | 55.24 | 128 | 7966.66 | 16.07 |
| 1024 | 12483.13 | 82.03 | 128 | 8524.37 | 15.02 |
| 4096 | 15075.61 | 271.70 | 128 | 8745.72 | 14.64 |
| 16384 | 27554.56 | 594.60 | 128 | 8239.76 | 15.53 |
| 65536 | 76286.40 | 859.08 | 128 | 8566.24 | 14.94 |
| 262144 | 423405.21 | 619.13 | 128 | 7262.20 | 17.63 |

Read it as: prefill climbs with the batch size as the fixed per-pass costs amortize, peaking at the full 65,536-token
batch (859 t/s); 262k spans four batches and settles at 619 t/s as the per-batch selection state grows with the
context. Decode is **flat 14.6–17.6 t/s across the whole ladder — 262k included (17.6)** after the graph-reuse fix
below; before it, a fixed 10 t/s cliff started past ~2k context.

Same machine, supporting measurements: full 65k batch with the expert cache off — 643 t/s; the cache-scheduling
adoption (admission + lookahead) A/B — decode 14.9 → 15.5 t/s; peak VRAM 46.9 / 48 GB with 1M context, q8_0 KV, the
cache and the vision adapter all reserved; correctness — sanity arithmetic plus verifiable recall at
69k / 125k / 138k / 179k / 250k tokens, all coherent. At the fork point none of this ran: selection memory
extrapolated to ~93 GB at 1M and 53k+ batches crashed.

**Why big prefill batches.** With the experts in RAM, every pipeline pass pays the same fixed costs regardless of how
many tokens it carries: uploading the batch's activated expert weights over PCIe and crossing the CPU/GPU scheduling
boundary. At 64–256-token batches those round trips dominate the whole pass (48–55 t/s); at 4k they still weigh
(272 t/s); at 65k the identical uploads amortize over 16x more tokens and the PCIe stream stays continuously
overlapped with compute (859 t/s). Long inputs are exactly where this matters — hence the 65,536-token prefill
batch. Decode never pays this cost: it runs 1–16-token batches against the expert cache, and its per-token work is
independent of the batch size chosen for prefill.

## Which path runs when

The sparse-attention layers switch between three execution paths by shape, not by configuration. The crossover is
the selection window (2,051 rows = indexer top-k × pool size); the tile size is 512 tokens.

| existing context | new input | path | why |
|---|---|---|---|
| ≤ ~2k | decode (1–16 tokens) or small batch | monolithic scatter | the KV canvas is still smaller than the 2,051-row selection — scattering is the least work |
| > ~2k | decode (1–16 tokens) | gather, single chunk | per-token work pinned to the 2,051 selected rows; a canvas would grow with context |
| > ~2k | small batch ≤ 512 tokens (follow-up turns) | gather, chunked | same crossover; chunking bounds the gather's memory |
| any | batch > 512 tokens, single sequence | token-tiled scatter | at big batches the gather's latent copies are a bandwidth wall; tiles keep mask memory bounded |
| any | multi-sequence / 2d-rope (vision) | gather | the scatter tiles assume one sequence with 1d positions |

Selection scoring itself splits into (context-chunk × token-tile) blocks with a running top-k merge once
pools × tokens exceeds ~2^24 — that is what keeps peak memory flat at any context; smaller workloads keep the
monolithic scoring path.

## The optimizations

**A. Selection memory made context-bounded (this is what fits 1M).**
- *Chunked indexer scoring.* The sparse-indexer score is computed in (context-chunk x token-tile) blocks with a running
  top-k merge; a product gate keeps small batches on the monolithic path. Selection peak stops growing with
  context x batch: measured **flat 46.8 GB peak from 69k to 250k tokens** at full 1M reservation.
- *Per-token visibility instead of dense masks.* The [pools x tokens] causal-mask input (hundreds of GB of host data at
  1M) becomes a per-token visible-count vector; chunk masks are computed on the GPU when needed.
- *Fused causal mask blocks.* Each tile's mask block is written by a single GPU pass straight into the attention layout,
  replacing a ~10-op tensor chain (~6 KB → ~1 KB per token of mask traffic).

**B. Prefill throughput: token-tiled sparse scatter.**
- *Tiled scatter attention.* Large single-sequence batches run the DSA attention in token tiles — dump-mapped indices,
  per-tile sparse flash attention, per-tile output projection. This escapes the gather path's latent-copy bandwidth
  wall: the gather alternative measures ~125 t/s at these batch shapes on this card, the tiled scatter **643 t/s**
  with the cache off and **859 t/s** end-to-end at the full 65k batch (table above).
- *Whole-KV f16 conversion cache.* The quantized-KV → f16 conversion for flash attention is cached for the whole KV
  store instead of redone per batch, removing ~90 GiB of redundant conversion traffic per full batch.

**C. Scale stability and determinism (64k → 262k+).**
- *Batched kernel launches* wherever a token or KV dimension rides a CUDA grid axis that caps at 65,535 — 262k-token
  full-context batches run stable.
- *Tiled, two-pass batched expert GEMM* keeps per-call shared memory bounded — 53k+ token batches no longer trip
  shared-memory assertions or progressive flush misalignment.
- *Grid-overflow guard for batched MMVQ/MMQ* — an upstream-level bug any model hits on batched quantized paths at these sizes.
- *Deterministic top-k by default* (segmented argsort): the CCCL DeviceTopK fast path races on CCCL < 3.4.3
  (acknowledged upstream). Selection is now reproducible run-to-run.
- *Cross-backend selection parity.* CPU and GPU top-k/argsort orderings are made identical, every selection input is
  filled on every path, and scheduler rebinding is restored when a graph re-splits — the chunked and monolithic
  selection paths are value-equivalent at any scale (verified at the token level at 250k).
- *Graph reuse across decode tokens.* The sparse-attention paths replace the causal-mask input with a 1-element dummy;
  the graph-reuse check compared that dummy against the context size, failed on every token, and forced a full graph
  rebuild and CUDA-graph re-capture per token — a fixed ~40% decode tax at any context past ~2k tokens (15 → 10 t/s,
  flat to 262k). The check now recognizes the dummy, and decode-sized gathers take the single-chunk form again.
  Long-context decode: **10 → 14.6–17.6 t/s across 4k–262k context** (full ladder above) — no cliff at any scale.

**D. The MoE pipeline on one card.**
- *Expert-chain tiling by tokens* keeps decode-scale activations from blowing the compute buffer, so the expert cache
  coexists with full 65k batches inside 48 GB.
- *One full copy per graph for host-resident expert weights shared by several consumers* in tiled graphs — tiled
  execution always reads complete weights, with no bandwidth cost at these sizes.
- *Admission control + dynamic upload lookahead for the expert cache* (adopted from the fork's newer line): a missed
  expert must recur within a 64-token window before taking a slot, and predicted uploads are timed by measured link
  bandwidth vs layer time. Decode here: **14.9 → 15.5 t/s (+3.8%)**, cache hit 12% → 14% on the short-chat workload.
- *Short-prompt cliff removed:* batches up to 512 tokens now use the cached-expert path in place (previously only
  ≤31-token batches did — short prompts lost ~9 t/s above that line).

**E. Enablement and instrumentation.**
- *GLM-5.3-Flash GGUF compatibility* for the unsloth-family quantized releases (architecture naming, KV prefixes,
  vision adapter): the five-shard Q4_K_M + mmproj load and serve.
- *Opt-in diagnostics* (~1.5x slower only when enabled): per-node post-eval dumps with backend-safe reads,
  allocation-plan liveness dumps, per-request cache churn logging — what made scale issues bisectable in single runs.

## vs stock llama.cpp

glm5-next / GLM-5.3-Flash is not supported upstream at the fork point. Mixed CPU/GPU MoE serving with the live expert
cache, placement and self-tuning come from the neurall fork. Everything in sections A–E above is this line.

## vs the neurall fork (base)

The base targets multi-GPU 24 GB-class rigs at ≤64k contexts (its own headline: 1.7–2.4x decode vs upstream). This
line targets **one 48 GB card at 1M context, 65k-token prefill batching and vision**, adding the tiled-scatter
prefill path, context-bounded selection memory, 262k-scale launch fixes, deterministic top-k and cross-backend
selection parity, GLM-5.3-Flash GGUF compatibility — and it adopts the base's newer cache scheduling
(admission/lookahead) while keeping the older knob names.

---

<a id="cn"></a>

# 中文：GLM-5.3-Flash 单张 48 GB 卡跑满 1M 上下文（dev-glm53f 分支）

**[→ English version](#dev-glm53f)** · **中文**

本分支在**单张 48 GB GPU + 内存后备**上运行 **GLM-5.3-Flash**（180 GB Q4_K_M 混合专家模型，每层 288 专家，
KDA 线性注意力 + DSA 稀疏索引注意力混合）：**1,048,576 上下文**、**专家缓存**与**视觉适配器**同时开启，
输出正确性已验证到 250k token。prefill 采用 65,536 大批**是长输入吞吐的手段，不是目的本身**
（见下文"为什么要大 prefill 批"）；decode 走它天然的 1–16 token 小批。切出基线 fork 时该配置任何上下文都装不下：选择/打分内存随
上下文×批增长（1M 外推约 93 GB），53k+ 批触发启动上限崩溃。

## 实测速度

单一机器：**RTX 4090D 48 GB（PCIe 3.0）** · EPYC 7642，8× DDR4-2133（~100 GB/s）· GLM-5.3-Flash Q4_K_M
（179.7 GB，专家驻内存）· 1M 上下文全额预留 + q8_0 KV · 6 专家缓存槽 · 并发数 1。
完整阶梯，同一会话，每点生成 128 token：

| 提示词长度 (tokens) | 预填充耗时 (ms) | 预填充速度 (t/s) | 输出长度 (tokens) | 输出耗时 (ms) | 输出速度 (t/s) |
|---:|---:|---:|---:|---:|---:|
| 64 | 1323.16 | 48.37 | 128 | 7960.47 | 16.08 |
| 256 | 4634.11 | 55.24 | 128 | 7966.66 | 16.07 |
| 1024 | 12483.13 | 82.03 | 128 | 8524.37 | 15.02 |
| 4096 | 15075.61 | 271.70 | 128 | 8745.72 | 14.64 |
| 16384 | 27554.56 | 594.60 | 128 | 8239.76 | 15.53 |
| 65536 | 76286.40 | 859.08 | 128 | 8566.24 | 14.94 |
| 262144 | 423405.21 | 619.13 | 128 | 7262.20 | 17.63 |

读法：prefill 随批增大摊薄固定成本而爬升，满 65,536 批达峰（859 t/s）；262k 跨四个批，随每批选择状态
随上下文增长而回落到 619 t/s。decode **全阶梯平坦 14.6–17.6 t/s——含 262k（17.6）**，这是下文的图复用修复
之后；修复前从 ~2k 上下文起恒定 10 t/s 断崖。

同机补充测量：满 65k 批关专家缓存——643 t/s；缓存调度采纳（准入+前瞻）A/B——decode 14.9 → 15.5 t/s；
1M 上下文 + q8_0 KV + 缓存 + 视觉适配器全预留峰值显存 46.9 / 48 GB；正确性——算术 sanity +
69k/125k/138k/179k/250k 可核验复述全部连贯。切点时以上全部跑不起来：选择内存 1M 外推 ~93 GB、53k+ 批崩溃。

**为什么要大 prefill 批。** 专家驻内存时，每个流水趟次的固定成本与带多少 token 无关：把该批激活的专家
权重经 PCIe 上传、跨一次 CPU/GPU 调度边界。64–256 批时这些往返主导整个趟次（48–55 t/s）；4k 批仍有分量
（272 t/s）；65k 批时同样的上传摊到 16 倍的 token 上，PCIe 数据流与计算持续重叠（859 t/s）。长输入正是
收益所在——这就是 65,536 prefill 批存在的原因。decode 不付这份成本：它对专家缓存跑 1–16 token 小批，
每 token 的工作量与 prefill 批大小无关。

## 路由：什么时候走哪条路

稀疏注意力层在三条执行路径间按形状切换，而非按配置。交叉点是选择窗口（2,051 行 = 索引 top-k × 池大小）；
分块大小 512 token。

| 已有上下文 | 新输入 | 路径 | 理由 |
|---|---|---|---|
| ≤ ~2k | decode（1–16 token）或小批 | 单块 scatter | KV 画布仍小于 2,051 行的选择集——散射工作量最小 |
| > ~2k | decode（1–16 token） | gather，单块 | 每 token 工作量钉在选中的 2,051 行；画布会随上下文增长 |
| > ~2k | 小批 ≤ 512 token（追问轮次） | gather，分块 | 同一交叉点；分块约束 gather 显存 |
| 任意 | 批 > 512 token，单序列 | token 分块 scatter | 大批下 gather 的 latent 拷贝是带宽墙；分块让掩码显存有界 |
| 任意 | 多序列 / 2d-rope（视觉） | gather | scatter 分块假设单序列、1d 位置 |

选择打分本身在 池数×token数 超过 ~2^24 后切换为（上下文块 × token 片）分块 + 滚动 top-k 归并——这就是
任意上下文下峰值显存恒定的原因；更小的负载保持单块打分路径。

## 优化内容

**A. 选择内存与上下文解耦（1M 装得下的根本）。**
- *索引打分分块*：稀疏索引打分按（上下文块 × token 片）分块计算 + 滚动 top-k 归并；乘积门控让小批留在单块路径。
  选择峰值不再随上下文×批增长：1M 全额预留下 69k→250k token 实测峰值**恒定 46.8 GB**。
- *逐 token 可见计数替代稠密掩码*：[池数×token数] 因果掩码输入（1M 时数百 GB 宿主数据）换成逐 token 可见计数向量，
  块掩码在 GPU 上按需现算。
- *融合因果掩码块*：每个分块的掩码由单趟 GPU 核直接按注意力布局写出，取代 ~10 算子张量链
  （掩码流量 ~6 KB → ~1 KB/token）。

**B. 长文吞吐：token 分块稀疏 scatter。**
- *分块 scatter 注意力*：大单序列批按 token 片执行 DSA 注意力——转储行映射索引、逐片稀疏 flash attention、
  逐片输出投影。绕开 gather 路的 latent 拷贝带宽墙：同卡同批形下 gather 替代路约 ~125 t/s，分块 scatter
  关缓存 **643 t/s**、满 65k 批端到端 **859 t/s**（见上表）。
- *全 KV 库 f16 转换缓存*：量化 KV→f16 转换按整个 KV 库缓存而非逐批重转，每个满批省 ~90 GiB 冗余转换流量。

**C. 规模稳定性与确定性（64k → 262k+）。**
- *核分批发射*：凡 token/KV 维度挂在 CUDA 网格轴（上限 65,535）的核一律分批——262k 全上下文批稳定运行。
- *分块两遍式批量专家 GEMM*：每次调用的共享内存有界——53k+ 批不再触发共享内存断言/渐进错位。
- *批量 MMVQ/MMQ 网格溢出防护*：上游级 bug，任何模型在批量量化路径达此规模都会踩。
- *默认确定性 top-k*（分段 argsort）：CCCL<3.4.3 的 DeviceTopK 快速路存在竞态（上游已确认）。选择逐次可复现。
- *跨后端选择一致性*：CPU/GPU top-k/argsort 排序对齐、所有路径都填充选择输入、图重分裂时调度器绑定还原——
  分块路与单块路在任何规模下取值等价（250k 处逐 token 验证）。
- *decode 逐 token 图复用*：稀疏注意力路径把因果掩码输入换成 1 元素哑元，而图复用检查拿哑元与上下文长度
  比较，每 token 必假 → 每个 token 整图重建 + CUDA graph 重捕获——一个与上下文无关的 ~40% decode 固定税
  （15 → 10 t/s，到 262k 恒定）。检查现在识别哑元，decode 尺寸的 gather 也恢复单块形态。
  长上下文 decode：**10 → 14.6–17.6 t/s（4k–262k 全阶梯，见上表）**——任何规模无断崖。

**D. 单卡 MoE 流水线。**
- *专家链按 token 分块*：解码规模激活不再撑爆计算缓冲，专家缓存与满 65k 批在 48 GB 内共存。
- *宿主驻留专家权重每图一次全量拷贝*（多消费者共享时）：分块执行总是读到完整权重，此规模下无带宽代价。
- *专家缓存准入控制 + 动态上传前瞻*（采纳基线新线）：miss 专家须在 64 token 窗口内复现才占槽，预测上传按实测
  链路带宽/层耗时定时机。本机解码 **14.9 → 15.5 t/s（+3.8%）**，短对话命中率 12% → 14%。
- *短 prompt 断崖消除*：≤512 token 批即走缓存专家就地路径（此前仅 ≤31 批——越过该线短 prompt 损失 ~9 t/s）。

**E. 使能与插桩。**
- *GLM-5.3-Flash GGUF 兼容*（unsloth 系量化发布：架构命名、KV 前缀、视觉适配器）：五分片 Q4_K_M + mmproj 加载即用。
- *可选诊断*（仅开启时 ~1.5× 减速）：逐节点产后读回转储、分配规划活性转储、逐请求缓存 churn 日志——
  规模问题单轮可二分的工具基础。

## 与 llama.cpp 原版的区别

glm5-next / GLM-5.3-Flash 在切点时上游不支持。CPU/GPU 混合 MoE 服务、专家缓存、放置与自调优来自 neurall fork。
上文 A–E 全部为本分支工作。

## 与 neurall fork（基线）的区别

基线面向多卡 24 GB 级机器、≤64k 上下文（其自述：解码较上游 1.7–2.4×）。本分支面向**单张 48 GB 卡、1M 上下文、
65k 大批 prefill + 视觉**，新增分块 scatter 长文路径、上下文有界的选择内存、262k 级发射修复、确定性 top-k 与
跨后端选择一致性、GLM-5.3-Flash GGUF 兼容；同时采纳基线新线的缓存调度（准入/前瞻），保留旧旋钮命名。

---

# llama.cpp fork: 1.7x to 2.4x faster decode on MoE models bigger than your VRAM

The newest mixture-of-experts models (GLM-5.3-Flash, MiMo-V2.6-Flash, Qwen3.8-Flash-Next, Qwen3.6) are far bigger than a gaming GPU.
This fork keeps their experts in RAM and turns the free VRAM into a live cache of the experts the model is using; the GPUs compute
the cached ones and the CPU the rest, at the same time. No special switches needed.

```sh
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
llama-cli    -m model.gguf -p "hello"
```

## What you get

Decode tokens/s, single stream, temperature 0, model in RAM. "Upstream" is stock llama.cpp.
Machine A: 2x RTX 3090 (PCIe 4.0 x16 + chipset x4), Ryzen 7 3700X, 125 GB DDR4-3200.
Machine B, a laptop: RTX 4060 8 GB, Ryzen 9 8945HS, 32 GB LPDDR5X-6400.
Machine C: no GPU, Ryzen 5 3600, 64 GB DDR4-3200.
Machine D, rented: 4x RTX 3090 (PCIe 4.0 x16 each), EPYC 7B12 (64 cores), 256 GB DDR4 on 4 of 8 memory channels (74 GB/s read measured).
GLM 3.5-bit, IQ3_S and IQ1_M are the first run of build b11707 against fresh upstream def4d406a (no discarded run before it); the other rows are hot runs of earlier builds.

| model (size) | machine | test | upstream | this fork | |
|---|---|---|---|---|---|
| **GLM-5.3-Flash** 3.0-bit (106 GB) | A | short chat, decode | 11.9 | **22.4** | **1.9x** |
| **GLM-5.3-Flash** 3.0-bit (117.5 GB file) | D | short chat, decode, first run | 24.7 | 25.2 | 1.0x |
| | | same, second run (saved state) | 24.7 | **31.5** | **1.3x** |
| | | same, second run, `-t 16` | 24.7 | **33.8** | **1.4x** |
| | | same, 3 of the 4 GPUs, second run (best ratio) | 20.1 | **32.6** | **1.6x** |
| **MiMo-V2.6-Flash** IQ3_XXS (132 GB, bigger than RAM) | A | short chat, decode | 4.6 | **10.9** | **2.4x** |
| | | 12k-token prompt, decode | 4.2 | **9.1** | **2.2x** |
| | | 12k-token prompt, processing | 156 | 112 | 0.7x |
| **Qwen3.8-Flash-Next** UD-IQ4_XS (88 GB) * | A | short chat, decode | 27.7 | **46.5** | **1.7x** |
| | | 12k-token prompt, decode / processing | 25.3 / 500 | **42.3 / 538** | 1.7x / 1.1x |
| **GLM-5.3-Flash** 3.5-bit (137 GB, bigger than RAM) | A | short tetris prompt, 100 tokens, decode (text-dependent) | 6.9 | **15.1** | **2.2x** |
| **Qwen3.8-Flash-Next** GSQ IQ3_S (83 GB) | A | short tetris prompt, 100 tokens, decode | 43.9 | **57.1** | **1.3x** |
| **Qwen3.8-Flash-Next** GSQ IQ1_M (55 GB, barely over 48 GB VRAM) | A | same | 69.1 | 67.5 (picks stock) | 1.0x |
| | B | same | 11.4 | **13.5** | **1.2x** |
| | B | same, prompt processing | 15.5 | 5.9 | 0.4x |
| **Qwen3.6-35B-A3B** Q2_0 (11 GB, on an 8 GB GPU) | B | short tetris prompt, 100 tokens, decode | 29.2 | **59.3** | **2.0x** |
| | C | same, CPU only (AVX Q2_0 kernels) | 6.3 | **11.3** | **1.8x** |
| Qwen3.8-27B IQ4_NL, dense (fits VRAM) | A | same | 45.1 | 45.0 | 1.0x |
| Qwen3.8-27B IQ3_S, dense, CPU only | C | same | 1.7 | 1.6 | 1.0x |
| Qwen3.8-27B Q5_K_M with MTP (`--spec-type draft-mtp`), fits VRAM | A | same | 78.3 | 77.0 | 1.0x (38.6 without MTP) |
| | | 2.2k-token prompt, decode / processing | 31.3 / 599 | **51.1 / 792** | 1.6x / 1.3x |
| Models that fit in VRAM | any | anything | same | same | 1.0x (cache off) |

D: upstream is the downloaded release b11323 (a source build of def4d406a gave 24.8 and 20.1); the fork is b11707 with the placement change in this branch (any model that does not fit takes the cache);
the model is 85% in VRAM on 4 GPUs, so the first run only matches stock placement. D numbers are from one session on a rented box (raw logs not kept, not in run-history.csv).

\* from the previous release. GLM and MiMo were measured on release-candidate builds (MiMo also on b11509) before the last placement and thread commits, Qwen3.6 on the release binary. Every run behind these numbers (commit, build, machine, settings) is in
[`tools/bench/run-history.csv`](tools/bench/run-history.csv).

## Nothing to configure

The defaults are chosen on your machine, not hard-coded:

- **Cache or stock.** If the model fits in VRAM it is placed exactly like stock llama.cpp. A model that does not fit takes
  the cache right away. When what you run is mostly long prompts (processing them would cost more than the faster generation
  gains), the first two runs compare both and keep the faster one.
- **Self-tuning on real token times.** The cache policy, the upload schedule and the decode and prompt thread counts are
  adjusted while you use it. A setting that does not help is dropped, a setting you fix yourself is never touched.
- **It remembers.** What it learned per model (hot experts, tuned settings, cache-or-stock) is kept in one file,
  `~/.cache/llama.cpp/moe-state.ini`, so the next start, even a one-shot short prompt, begins from it. Delete the file to start over.
- **It tells you what it does.** `llama-server` logs, and `llama-cli -lv 3` prints after each reply, whether the cache is on,
  the hit rate, the tuned settings, threads and batch sizes.

## MTP speculative decoding

Qwen3.8-Flash-Next MTP from PR [#28243](https://github.com/ggml-org/llama.cpp/pull/28243)
([@danielhanchen](https://github.com/danielhanchen)), GLM-5.3-Flash MTP from PR
[#27917](https://github.com/ggml-org/llama.cpp/pull/27917) (timkhronos), both in this build. Load a model's MTP draft head
with `-md`:

```sh
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
    -md mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf --spec-type draft-mtp
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf \
    -md GLM-5.3-Flash-MTP-Q4_K.gguf --spec-type draft-mtp --spec-draft-n-max 3
```

- MTP heads: Qwen3.8-Flash-Next from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
  (`MTP/`, the `shared` files reuse the main model's embeddings); GLM-5.3-Flash from
  [neuralll/GLM-5.3-Flash-MTP-GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-MTP-GGUF)
  (4.3 GiB, works with any `glm5-next` GLM-5.3-Flash GGUF). We made it: no GLM MTP GGUF existed, so
  `tools/bench/glm_splice_mtp.py` pulls just the MTP tensors out of unsloth's UD-Q4_K_XL GGUF with HTTP range requests
  (~4.3 GiB instead of the whole model) and writes them as a draft file; see [`tools/bench/`](tools/bench/) to rebuild it.
- GLM MTP: 89% of drafts accepted in our test, output identical to plain decoding. For a model far bigger than VRAM the draft
  is not loaded unless you pass `--spec-draft-n-max`.

## Your settings win

Anything you pass is used as given and is never auto-tuned:

| you pass | effect |
|---|---|
| `-t N`, `-tb N` | fixed thread counts |
| `--moe-expert-cache N` | cache slots per layer; `0` turns the cache off, `-1` sizes it from free VRAM |
| `--moe KEY=VAL,...` | `cache`, `prefetch-slots`, `inserts`, `window`, `predict`, `train`, or any tuning knob (`MARGIN`, `GATE`, `WAIT`, `BIG`, `SWAP_FRAC`, ...), for example `--moe gate=3,margin=0` |
| `--load-mode pin\|mmap` | pinned weights (the server default when the model fits in RAM, faster prompts) or mmap |
| `LLAMA_MOE_AUTO_MODE=stock\|cache` | force the placement; `LLAMA_MOE_STATE=0` ignores and never writes the state file |

## Turn it off

| you pass | effect |
|---|---|
| `--moe cache=0` | the whole fork off: no expert cache, nothing tuned or measured, plain stock behaviour (same as `--moe-expert-cache 0`) |
| `-at off` | only the self-tuning off (`--autotune off`, same as `--moe autotune=0`): the cache keeps working with fixed defaults, placement uses a static rule, nothing is measured or saved |

Environment forms: `LLAMA_AUTOTUNE=0`, `LLAMA_ARG_AUTOTUNE=off`, `LLAMA_ARG_MOE=cache=0`.

## Good to know

- **RAM is the limit.** Decode speed is bound by how fast the CPU reads the experts that are not in VRAM. More or faster RAM,
  more VRAM or a faster GPU link all raise it.
- Output can differ slightly from stock at temperature 0: a cached expert runs on the GPU, a missed one on the CPU.
- Prompt processing of models bigger than RAM (mmap) is 20-30% below stock: the experts stream over PCIe.
- A second GPU on a slow slot helps less; prompt processing goes to the fastest link.

## Credits and contact

The expert cache builds on [@csantiago78](https://github.com/csantiago78)'s llama.cpp PR
[#27861](https://github.com/ggml-org/llama.cpp/pull/27861); GLM-5.3-Flash support is upstream
([#27773](https://github.com/ggml-org/llama.cpp/pull/27773)).

**About the author of this fork**: I'm actively looking for an AI engineering/research
role and open to relocating out of Eastern Europe. If this work is useful to you or
your team, reach out: [linkedin.com/in/neuralll](https://www.linkedin.com/in/neuralll/)

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
