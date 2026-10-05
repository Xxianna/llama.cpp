# llama.cpp-glm-fast

更好的 GLM-5.3-Flash 混合推理方案。

- 混合推理：更低的显存需求
- 修复对GLM-5.3-Flash 1M上下文和大batch叠加的支持，降低长输入对PCIE带宽的依赖
- 保留热门专家缓存机制，降低cpu计算量
- 分段选择prefill、decode计算方式，提高不同上下文背景下的速度

**在线页面**：<https://xxianna.github.io/llama.cpp-glm-fast/>（GitHub Pages，页内中/EN 切换）

## 1. 测试数据

<details>
<summary><b>RTX 4090D 48 GB（PCIe 3.0）· EPYC 7642 · GLM-5.3-Flash Q4_K_M · 1M 上下文</b></summary>

### 测试环境

| 项目 | 配置 |
| --- | --- |
| GPU | RTX 4090D 48 GB，PCIe 3.0 |
| CPU / 内存 | EPYC 7642（48 核），8 通道 DDR4-2133，实测约 100 GB/s |
| 模型 | GLM-5.3-Flash Uncensored Q4_K_M（179.7 GB，4.81 BPW） |
| 推理形态 | 非专家权重与 KV（q8_0）驻 GPU；路由专家驻 CPU 内存；6 专家缓存槽/层 |
| 服务参数 | 上下文 1,048,576；batch 65,536；并发 1 |

### 性能

完整阶梯测量，同一会话，每点输入后生成 128 token：

| 输入长度 (token) | Prefill (token/s) | Decode (token/s) |
| ---: | ---: | ---: |
| 64 | 48.37 | 16.08 |
| 256 | 55.24 | 16.07 |
| 1,024 | 82.03 | 15.02 |
| 4,096 | 271.70 | 14.64 |
| 16,384 | 594.60 | 15.53 |
| 65,536 | 859.08 | 14.94 |
| 262,144 | 619.13 | 17.63 |

批大小 65,536 的设置依据：每趟流水线存在与 token 数无关的固定开销（激活专家权重的 PCIe 上传、CPU/GPU 调度边界），批越大摊得越薄——64 token 批 48 token/s，65,536 批 859 token/s。

显存：1M 上下文全额预留（q8_0 KV）+ 专家缓存 + 视觉适配器，峰值 46.9 / 48 GB，且不随输入长度增长（69k–250k 实测恒定）。

### 启动命令

```bash
env CUDA_VISIBLE_DEVICES=<GPU UUID> \
    GGML_SCHED_H2D_ASYNC=1 \
    LLAMA_MOE_CACHE_MAX_BATCH=512 \
    GGML_DSA_SCATTER_TILE=512 \
    ./build-release/bin/llama-server \
    -m <模型路径>/GLM-5.3-Flash-Uncensored-Q4_K_M-00001-of-00005.gguf \
    --mmproj <模型路径>/mmproj-GLM-5.3-Flash-Uncensored-F16.gguf \
    -ngl 99 --cpu-moe --moe cache=6 -fa on -np 1 \
    -c 1048576 -ctk q8_0 -ctv q8_0 -t 48 -b 65536 -ub 65536 \
    --host 0.0.0.0 --port 8300 --alias glm53f --jinja
```

环境变量：`H2D_ASYNC` 为 CPU→GPU 输入异步拷贝；`MAX_BATCH=512` 使 ≤512 token 批走缓存专家就地路径；`SCATTER_TILE` 为 tiled scatter 分块大小。`cache=6` 为每层专家缓存槽数，是 48 GB 下的显存弹性项。

</details>

## 2. 显存占用估算

显存 ≈ GPU 驻留权重 + KV 缓存 + KDA 循环状态 + 视觉适配器 + k16 转换缓存 + 专家缓存 + 计算池：

| 分项 | 决定参数 | 估算（Q4_K_M + q8_0 KV） |
| --- | --- | --- |
| 非专家权重 | 模型 | 8.7 GB（Q4_K_M） |
| KV 缓存 | 上下文（`-c`） | ≈ 10 KiB/token（-ctk q8_0 -ctv q8_0） |
| KDA 循环状态 | 层数（常数） | 0.15 GB |
| 视觉 | --mmproj | 1.2 GB（f16 mmproj） |
| k16 转换缓存 | 上下文（`-c`） | ≈ 1.45 KiB/token（仅量化 KV 时存在） |
| 专家缓存 | 缓存槽数（`--moe cache=N`） | ≈ 0.59 GB/槽（Q4_K_M） |
| 计算池 | `-ub` 为主，弱 `-c` 项 | ≈ 0.28 MB/token × `-ub` + 1 KiB/token × `-c` |
| 其他 | — | ≈ 0.5 GB |

显存不足时的调节：

1. **缓存槽数**（`--moe cache=N`）：降低影响decode速度
2. **batch**（`-b`/`-ub`）：影响长单次输入prefill速度，4090+pcie3推荐16384,推荐值和算力正比、和pcie带宽反比
3. **上下文**（`-c`）

实测锚点：1M 上下文 + 6 槽 + 65,536 批 → 空闲 44.6 GB，满载峰值 46.9 GB。

## 3. 路径路由

DSA 层按批次形状在三条执行路径间选择，与运行配置无关。两个阈值：

- 选择窗口 n_sel = 2,051 cell（= 4 cell/池 × 512 池 + 3 尾格）；
- 分块大小 512 token。

| 负载 | 已有上下文 | 路径 | 原因 |
| --- | --- | --- | --- |
| prefill，> 512 token | 任意 | tiled scatter | 大批下 gather 需搬运 TB 级选中 latent；scatter 分块使掩码显存有界 |
| prefill，≤ 512 token | ≤ 2,051 | 单块 scatter | KV 画布小于选择窗口，直接散射开销最低 |
| prefill，≤ 512 token | > 2,051 | gather，分块 | 选择窗口小于画布；分块约束显存 |
| decode（1–16 token/步） | ≤ 2,051 | 单块 scatter | 同上 |
| decode | > 2,051 | gather，单块 | 单步开销恒为 2,051 cell，与上下文长度无关 |
| 多序列 / 视觉（2d-rope） | 任意 | gather | scatter 路径仅支持单序列、一维位置 |

索引打分在 池数 × token 数 > 2²⁴ 时切换为（上下文块 × token 片）分块计算 + 滚动 top-k 归并，更小负载保持单块路径。该机制是 1M 上下文峰值显存恒定的直接原因。

## 4. 优化内容

### 4.1 上下文有界的选择内存

- **索引打分分块**：得分按（上下文块 × token 片）计算，滚动 top-k 归并；乘积门控让小负载保持单块路径。选择阶段峰值显存不再随 上下文 × 批 增长——1M 预留下 69k–250k 输入实测恒定，单块路径在 1M 外推 93 GB。
- **逐 token 可见计数**：以每 token 可见池计数向量取代 [池数 × token 数] 的稠密因果掩码输入，块掩码由 GPU 现算；消除 1M 上下文下数百 GB 的宿主侧掩码输入及其 GPU 副本。
- **融合掩码块**：单趟 GPU 核将因果掩码块直写为注意力布局，取代约 10 算子的张量链，掩码数据量 6 → 1 KB/token。

### 4.2 Prefill 吞吐

- **token 分块 scatter**：大单序列批按 token 片执行 DSA 注意力——转储行映射索引、逐片稀疏 flash attention、逐片输出投影。绕开 gather 的 latent 搬运瓶颈：同卡 65k 批形下 gather 约 125 token/s，分块 scatter 643（关缓存）至 859（端到端）token/s。
- **全 KV 库 f16 转换缓存**：量化 KV → f16 转换按整个 KV 库缓存而非每批重转，每个满批减少约 90 GiB 冗余转换流量。

### 4.3 规模稳定性与确定性（64k → 262k+）

- **核分批发射**：token/KV 维度超过 CUDA 网格轴上限（65,535）的核分批发射；262k 全上下文批稳定运行，此前 53k+ 批触发 CUDA invalid argument。
- **分块两遍式专家 GEMM**：批量量化矩阵乘按共享内存上限分块、两遍式刷写；53k+ 批不再触发共享内存断言与刷写错位。
- **MMVQ/MMQ 网格防护**：批量输入 gridDim.z 溢出防护。上游级缺陷，任何模型在批量量化路径达此规模均会触发。
- **确定性 top-k**：默认走分段 argsort 路径，规避 CCCL < 3.4.3 的 DeviceTopK 竞态（上游已确认）；选择结果逐次可复现。
- **跨后端选择一致性**：CPU/GPU 的 top-k 与 argsort 排序对齐，所有路径填充全部选择输入，图重分裂时恢复调度器绑定；分块与单块选择路径在任意规模下取值等价（250k 逐 token 验证）。
- **decode 逐 token 图复用**：图复用检查识别稀疏路径的 1 元素哑元掩码，消除长上下文 decode 每 token 整图重建的固定开销；10 → 14.6–17.6 token/s（4k–262k）。

### 4.4 MoE 混合流水线

- **专家链按 token 分块**：路由专家计算链按 token 切分，解码规模激活不再占用整批缓冲；专家缓存与 65k 满批共存于 48 GB。
- **宿主权重每图全量拷贝**：多消费者共享的 CPU 驻留专家权重每图拷贝一次，分块执行始终读取完整权重，大批下无额外带宽代价。
- **准入控制 + 动态上传前瞻**（采纳自基线新线）：miss 专家须在 64 token 窗口内复现方可占槽，预测上传按实测链路带宽与层耗时定时；本机解码 14.9 → 15.5 token/s（+3.8%），短对话命中率 12% → 14%。
- **短批缓存阈值**：缓存专家就地路径的批上限由 31 提升至 512，消除短输入越过该线后约 9 token/s 的损失。

### 4.5 兼容与工具

- **GGUF 兼容**：unsloth 系 GLM-5.3-Flash 量化发布（架构命名、KV 前缀、视觉适配器），五分片 Q4_K_M + mmproj 直接加载。
- **诊断工具**：逐节点产后读回转储、分配规划活性转储、逐请求缓存换入换出日志；仅显式开启时生效，约 1.5 倍减速。

## 5. 与基线的关系

| 对比对象 | 基线状态 | 本分支 |
| --- | --- | --- |
| llama.cpp 上游 | 不支持 glm5-next 架构 | 经兼容层加载运行；混合推理与专家缓存基础设施来自 neurall fork；第 4 节全部为本分支工作 |
| neurall fork | 面向多卡 24 GB 级、≤ 64k 上下文（自述解码较上游 1.7–2.4 倍） | 面向单卡 48 GB、1M 上下文、65k prefill 批、视觉输入；新增 tiled scatter、有界选择内存、262k 级稳定性、确定性 top-k、跨后端一致性、GGUF 兼容；采纳基线新线的缓存调度，保留旧旋钮命名 |

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
