# GPU memory: why a model OOMs with compute to spare

> **Week 1 · Day 5** · Deliverable for `nvidia-ai-plan/week-01/day-05`
> Status: **draft: model and predictions done, OOM boundary pending pod run** · Last updated: **2026-09-30**
> Code: [`01-cuda/vram-lab`](../01-cuda/vram-lab/) · Environment: [`00-environment/versions.md`](../00-environment/versions.md)

> ⏳ **Review note (2026-09-30).** The explanation, KV-cache math and predictions are filled in.
> The measured OOM boundary (§4) needs ≈ 10 min on an L4 (`python vram_probe.py`, commands in the
> [vram-lab README](../01-cuda/vram-lab/README.md)). Nothing here is a measurement yet.

**One-line answer.** VRAM is a *capacity* budget and compute is a *rate* budget. An LLM's weights are a fixed
cost, but its **KV cache and activations grow with batch × context**, so you run out of bytes long before you
run out of FLOPs.

---

## 1. The memory hierarchy (L4)

Capacity and bandwidth move in opposite directions as you go outward. That one fact explains tiling, shared memory, and why "just load it into VRAM" is where the trouble starts.

| Level | Scope | Size (order of) | Speed | What lives there |
|---|---|---|---|---|
| Registers | one thread | ~255 × 32-bit per thread | fastest | the values a thread is working on right now |
| Shared memory / L1 | one SM (block) | tens of KB per SM | ~register speed | tiles a block reuses (matmul tiles, attention blocks) |
| L2 cache | whole GPU | tens of MB | fast | recently touched global memory |
| Global memory (VRAM, GDDR6) | whole GPU | **24 GB** | **300 GB/s** | weights, KV cache, activations, everything PyTorch allocates |
| Host RAM over PCIe | CPU | 100s of GB | **Gen4 x16: 64 GB/s both directions (≈ 32 each way)** | data before `.to("cuda")`, offloaded layers |
| Other GPUs (NVLink) | node | n/a on L4 | L4 has **no NVLink**; multi-GPU goes over PCIe | tensor-parallel shards (Week 2 Day 5, Week 5 Day 6) |

Sources: [CUDA Programming Guide, memory hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy) ·
[L4 product page](https://www.nvidia.com/en-us/data-center/l4/) (24 GB, 300 GB/s, PCIe Gen4 x16 64 GB/s; checked 2026-09-27).
Per-SM sizes vary by architecture: read the exact ones from `torch.cuda.get_device_properties(0)` on the pod.

## 2. What consumes VRAM when serving an LLM

| Consumer | Scales with | Fixed or variable? |
|---|---|---|
| **CUDA context** | driver/CUDA version, loaded kernels | fixed, ~0.3–0.6 GB, invisible to PyTorch |
| **Weights** | parameters × bytes (FP16 = 2 B, FP8/INT8 = 1 B, INT4 = 0.5 B) | fixed once loaded |
| **KV cache** | **batch × context × layers × KV heads × head_dim × 2 (K and V) × bytes** | variable: grows with traffic |
| **Activations** | batch × tokens-in-this-pass × hidden/intermediate size | transient; large in prefill, tiny in decode |
| **Allocator slack** | fragmentation, cached-but-free blocks | variable |
| **Logits** | batch × tokens × **vocab** × bytes | transient; huge if you compute them for every prompt token (vocab ≈ 150k) |

**KV cache per token (FP16):** `2 × layers × kv_heads × head_dim × 2 bytes`. It does **not** depend on parameter count directly:
a model with fewer KV heads (GQA) has a smaller cache per token than a smaller model without GQA.

## 3. Worked numbers: what fits on a 24 GB L4

Assume ≈ 22.5 GiB usable (24 GB decimal ≈ 22.4 GiB; `mem_get_info` reports the exact figure) and ≈ 0.5 GiB of CUDA context.

| Model (FP16) | Layers / KV heads / head_dim | Weights | KV per token | Room left for KV | Max cached tokens (all sequences) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 28 / 2 / 128 | ≈ 2.9 GiB | **28 KiB** | ≈ 19 GiB | ≈ 700k |
| Llama-3.1-8B (GQA) | 32 / 8 / 128 | ≈ 15 GiB | **128 KiB** | ≈ 7 GiB | ≈ 57k |
| Llama-2-7B (no GQA) | 32 / 32 / 128 | ≈ 12.5 GiB | **512 KiB** | ≈ 9.5 GiB | ≈ 19k |

Qwen2.5-1.5B figures from its [`config.json`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) (checked 2026-09-30).
The Llama rows use their published configs and are the same math as `01-cuda/week1-catchup/gpu_intuition.py`, Part B.

What that means for Llama-3.1-8B on an L4:

| batch × context | KV cache | Fits? |
|---|---|---|
| 1 × 4k | 0.5 GiB | yes |
| 8 × 4k | 4 GiB | yes |
| 16 × 4k | 8 GiB | **no**, ~1 GiB over |
| 1 × 128k (its max context) | 16 GiB | **no**: one long-context user alone doesn't fit next to the weights |

**Takeaways**
- GQA cut the KV cache 4× (Llama-2 → Llama-3). That's why modern models use it.
- Halving weight bytes (FP8/INT4, Weeks 3 & 5) frees VRAM *for KV cache*, and that is often the real reason to quantize on a 24 GB card.
- Serving engines (TensorRT-LLM, vLLM) pre-allocate a KV pool and page it in blocks. That's why they grab ~90% of VRAM at start-up (Week 5 Day 5, Week 6).

## 4. Measured OOM boundary

⏳ **Pending pod run.** `python vram_probe.py` (Qwen2.5-1.5B, FP16, context 4096, prefill with `use_cache=True`, batch doubled then binary-searched).

**Prediction (written before the run):** the KV cache is only 28 KiB/token, but prefill *activations* in the MLP
(intermediate size 8960: gate, up and product tensors alive at once) add roughly 50–65 KiB per token in flight.
So I expect OOM at around **batch 48–56 × 4096** (~200–230k tokens), with the measured KV cache at OOM
**well under** the free memory. If that's what happens, prefill activations, not the KV cache, set the limit.
That's exactly why serving engines use **chunked prefill**. Decode, by contrast, is KV-bound.

| Field | Value |
|---|---|
| Card / VRAM | … (from `results/vram_*.json`) |
| Boundary | *"OOM at batch … / context 4096 on NVIDIA L4, … GiB"* |
| Weights (allocator) | … GiB (predicted ≈ 2.9) |
| KV at last OK batch: formula vs measured | … vs … GiB |
| Peak allocated at last OK batch | … GiB |
| What consumed VRAM before the OOM | … |

## 5. `memory_allocated()` vs `nvidia-smi`: they never match

| Number | Who reports it | Includes |
|---|---|---|
| `torch.cuda.memory_allocated()` | PyTorch | only tensors that are alive right now |
| `torch.cuda.memory_reserved()` | PyTorch | allocated + blocks the **caching allocator** keeps for reuse after tensors are freed |
| `nvidia-smi` memory.used (= `mem_get_info` total − free) | driver | reserved + **CUDA context** (+ cuBLAS/cuDNN workspaces, + other processes) |

So `nvidia-smi ≥ reserved ≥ allocated`, always. The caching allocator doesn't return freed memory to the driver
(that's what makes the next allocation fast), so `nvidia-smi` staying high after `del tensor` is **not** a leak.
`torch.cuda.empty_cache()` gives the cached blocks back. Source: [PyTorch CUDA semantics, memory management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management).

Measured gap on the pod: ⏳ `driver_used − reserved` = … GiB (the probe prints `context_overhead_gib` after loading).

## 6. Financial-crime implications

- **Capacity planning is KV planning.** For an AML investigation assistant, sizing is concurrent investigators × context length, not model size alone. 16 analysts each pasting a 4k-token case file into Llama-3.1-8B already overflows one L4.
- **Long case files are expensive per user.** A 128k-token SAR narrative plus the transaction history needs its own GPU (or a quantized KV cache). Summarize or retrieve (RAG, Week 8) instead of stuffing the context.
- **Watch memory, not just GPU utilization.** Alert on KV-cache usage and preemptions (DCGM + engine metrics, Week 6), because GPU util can look healthy right up to the OOM.

## 7. Done when

- [ ] You can name what consumed VRAM before the OOM (§4, after the run)
- [ ] Boundary recorded **with the card and VRAM** it applies to
- [ ] Both allocator-reported and `nvidia-smi` memory captured, with the gap explained (explanation §5 ✔, numbers pending)
- [x] You can state how KV cache scales, and what that predicts for long-context serving (§2–§3)
- [ ] Pod stopped/destroyed; spend logged
