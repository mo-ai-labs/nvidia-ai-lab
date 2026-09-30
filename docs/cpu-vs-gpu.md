# CPU vs GPU: mental model + measured crossover

> **Week 1 · Day 4** · Deliverable for `nvidia-ai-plan/week-01/day-04`
> Status: **draft: predictions and analysis done, measurements pending** · Last updated: **2026-09-30** · Code: [`01-cuda/matmul-bench`](../01-cuda/matmul-bench/)
> Raw data: `01-cuda/matmul-bench/results/*.csv` · Environment: [`00-environment/versions.md`](../00-environment/versions.md)

**Purpose.** Know *why* GPUs win for AI and, more importantly, *when they don't*. Back every claim with a number from my own sweep, not a product page.

> ⏳ **Review note (2026-09-30).** Sections 1, 6, 7 and 8 are filled in. Sections 2, 4 and 5 need the pod run
> (≈ 15 min on an L4, ≈ $0.15). Every line that is still a prediction says so. Commands are in
> [`matmul-bench/README.md`](../01-cuda/matmul-bench/README.md). No number in this doc is a measurement yet.

**Rules for filling this in**

- Fill §1 **before** renting the pod. A prediction written after seeing the data isn't a prediction.
- Every number states the **card, dtype and mode** it came from.
- Prefer GFLOP/s over seconds: seconds hide the story.
- Any datasheet figure gets a **source + date**, and note dense vs 2:4-sparsity.

---

## 1. Predict before you measure

### 1.1 Datasheet numbers (L4)

| Quantity | Value | Dense or sparse? | Source (checked on) |
| --- | --- | --- | --- |
| SMs / CUDA cores / Tensor Cores | … / … / … (fill from the pod: `torch.cuda.get_device_properties(0)`; the sweep's `results/pod_cuda_*.json` records `sm_count`) | n/a | |
| Peak FP32 | 30.3 TFLOPS | dense (no asterisk) | L4 page, Product Specifications (2026-09-27) |
| Peak TF32 (Tensor Core) | 120\* TFLOPS → **60 dense** | sparse (2:4); dense = ½ | L4 page, Product Specifications (2026-09-27) |
| Peak FP16 / BF16 (Tensor Core) | 242\* TFLOPS → **121 dense** | sparse (2:4); dense = ½ | L4 page, Product Specifications (2026-09-27) |
| Peak FP8 (Tensor Core) | 485\* TFLOPS → **242.5 dense** | sparse (2:4); dense = ½ | L4 page, Product Specifications (2026-09-27) |
| Memory | 24 GB | n/a | L4 page, Product Specifications (2026-09-27) |
| Memory bandwidth | 300 GB/s | n/a | L4 page, Product Specifications (2026-09-27) |
| Host link (PCIe) | Gen4 x16, 64 GB/s (both directions) | n/a | L4 page, Product Specifications (2026-09-27) |
| Board power limit (TDP) | 72 W | n/a | L4 page, Product Specifications (2026-09-27) |

> \* Footnote: "Shown with sparsity. Specifications are one-half lower without sparsity." My matmul is **dense**, so % of peak uses the dense figures (FP32 30.3 · TF32 60 · FP16 121).

### 1.2 Roofline: when is a matmul compute-bound?

- Arithmetic intensity of N×N matmul: `AI = 2N³ FLOPs ÷ 3N²·bytes = 2N / (3·bytes)` FLOP/byte (read A and B once, write C once: the best case a tiled kernel approaches)
- Ridge point = `peak FLOP/s ÷ memory bandwidth`:
  - FP32: 30.3 T ÷ 300 G = **≈ 101 FLOP/byte**
  - TF32: 60 T ÷ 300 G = **≈ 200 FLOP/byte**
  - FP16: 121 T ÷ 300 G = **≈ 403 FLOP/byte**
- Compute-bound once `AI > ridge`, i.e. `N > 1.5 · bytes · ridge`:
  - FP32 (4 B): `N > 1.5 × 4 × 101 ≈ 606`, so **from N = 1024** in a powers-of-2 sweep
  - TF32 (4 B): `N > 1.5 × 4 × 200 ≈ 1200`, so **from N = 2048**
  - FP16 (2 B): `N > 1.5 × 2 × 403 ≈ 1210`, so **from N = 2048**
- Below those sizes the Tensor Cores wait on memory, so FP16's 4× peak advantage over FP32 can't show up yet.
  Real kernels re-read tiles, so the true threshold sits a bit higher than this ideal.

### 1.3 The PCIe tax (e2e mode)

- Bytes over PCIe per call: `3·N²·bytes` (A and B in, C out). Compute: `2N³ / achieved FLOP/s`
- 64 GB/s is both directions combined, so ≈ 32 GB/s each way on paper. Assume **≈ 25 GB/s** in practice (less from pageable memory, which is what `tensor.to("cuda")` uses by default).
- Transfer ≈ compute when `3N²b / BW = 2N³ / F` → **`N = 3·b·F / (2·BW)`**
  - FP32, F ≈ 25 TFLOPS achieved: `N ≈ 3 × 4 × 25e12 / (2 × 25e9)` **≈ 6,000**
  - FP16, F ≈ 80 TFLOPS achieved: `N ≈ 3 × 2 × 80e12 / (2 × 25e9)` **≈ 9,600**
- ⇒ **Prediction:** across the whole sweep (≤ 8192), e2e time on the L4 is dominated by the copy, not the math. The GPU still beats the CPU end to end at large N, because the CPU's compute is slower than the copy. The real lesson is **keep data resident on the GPU**.

### 1.4 My predictions

| Bet | Prediction | Reasoning |
| --- | --- | --- |
| A — smallest N where L4 beats pod CPU | 100 | **Compute mode.** A kernel launch costs ~5–10 µs. A multi-core CPU does N = 64 in ~10 µs and N = 128 in ~25 µs (a cloud-VM CPU measured 0.009 ms and 0.025 ms), so the GPU should pull ahead between 64 and 128. **e2e mode** adds three copies with ~10 µs latency each, so I expect the e2e crossover later, around N ≈ 256–512. |
| B — % of FP16 peak at N=8192 | 50% | Large cuBLAS GEMMs usually reach 70–90% of dense peak. But the L4 is capped at **72 W**, and sustained Tensor Core load pulls the SM clock below boost. Watch `nvidia-smi dmon` for the clock drop. Estimate 50–70%. |
| C — how many × too fast the no-sync timing claims | 20× | Naive timing measures launch enqueue (~5–10 µs), not compute. At N = 8192 FP16 the real matmul takes ~15 ms, so the inflation could be **100–1000×** at large N and only a few × at small N. 20× is probably too low for big N. |

---

## 2. Hardware under test

⏳ Fill from `results/*.json` after the runs.

| Tag | Device | Model | Peak used | torch / CUDA |
| --- | ------ | ----- | --------- | ------------ |
| pod | cuda   | NVIDIA L4 (expected) | FP32 30.3 · TF32 60 · FP16 121 TFLOPS dense | … / … |
| pod | cpu    | … (`cpu` + `cpu_threads` in the json) | n/a | … |
| mac | mps    | Apple GPU (MPS) | n/a (no official FLOPS figure used) | … |

## 3. Method

- Sweep N = 16 … 8192 (powers of 2). CPU capped at 4096.
- Per point: 3 warm-up calls → synchronize → iterations calibrated so each repeat is ≥ 0.2 s → 5 repeats → median.
- Modes: `compute` (on-device, synced), `e2e` (copy in + multiply + copy out), `naive` (no final sync, kept to show the bug).
- `fp32` on CUDA runs with `allow_tf32 = False` (true FP32); `tf32` turns Tensor Core TF32 on.
- Deviations today: none yet.

## 4. Results

⏳ **Pending pod run.** Paste from `results/summary.md` (`uv run report.py --latest`).

### 4.1 Achieved GFLOP/s

_(table from summary.md)_

### 4.2 Median latency

_(table from summary.md)_

### 4.3 Charts

![GFLOP/s](img/cpu-vs-gpu-gflops.png)
![Latency](img/cpu-vs-gpu-latency.png)

## 5. Analysis

⏳ Each line below states the **prediction** from §1. Replace it with the measured value and say whether the bet won.

### 5.1 The crossover

- Compute-only: GPU wins from N ≥ … (predicted 64–128) · e2e: GPU wins from N ≥ … (predicted 256–512)
- Vs Bet A: right / wrong, because …

### 5.2 How close to peak?

- Best FP16 GFLOP/s ÷ 121,000 = …% (predicted 50–70%) · Why not 100%: power cap (72 W) → lower SM clock; small N can't fill all SMs; memory-bound below N ≈ 2048
- Vs Bet B: …

### 5.3 Where the CPU wins

- Expected: N ≤ 64 in compute mode (launch overhead), and N ≤ ~256 in e2e (PCIe copy latency + bandwidth). Measured: …

### 5.4 The no-sync bug

- Naive claimed … GFLOP/s = …× peak (impossible) · Vs Bet C: …

## 6. Mental model (my own words)

- **SM:** a GPU is made of Streaming Multiprocessors (SMs), grouped into Graphics Processing Clusters (GPCs).
- **Warp:** a group of 32 threads that executes kernel code in a Single-Instruction Multiple-Threads (SIMT) paradigm.
- **Warp divergence:** when different threads in a warp follow different code paths, like in conditional control-flow branches. It hurts because when some threads have to follow one branch, the others have to be masked off doing nothing until the first group is finished (a 50/50 `if/else` ≈ **2×** slower).
- **CUDA core vs Tensor Core:** a CUDA core does one multiply-add on single numbers per step; a Tensor Core does a whole small matrix multiply-add (a tile) per step, which is why TF32/FP16 peaks are 2–4× FP32 on the L4 (dense: 60 / 121 vs 30.3 TFLOPS).
- **Why GPUs need big problems:** every GPU call pays a fixed cost (kernel launch ~µs, plus PCIe copies if data isn't already there) and needs enough independent work to keep every SM busy. Matmul work grows as N³ while data grows as N², so only large problems have enough arithmetic per byte to hide memory and launch costs.
- **One-sentence crossover:** the GPU wins once the work (∝ N³) is big enough to pay back its fixed launch-and-copy cost (∝ constant + N²), so it wins early when data already lives on the GPU and much later when every call has to cross PCIe.

## 7. Financial-crime implications

| Workload                   | Shape                  | CPU or GPU, why |
| -------------------------- | ---------------------- | --------------- |
| Real-time single-txn score | 1 row, tight SLA       | **CPU** for a single tree/linear model: one row is far below any crossover, and a PCIe round trip plus launch adds latency for nothing. GPU only makes sense if requests are **batched** server-side (Triton dynamic batching, Week 6) or the model is a large deep net. |
| Nightly batch rescoring    | millions of rows       | **GPU**: huge N, the data can be loaded once and stay resident, so the PCIe tax is paid once, not per row. RAPIDS cuDF/cuML or Triton FIL for XGBoost-style models. |
| LLM investigation summary  | large matmuls + decode | **GPU, no contest**: billions of parameters and large matmuls. But decode at batch 1 is memory-bandwidth-bound (tiny N per step), so throughput comes from batching and KV-cache management (Day 5, Weeks 4–6), not raw FLOPS. |

## 8. Open questions → later weeks

- [ ] Why does `e2e` stay PCIe-bound, and do pinned memory and CUDA streams fix it? → Week 2 Day 2 (transfers & streams)
- [ ] Does the L4 hold its boost clock under sustained FP16, or does the 72 W cap throttle it? → Week 2 Day 3 (Nsight)
- [ ] What happens to memory at the sizes where the GPU wins (weights + activations + KV cache)? → Day 5
- [ ] FP8 matmul on the L4 (Ada supports it) vs FP16: how close to 2×? → Week 3 Day 3 / Week 5 Day 4

## 9. Done when

- [x] §1 written before the run
- [ ] Crossover explained (compute + e2e)
- [ ] Naive table shows the no-sync lie
- [ ] % of dense peak reported
- [ ] One CPU-wins row with mechanism
- [ ] Pod stopped, spend logged
