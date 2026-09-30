# GPU Environment — pinned

## Rental strategy

Provider(s): **RunPod**
Default tier: **L4 24GB @ ~$0.49/hr** (also covers FP8: Ada supports it)
Burst tier (VRAM only): **A100 80GB @ ~$1.59/hr**, only when a lab needs > 24 GB (e.g. an 8B model in FP16 plus a long-context KV cache). **Not** for FP8: A100 (Ampere) has no FP8 Tensor Cores. If a lab needs FP8 *and* > 24 GB, use H100.
Free tiers in use: **build.nvidia.com**
Teardown rule: **stop if <24h, destroy if longer**

## Hardware (of the rented box)

GPU: **NVIDIA L4 24GB**
Architecture: **Ada Lovelace**
Compute capability: **8.9**
VRAM: **24 GB** GPU count: **1**
Precisions unlocked: FP16/BF16 [x] TF32 [x] INT8 [x] FP8 [x] FP4 [ ] (FP4 needs Blackwell)
Source: [L4 product page](https://www.nvidia.com/en-us/data-center/l4/), which lists FP8 Tensor Core throughput (checked 2026-09-27)

## Stack (do not change without a note below)

⏳ Fill on the next pod session (Week 1 Day 4/5 runs). The sweep's `results/pod_cuda_*.json` records torch + CUDA:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda)"
cat /etc/os-release | head -2
```

Host image / OS: …
NVIDIA driver: …
CUDA (toolkit/runtime): …
torch / torch.version.cuda: … / …

## Rebuild recipe

1. Rent L4 on RunPod (or Vast.ai)
2. SSH into the pod
3. Verify GPU is visible: `nvidia-smi`

## Spend log

| Date  | Tier    | Hours | $     | What it bought |
| ----- | ------- | ----- | ----- | -------------- |
| 09-21 | Default | 21    | $0.30 | Initial Setup  |

> ⚠️ Check the 09-21 row: 21 h at $0.49/hr would be ~$10, and $0.30 is ~37 min. Probably 21 *minutes* (0.35 h)?

## Change log

| Date  | What changed   | Why | Benchmarks invalidated? |
| ----- | -------------- | --- | ----------------------- |
| 09-21 | initial update |     |                         |
| 09-30 | Burst tier: FP8 no longer needs a burst card; A100 kept for VRAM only. Added architecture + precisions. | L4 (Ada, CC 8.9) supports FP8; A100 (Ampere) does not. The old plan note had this backwards. | No |
