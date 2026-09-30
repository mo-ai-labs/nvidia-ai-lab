# matmul-bench: CPU vs GPU crossover (Week 1 · Day 4)

Times an N×N matrix multiply from N = 16 up to 8192 (4096 on CPU) and writes one CSV row per point.
The write-up lives in [`docs/cpu-vs-gpu.md`](../../docs/cpu-vs-gpu.md).

| File | What it does |
|---|---|
| `bench_matmul.py` | The sweep. Devices `cpu` / `cuda` / `mps`; dtypes `fp32` `tf32` `fp16` `bf16`; modes `compute` · `e2e` · `naive`. Warm-up and synchronize built in. |
| `report.py` | CSVs → `results/summary.md` (tables, crossover, naive inflation) + `results/gflops.png`, `results/latency.png` |
| `results/` | Raw `*.csv` + environment `*.json`, one pair per run (`<tag>_<device>_<timestamp>`) |

## Timing modes

| Mode | Measures | Why it's here |
|---|---|---|
| `compute` | Operands already on the device. Warm-up, then `synchronize()` before stopping the clock | The honest kernel number |
| `e2e` | Copy host → device, multiply, copy back | What a real API call pays, including the PCIe bill |
| `naive` | Same as `compute` but **no final `synchronize()`** | Kept on purpose: shows how asynchronous launches make timing lie |

Each point: 3 warm-up calls → iterations calibrated so one repeat lasts ≥ `--min-time` (0.2 s) → 5 repeats → median.
`fp32` on CUDA forces true FP32 (`allow_tf32 = False`); `tf32` switches Tensor Core TF32 on.

## Run it

```bash
uv sync

# 1. Smoke test on the laptop (then delete the output)
uv run bench_matmul.py --device cpu --max-n 256 --min-time 0.05 --tag smoke

# 2. Free third curve on the MacBook (optional)
uv run bench_matmul.py --device cpu --max-n 4096 --tag mac
uv run bench_matmul.py --device mps --dtypes fp32 fp16 --modes compute e2e --tag mac

# 3. On the L4 pod (meter running: ≤ 30 min, hard time-box)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,power.limit --format=csv
python bench_matmul.py --device cuda --dtypes fp32 tf32 fp16 --modes compute e2e naive --preset l4 --tag pod
#    second terminal while N = 4096/8192 run:  nvidia-smi dmon -s pucm   (note power W and SM clock)
python bench_matmul.py --device cpu --dtypes fp32 --max-n 4096 --tag pod     # same-host CPU baseline

# 4. Back on the laptop
scp -P <PORT> "root@<POD_IP>:/workspace/results/*" ./results/
uv run report.py --latest
```

Then copy `results/gflops.png` → `docs/img/cpu-vs-gpu-gflops.png`, `results/latency.png` → `docs/img/cpu-vs-gpu-latency.png`,
and paste the tables from `results/summary.md` into sections 4 and 5 of the doc.

## Peaks used for "% of peak"

`PRESETS["l4"]` holds the **dense** L4 figures: FP32 30.3 · TF32 60 · FP16/BF16 121 TFLOPS.
The datasheet's Tensor Core numbers are quoted *with 2:4 sparsity*; dense is half
([L4 product page](https://www.nvidia.com/en-us/data-center/l4/), checked 2026-09-27).
A dense matmul can never beat the dense peak. A `naive` row that does is the no-sync bug, and the script flags it.
