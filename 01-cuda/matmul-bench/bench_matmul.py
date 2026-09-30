"""Week 1 Day 4 - CPU vs GPU matmul sweep.

Times an N x N matrix multiply across sizes, dtypes and three timing modes:

  compute  operands already on the device; warm-up + synchronize (the honest number)
  e2e      host -> device copy, multiply, device -> host copy (what an API call really pays)
  naive    like compute, but WITHOUT the final synchronize (the classic benchmarking bug,
           kept on purpose so the lie shows up in your own data)

Writes results/<tag>_<device>_<stamp>.csv (one row per point) and a matching .json with the
environment (torch/CUDA versions, GPU name, SM count, compute capability, VRAM).

Examples
  uv run bench_matmul.py --device cpu --max-n 256 --min-time 0.05 --tag smoke
  uv run bench_matmul.py --device mps --dtypes fp32 fp16 --modes compute e2e --tag mac
  python bench_matmul.py --device cuda --dtypes fp32 tf32 fp16 --modes compute e2e naive --preset l4 --tag pod
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch

# Dense (no 2:4 sparsity) peak TFLOPS, used for "% of peak". The datasheet's Tensor Core figures
# carry an asterisk: "shown with sparsity, one-half lower without". These are the halved numbers.
# L4 source: https://www.nvidia.com/en-us/data-center/l4/ (checked 2026-09-27)
PRESETS: dict[str, dict[str, float]] = {
    "l4": {"fp32": 30.3, "tf32": 60.0, "fp16": 121.0, "bf16": 121.0},
}

DTYPES = {"fp32": torch.float32, "tf32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
RESULTS = Path(__file__).parent / "results"


def pick_sync(device: str):
    if device == "cuda":
        return torch.cuda.synchronize
    if device == "mps":
        return torch.mps.synchronize
    return lambda: None


def device_info(device: str) -> dict:
    info = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads": torch.get_num_threads(),
    }
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        info.update(
            torch_cuda=torch.version.cuda,
            gpu=p.name,
            compute_capability=f"{p.major}.{p.minor}",
            sm_count=p.multi_processor_count,
            vram_gib=round(p.total_memory / 1024**3, 2),
        )
    elif device == "mps":
        info.update(gpu="Apple GPU (MPS)")
    return info


def time_it(fn, sync, min_time: float, repeats: int, do_sync: bool = True) -> tuple[float, int]:
    """Median seconds per call over `repeats` runs, each at least `min_time` long."""
    for _ in range(3):  # warm-up: context init, kernel selection, allocator growth
        fn()
    sync()

    # Calibrate iterations so one repeat lasts at least min_time.
    iters = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        if time.perf_counter() - t0 >= min_time or iters >= 1 << 20:
            break
        iters *= 2

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if do_sync:
            sync()  # without this, you stop the clock while the GPU is still working
        samples.append((time.perf_counter() - t0) / iters)
        sync()  # always drain before the next repeat, even in naive mode
    return statistics.median(samples), iters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], required=True)
    ap.add_argument("--dtypes", nargs="+", default=["fp32"], choices=list(DTYPES))
    ap.add_argument("--modes", nargs="+", default=["compute"], choices=["compute", "e2e", "naive"])
    ap.add_argument("--min-n", type=int, default=16)
    ap.add_argument("--max-n", type=int, default=None, help="default: 8192 on GPU, 4096 on CPU")
    ap.add_argument("--min-time", type=float, default=0.2, help="seconds per repeat")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--preset", choices=list(PRESETS), default=None, help="peak table for %% of peak")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    if dev == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS not available")
    max_n = args.max_n or (4096 if dev == "cpu" else 8192)
    sync = pick_sync(dev)
    peaks = PRESETS.get(args.preset or "", {})

    sizes = []
    n = args.min_n
    while n <= max_n:
        sizes.append(n)
        n *= 2

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = RESULTS / f"{args.tag}_{dev}_{stamp}"
    info = device_info(dev) | {"tag": args.tag, "device": dev, "args": vars(args), "stamp": stamp}
    base.with_suffix(".json").write_text(json.dumps(info, indent=2))
    dev_name = info.get("gpu", info["cpu"])
    print(json.dumps(info, indent=2))

    rows = []
    for dname in args.dtypes:
        if dname == "tf32" and dev != "cuda":
            print(f"skip tf32 on {dev} (CUDA-only setting)")
            continue
        dtype = DTYPES[dname]
        if dev == "cuda":
            # fp32 must mean true FP32; tf32 opts into Tensor Core TF32 for FP32 matmuls.
            torch.backends.cuda.matmul.allow_tf32 = dname == "tf32"
        for n in sizes:
            a_h = torch.randn(n, n, dtype=dtype)
            b_h = torch.randn(n, n, dtype=dtype)
            a_d, b_d = a_h.to(dev), b_h.to(dev)
            flops = 2 * n**3
            for mode in args.modes:
                if dev == "cpu" and mode != "compute":
                    continue  # e2e and naive only mean something on an accelerator
                if mode == "e2e":
                    fn = lambda: (a_h.to(dev) @ b_h.to(dev)).cpu()  # noqa: E731
                else:
                    fn = lambda: a_d @ b_d  # noqa: E731
                sec, iters = time_it(fn, sync, args.min_time, args.repeats, do_sync=(mode != "naive"))
                gflops = flops / sec / 1e9
                peak = peaks.get(dname)
                pct = round(100 * gflops / (peak * 1000), 1) if peak else ""
                row = dict(tag=args.tag, device=dev, device_name=dev_name, dtype=dname, mode=mode, n=n,
                           median_s=f"{sec:.6e}", gflops=round(gflops, 2), pct_peak=pct,
                           repeats=args.repeats, iters=iters)
                rows.append(row)
                flag = "  <-- impossible: above peak" if peak and gflops > peak * 1000 else ""
                print(f"{dname:>5} {mode:>7} N={n:>5}  {sec * 1e3:10.4f} ms  {gflops:12,.1f} GFLOP/s {pct!s:>6}%{flag}")
            del a_d, b_d
            if dev == "cuda":
                torch.cuda.empty_cache()

    with base.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {base.with_suffix('.csv')} and .json")


if __name__ == "__main__":
    main()
