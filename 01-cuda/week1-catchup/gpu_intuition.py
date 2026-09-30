"""Week 1 Days 4 + 5 in one short run.

Part A (Day 4): when does the GPU beat the CPU, and why naive timing lies.
Part B (Day 5): why a model runs out of VRAM with compute to spare (KV cache math).
Part C (Day 5, CUDA only, optional): fill the card until OOM and compare what
          PyTorch thinks it uses with what the driver sees.

Free run on the MacBook (uses the Apple GPU via MPS):
    uv run --with torch gpu_intuition.py
On a rented L4 (about 10 minutes of pod time):
    python gpu_intuition.py --oom
"""
import argparse
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--oom", action="store_true", help="CUDA only: grow a KV cache until OOM")
args = p.parse_args()

if torch.cuda.is_available():
    dev = "cuda"
    sync = torch.cuda.synchronize
    name = torch.cuda.get_device_name()
elif torch.backends.mps.is_available():
    dev = "mps"
    sync = torch.mps.synchronize
    name = "Apple GPU (MPS)"
else:
    dev, sync, name = None, (lambda: None), "none"
print(f"accelerator: {name}\n")


def bench(fn, iters, do_sync=True):
    for _ in range(3):  # warm-up: first calls pay for context set-up and kernel selection
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if do_sync:
        sync()  # GPU work is asynchronous: without this you time the queueing, not the math
    return (time.perf_counter() - t0) / iters


# ------------------------------------------------------------------ Part A
print("PART A - matrix multiply, FP32. Times in ms. GFLOP/s = 2*N^3 / time.")
print("  compute = data already on the device   e2e = copy in + multiply + copy out")
print("  naive   = GPU compute timed WITHOUT synchronize (the classic mistake)\n")
hdr = f"{'N':>5} | {'CPU':>9} | {'GPU compute':>11} | {'GPU e2e':>9} | {'GPU naive':>9} | winner (e2e)"
print(hdr)
print("-" * len(hdr))
for n in (32, 128, 512, 1024, 2048, 4096):
    iters = max(3, int(2e9 / (2 * n**3)))  # keep each size to roughly a second
    a, b = torch.randn(n, n), torch.randn(n, n)
    cpu = bench(lambda: a @ b, iters)
    row = f"{n:>5} | {cpu * 1e3:9.3f} |"
    if dev:
        ag, bg = a.to(dev), b.to(dev)
        gpu = bench(lambda: ag @ bg, iters)
        e2e = bench(lambda: (a.to(dev) @ b.to(dev)).cpu(), iters)
        naive = bench(lambda: ag @ bg, iters, do_sync=False)
        sync()
        win = "GPU" if e2e < cpu else "CPU"
        gf = 2 * n**3 / gpu / 1e9
        row += f" {gpu * 1e3:11.3f} | {e2e * 1e3:9.3f} | {naive * 1e3:9.3f} | {win}  ({gf:,.0f} GFLOP/s on GPU)"
    print(row)

print("""
What to notice:
  * Small N: the CPU wins. Launching a GPU kernel and copying data has a fixed cost
    that a tiny multiply cannot pay back.
  * Large N: the GPU wins by a lot. Work grows as N^3 but data grows only as N^2,
    so each byte moved gets reused many times (high arithmetic intensity).
  * 'e2e' crosses over later than 'compute': the copy over PCIe is part of the real bill.
  * 'naive' can look impossibly fast at large N: you timed Python queueing work, not the GPU.
""")

# ------------------------------------------------------------------ Part B
print("PART B - where the VRAM goes when serving an LLM (FP16 = 2 bytes per value)\n")
GIB = 1024**3
models = {
    # name: (params in billions, layers, KV heads, head dim)
    "Llama-2-7B (no GQA)": (6.7, 32, 32, 128),
    "Llama-3.1-8B (GQA)": (8.0, 32, 8, 128),
}
VRAM = 24  # L4
for mname, (params, layers, kv_heads, hdim) in models.items():
    weights = params * 1e9 * 2 / GIB
    per_token = 2 * layers * kv_heads * hdim * 2  # K and V, every layer, 2 bytes each
    free = VRAM - weights - 1.5  # about 1.5 GiB for CUDA context and activations
    print(f"{mname}: weights {weights:.1f} GiB, KV cache {per_token / 1024:.0f} KiB per token")
    print(f"  on a {VRAM} GB L4 that leaves ~{free:.1f} GiB -> room for ~{int(free * GIB / per_token):,} cached tokens in total")
    for batch, ctx in ((1, 4096), (8, 4096), (16, 8192)):
        need = batch * ctx * per_token / GIB
        verdict = "fits" if need <= free else "OOM"
        print(f"    batch {batch:>2} x context {ctx:>5}: KV = {need:5.1f} GiB  -> {verdict}")
    print()
print("""What to notice:
  * The weights are a fixed cost. The KV cache grows with batch x context and is
    what actually runs you out of memory. The GPU still has compute to spare.
  * GQA (fewer KV heads) cuts the KV cache by 4x here. That is why modern models use it.
  * This is the whole reason for paged KV cache, KV quantization and batching limits
    in TensorRT-LLM (Week 5) and serving (Week 6).
""")

# ------------------------------------------------------------------ Part C
if args.oom:
    if dev != "cuda":
        print("PART C skipped: --oom needs a CUDA GPU")
    else:
        print("PART C - fill the card with 1 GiB KV-cache blocks until OOM\n")
        blocks = []
        try:
            while True:
                blocks.append(torch.empty(GIB // 2, dtype=torch.float16, device="cuda"))
        except torch.OutOfMemoryError:
            pass
        alloc = torch.cuda.memory_allocated() / GIB
        reserved = torch.cuda.memory_reserved() / GIB
        free_b, total_b = torch.cuda.mem_get_info()
        used_driver = (total_b - free_b) / GIB
        print(f"  OOM after {len(blocks)} GiB on {name} ({total_b / GIB:.1f} GiB total)")
        print(f"  PyTorch allocated: {alloc:.2f} GiB   PyTorch reserved: {reserved:.2f} GiB")
        print(f"  Driver says used:  {used_driver:.2f} GiB  (what nvidia-smi shows)")
        print(f"  Gap = {used_driver - alloc:.2f} GiB: CUDA context + caching allocator. nvidia-smi never matches PyTorch.")
        del blocks
        torch.cuda.empty_cache()
