"""Week 1 Day 5 - find the OOM boundary and see what ate the VRAM.

Loads a real LLM in FP16, then runs prefill (one forward pass with use_cache=True) at a fixed
context length while growing the batch until the GPU runs out of memory. At every step it records:

  weights     memory after loading the model (fixed cost)
  kv_measured memory the returned KV cache actually holds
  kv_formula  2 (K,V) x layers x kv_heads x head_dim x bytes x batch x ctx (the prediction)
  peak        torch.cuda.max_memory_allocated() during the forward (weights + KV + activations)
  reserved    what PyTorch's caching allocator holds
  driver_used what the driver (and nvidia-smi) sees: reserved + CUDA context + anything else

Run on the pod (watch live in a 2nd terminal: nvidia-smi --query-gpu=memory.used --format=csv -l 1):
    pip install transformers
    python vram_probe.py                                  # Qwen2.5-1.5B, ctx 4096
    python vram_probe.py --model Qwen/Qwen2.5-7B-Instruct --ctx 2048

Offline logic check without a GPU or a download (numbers are meaningless):
    python vram_probe.py --tiny --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch

GIB = 1024**3
RESULTS = Path(__file__).parent / "results"


def kv_bytes_per_token(cfg, dtype_bytes: int) -> int:
    heads = cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    return 2 * cfg.num_hidden_layers * kv_heads * head_dim * dtype_bytes


def mem(device: str) -> dict:
    if device != "cuda":
        return {"allocated": 0.0, "reserved": 0.0, "driver_used": 0.0, "total": 0.0}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated() / GIB,
        "reserved": torch.cuda.memory_reserved() / GIB,
        "driver_used": (total - free) / GIB,
        "total": total / GIB,
    }


def load(args):
    from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

    dtype = torch.float16
    if args.tiny:
        cfg = LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                          num_attention_heads=8, num_key_value_heads=2, vocab_size=1000)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
    else:
        cfg = AutoConfig.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    return model.to(args.device).eval(), cfg


def prefill(model, batch: int, ctx: int, device: str, vocab: int):
    """One forward pass that builds the KV cache. Returns (kv_gib, peak_gib, mem snapshot)."""
    ids = torch.randint(0, vocab, (batch, ctx), device=device)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
    with torch.no_grad():
        # The base model (no LM head): logits would be batch x ctx x vocab and would OOM first,
        # which is a real lesson too, but not today's. See the doc.
        out = model.model(input_ids=ids, use_cache=True)
    snap = mem(device)
    if device == "cuda":
        torch.cuda.synchronize()
        hidden = out.last_hidden_state.numel() * out.last_hidden_state.element_size()
        kv = (torch.cuda.memory_allocated() - before - hidden - ids.numel() * ids.element_size()) / GIB
        peak = torch.cuda.max_memory_allocated() / GIB
    else:
        kv, peak = 0.0, 0.0
    del out, ids
    return kv, peak, snap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-batch", type=int, default=4096)
    ap.add_argument("--tiny", action="store_true", help="random tiny model, no download (logic check only)")
    args = ap.parse_args()

    base = mem(args.device)
    model, cfg = load(args)
    after_load = mem(args.device)
    per_tok = kv_bytes_per_token(cfg, 2)
    gpu = torch.cuda.get_device_name() if args.device == "cuda" else args.device
    head = {
        "gpu": gpu, "vram_gib": round(after_load["total"], 2), "model": "tiny" if args.tiny else args.model,
        "ctx": args.ctx, "weights_gib": round(after_load["allocated"] - base["allocated"], 3),
        "context_overhead_gib": round(after_load["driver_used"] - after_load["reserved"], 3),
        "kv_kib_per_token": per_tok / 1024, "torch": torch.__version__,
        "layers": cfg.num_hidden_layers, "kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
    }
    print(json.dumps(head, indent=2))

    rows, ok, bad = [], 0, None

    def attempt(b: int) -> bool:
        nonlocal ok, bad
        try:
            kv, peak, snap = prefill(model, b, args.ctx, args.device, cfg.vocab_size)
        except torch.OutOfMemoryError:
            bad = b if bad is None else min(bad, b)
            rows.append({"batch": b, "status": "OOM"})
            print(f"batch {b:>5} x ctx {args.ctx}: OOM")
            if args.device == "cuda":
                torch.cuda.empty_cache()
            return False
        formula = per_tok * b * args.ctx / GIB
        rows.append({"batch": b, "status": "ok", "tokens": b * args.ctx, "kv_formula_gib": round(formula, 3),
                     "kv_measured_gib": round(kv, 3), "peak_alloc_gib": round(peak, 3),
                     "reserved_gib": round(snap["reserved"], 3), "driver_used_gib": round(snap["driver_used"], 3)})
        print(f"batch {b:>5} x ctx {args.ctx}: KV formula {formula:6.2f} GiB | measured {kv:6.2f} | "
              f"peak alloc {peak:6.2f} | reserved {snap['reserved']:6.2f} | driver {snap['driver_used']:6.2f}")
        ok = max(ok, b)
        return True

    b = 1
    while b <= args.max_batch and attempt(b):  # 1) double until the first OOM
        b *= 2
    if bad:  # 2) binary search between the last success and the first OOM
        lo, hi = ok, bad
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if attempt(mid):
                lo = mid
            else:
                hi = mid

    boundary = f"OOM at batch {bad} / context {args.ctx} on {gpu}, {head['vram_gib']} GiB (last OK: batch {ok})" if bad \
        else f"No OOM up to batch {ok} / context {args.ctx} on {gpu}"
    print("\n" + boundary)

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"vram_{'tiny' if args.tiny else args.model.split('/')[-1]}_{args.ctx}_{stamp}"
    out.with_suffix(".json").write_text(json.dumps({**head, "boundary": boundary, "last_ok_batch": ok, "oom_batch": bad}, indent=2))
    keys = ["batch", "status", "tokens", "kv_formula_gib", "kv_measured_gib", "peak_alloc_gib", "reserved_gib", "driver_used_gib"]
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["batch"]))
    print(f"wrote {out}.csv/.json")


if __name__ == "__main__":
    main()
