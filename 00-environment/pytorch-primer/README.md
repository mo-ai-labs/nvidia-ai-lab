# PyTorch & ONNX gut-check (Week 1 · Day 6)

A tiny fraud-style classifier (20 features → 64 → 32 → 1 logit, synthetic data) trained end to end, exported to ONNX
with a dynamic batch dimension, and checked against ONNX Runtime at three batch sizes.

```bash
uv run --with torch --with onnx --with onnxruntime --with onnxscript primer.py
```

No GPU needed. The MacBook CPU is ideal: ONNX export traces the graph, and CPU tensors give a clean, device-agnostic graph.

## Recorded decisions

| Item | Value | Why |
|---|---|---|
| Exporter | `torch.onnx.export(..., dynamo=True)` | The modern `torch.export`-based exporter. The legacy TorchScript path is being phased out. |
| **Opset** | **18** | Recent enough for current ONNX Runtime and TensorRT 10.x parsers, and not so new that older parsers reject it. Week 3 Day 2 will ask what TensorRT accepts. |
| Dynamic axis | `features[0]` = `batch` (via `dynamic_shapes`; `dynamic_axes` is the legacy form) | So Week 3 can build TensorRT optimization profiles (min/opt/max batch) instead of a rigid batch-8 graph |
| **Parity tolerance** | `rtol = 1e-4`, `atol = 1e-5` | Both sides are FP32, but PyTorch and ONNX Runtime may fuse and reorder ops, so the last bits differ (observed max diff ~1e-6). This is tight enough to catch a real bug (wrong op, dropout left on) and loose enough to ignore float reordering. Week 4 revisits it for FP16/INT8, where you need ~1e-2 or task-level metrics instead. |
| Export hygiene | `model.eval()` + `torch.no_grad()`, CPU example input | Dropout in train mode makes PyTorch output random, so parity would fail for a reason unrelated to ONNX |
| Single file | `external_data=False` | Keeps `model.onnx` self-contained (tiny model). Models over 2 GB *must* use external data. |

## Reference run

Run on 2026-09-30 in a cloud Linux container on CPU (torch 2.14, onnxruntime 1.25), **not on your Mac**.
Re-run it yourself and overwrite `artifacts/`.

| | |
|---|---|
| Validation accuracy | 0.850 (majority-class baseline 0.719: the model learned something) |
| Max \|torch − onnx\|, batch 1 / 7 / 256 | 9.5e-7 / 4.8e-7 / 1.4e-6: all within tolerance |
| Artifacts | `artifacts/model.pt`, `artifacts/model.onnx` (opset 18), `artifacts/parity.json` |

## Self-test: training loop from memory

Delete the five lines marked `>>> LOOP <<<` in `primer.py` and retype them without looking:
`zero_grad → forward → loss → backward → step`. If the script still passes, tick the box.
