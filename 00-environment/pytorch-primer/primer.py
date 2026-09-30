"""Week 1 Day 6 - PyTorch & ONNX gut-check.

Trains a tiny fraud-style classifier on synthetic data, exports it to ONNX with a
dynamic batch dimension, and checks that ONNX Runtime gives the same outputs as PyTorch.

Run (Mac or Windows, no GPU needed):
    uv run --with torch --with onnx --with onnxruntime --with onnxscript primer.py

Self-test for "wrote a training loop from memory": delete the 5 lines marked
>>> LOOP <<< below, then retype them without looking.
"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

OUT = Path(__file__).parent / "artifacts"
OUT.mkdir(exist_ok=True)
torch.manual_seed(0)

# Parity tolerance, chosen on purpose: FP32 on both sides, but PyTorch and ONNX Runtime
# may fuse and reorder ops differently, so the last few bits can differ. 1e-4 relative
# is tight enough to catch a real bug (a wrong op or a dropout left on) and loose enough
# to ignore float reordering. Week 4 revisits this when precision drops to FP16/INT8.
RTOL, ATOL = 1e-4, 1e-5
OPSET = 18

# ---------------------------------------------------------------- data
# 20 transaction features. A hidden (partly non-linear) rule makes ~28% of rows "fraud".
N, F = 20_000, 20
X = torch.randn(N, F)
logit = 2.5 * X[:, 0] - 1.5 * X[:, 1] + X[:, 2] * X[:, 3] - 2.0
y = (torch.rand(N) < torch.sigmoid(logit)).float().unsqueeze(1)
split = int(0.8 * N)
fraud_rate = y.mean().item()
print(f"fraud rate {fraud_rate:.1%} -> always guessing 'not fraud' scores {1 - fraud_rate:.1%} accuracy")
Xtr, ytr, Xva, yva = X[:split], y[:split], X[split:], y[split:]

# ---------------------------------------------------------------- model
model = nn.Sequential(
    nn.Linear(F, 64), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1),  # raw logit; the loss applies the sigmoid
)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------- train
BATCH, EPOCHS = 256, 5
t0 = time.perf_counter()
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(split)
    for i in range(0, split, BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = Xtr[idx], ytr[idx]
        opt.zero_grad()                 # >>> LOOP <<<
        pred = model(xb)                # >>> LOOP <<<
        loss = loss_fn(pred, yb)        # >>> LOOP <<<
        loss.backward()                 # >>> LOOP <<<
        opt.step()                      # >>> LOOP <<<
    model.eval()
    with torch.no_grad():
        va_logits = model(Xva)
        va_loss = loss_fn(va_logits, yva).item()
        acc = ((va_logits > 0).float() == yva).float().mean().item()
    print(f"epoch {epoch + 1}: train_loss={loss.item():.4f} val_loss={va_loss:.4f} val_acc={acc:.3f}")
print(f"training took {time.perf_counter() - t0:.1f}s on CPU")

torch.save(model.state_dict(), OUT / "model.pt")

# ---------------------------------------------------------------- export
# eval() turns dropout off. Without it, PyTorch output is random and parity fails
# for a reason that has nothing to do with ONNX.
model.eval()
example = torch.randn(8, F)  # CPU tensor -> device-agnostic graph
onnx_path = OUT / "model.onnx"
with torch.no_grad():
    torch.onnx.export(
        model, (example,), str(onnx_path),
        input_names=["features"], output_names=["logit"],
        # dynamo=True is the modern exporter; it takes dynamic_shapes (dynamic_axes is the legacy form).
        # Batch is dynamic so Week 3's TensorRT work can build optimization profiles over it.
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        opset_version=OPSET, dynamo=True, external_data=False,  # one self-contained .onnx file
    )
print(f"exported {onnx_path.name} (opset {OPSET})")

# ---------------------------------------------------------------- parity
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
results = {}
for bs in (1, 7, 256):  # different batch sizes prove the batch axis is really dynamic
    x = torch.randn(bs, F)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"features": x.numpy()})[0]
    diff = float(np.max(np.abs(ref - got)))
    np.testing.assert_allclose(ref, got, rtol=RTOL, atol=ATOL)
    results[bs] = diff
    print(f"batch {bs:>3}: max |torch - onnx| = {diff:.2e}  OK")

summary = {
    "torch": torch.__version__, "onnxruntime": ort.__version__, "opset": OPSET,
    "rtol": RTOL, "atol": ATOL, "max_abs_diff_by_batch": results,
    "val_acc": round(acc, 4), "majority_baseline_acc": round(1 - fraud_rate, 4), "dynamic_axis": "batch",
}
(OUT / "parity.json").write_text(json.dumps(summary, indent=2))
print("PASS - wrote artifacts/parity.json")
