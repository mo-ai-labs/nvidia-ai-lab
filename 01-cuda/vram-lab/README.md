# vram-lab: find the OOM boundary (Week 1 · Day 5)

`vram_probe.py` loads an LLM in FP16, runs prefill at a fixed context, and grows the batch (doubling, then binary search)
until the GPU runs out of memory. For every batch it records the KV-cache formula vs what was actually allocated, peak
allocated, reserved, and what the driver sees. The write-up is [`docs/gpu-memory.md`](../../docs/gpu-memory.md).

## Run on the L4 pod (≈ 10 min, meter running)

```bash
pip install transformers
# terminal 2, watch it fill:
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

python vram_probe.py                                   # Qwen2.5-1.5B-Instruct, ctx 4096 (ungated, ~3 GB download)
python vram_probe.py --ctx 8192                        # optional: same model, longer context
```

Output: `results/vram_<model>_<ctx>_<stamp>.csv` and `.json`, and a boundary line ready to paste:
*"OOM at batch N / context M on NVIDIA L4, X GiB"*. Copy it into §4 of the doc.

Logic check with no GPU and no download (numbers are meaningless): `python vram_probe.py --tiny --device cpu --ctx 64 --max-batch 4`

## Notes

- It calls the **base model** (`model.model`), so no LM head. Full logits would be batch × ctx × 151,936 vocab and would OOM first.
  That's a real lesson (don't compute logits for every prompt token), but it's not today's.
- "KV measured" = allocated memory after the forward pass − before − the output hidden state. It's close to the KV cache, but not exact.
- Afterwards, **stop or destroy the pod** and log the spend in `00-environment/versions.md`.
