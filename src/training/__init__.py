# FILE MAP | Training package: the canonical runner's guts.
#   engine.py    - torch training loop (loss, model dispatch, timing, best-only checkpoint)
#   reporting.py - I/O layer: metrics.json, mask overlays, run.log, Drive mirror
# Imported lazily by train.py so `--dry-run` and the pure config/metric tests never need torch.
