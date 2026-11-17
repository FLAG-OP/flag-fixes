#!/usr/bin/env python3
"""GELU path candidates at REAL-shape distribution (issue #9 B).

Compares, on the patched FlagGems tree, per-call cost of:
  none (erf, the training config's approximate)
  tanh (candidate)
over the real batch shape distribution (10 rows x per-batch padded length,
measured from the real loader: p50/p90/max groups), vs the vendor fp32
highprecision path measured from the native profile (1,169.7 ms/step total).
"""
import json
import torch

import flag_gems
flag_gems.only_enable(include=["gelu", "gelu_backward"])
H = 1024
ROWS = 10
# real padded-length groups (tokens per batch), from the rebuilt loader probe
GROUPS = [140, 360, 480, 512]  # ~p25/p50/p90/max(capped by max_length=512)


def bench(fn, warmup=20, iters=100, rounds=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


out = {}
for ap in ("none", "tanh"):
    per_group = {}
    for L in GROUPS:
        x = (torch.randn(ROWS * L, H, device="cuda", dtype=torch.bfloat16) * 2).requires_grad_(True)
        w = torch.randn_like(x)
        fwd = bench(lambda: torch.nn.functional.gelu(x, approximate=ap))
        bwd = bench(lambda: torch.nn.functional.gelu(x, approximate=ap).backward(w))
        per_group[L] = {"fwd_ms": round(fwd, 4), "bwd_ms": round(bwd, 4)}
    out[ap] = per_group

# estimate per-step totals over the real distribution:
# 480 gelu fwd + 480 bwd calls/step; padded-length mix ~ p25:20%, p50:40%, p90:25%, max:15%
mix = {140: 0.20, 360: 0.40, 480: 0.25, 512: 0.15}
for ap in out:
    total = sum(out[ap][L]["fwd_ms"] * 480 * w + out[ap][L]["bwd_ms"] * 480 * w for L, w in mix.items())
    out[ap]["est_step_ms_mixed"] = round(total, 1)
# vendor reference from the clean native profile
out["vendor_highprecision_step_ms"] = {"fwd": 426.5, "bwd": 743.2, "total": 1169.7}
json.dump(out, open("gelu_none_vs_tanh_realshapes.json", "w"), indent=1)
print(json.dumps(out, indent=1))
