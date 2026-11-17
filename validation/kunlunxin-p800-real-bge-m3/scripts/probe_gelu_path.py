#!/usr/bin/env python3
"""Probe: which kernel does bf16 F.gelu fwd/bwd dispatch to on KunlunXIN P800?

Runs the same (2990, 1024) bf16 gelu forward+backward under torch.profiler in
two modes:
  native  - vendor aten path (expected: *_highprecision_nv kernels)
  gems    - patched FlagGems registered via flag_gems.only_enable([gelu family])
            (expected: FlagGems triton gelu kernels)

Saves a JSON with the raw kernel-name -> total us table per mode.
Profiler env (KunlunXIN): XPU_ENABLE_PROFILER_TRACING=1, XPU_CUPTI_ENABLE_DEVICE=<logical dev>.
"""
import json
import os
import sys

import torch

mode = sys.argv[1] if len(sys.argv) > 1 else "native"
out = sys.argv[2] if len(sys.argv) > 2 else f"gelu-path-{mode}.json"

if mode == "gems":
    import flag_gems
    flag_gems.only_enable(include=["gelu", "gelu_backward"])

torch.manual_seed(0)
x = (torch.randn(2990, 1024, device="cuda", dtype=torch.bfloat16) * 2).requires_grad_(True)
w = torch.randn_like(x).requires_grad_(True)

def step():
    y = torch.nn.functional.gelu(x)          # approximate='none' (BGE-M3/XLM-R default)
    y.backward(w)
    x.grad = None

# warmup (JIT compile) outside profiler
for _ in range(5):
    step()
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        step()
    torch.cuda.synchronize()

rows = {}
for ev in prof.key_averages():
    name = ev.key
    # keep device-kernel-ish rows only (kunlunxin event keys are bare kernel names;
    # vendor gelu kernels carry _highprecision_nv / xpukernel markers)
    if ev.device_time_total > 0 and ev.count > 0:
        rows[name] = {"count": ev.count, "device_us_total": round(ev.device_time_total, 1)}

json.dump({"mode": mode, "events": rows}, open(out, "w"), indent=1)
print(f"[{mode}] gelu-related kernel events:")
for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["device_us_total"]):
    kl = k.lower()
    if any(s in kl for s in ("gelu", "triton", "xpukernel")):
        print(f"  {v['device_us_total']:>10.1f} us  x{v['count']:<4} {k}")
