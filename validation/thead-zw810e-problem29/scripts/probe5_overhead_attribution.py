"""Probe 5 v2 (review-directed overhead attribution, issue #5 round-2).

v1 correction: the size sweep must compare native vs FlagGems in SEPARATE
processes (enable() at import makes same-process "native" labels invalid).
v1 decomposition stays valid (all measured inside the gems process) and
indicates a HOST-side enqueue floor; v2 measures the native-process enqueue
for contrast.

Subprocess A (no flag_gems import): native event p50 + host enqueue per op at
  1.2M / 12.25M for gelu fp32, add fp32.
Subprocess B (flag_gems.enable()): same measurements.
Main process: linear fit (intercept/slope) per config from sweep A/B across
  0.3M..98M; direct-entry vs aten-entry kernel-set + timing verification.
"""
import json
import statistics
import subprocess
import sys
import time

import numpy as np

CHILD = r'''
import json, statistics, sys, time
import torch
import torch.nn.functional as F
mode = sys.argv[1]
if mode == "gems":
    import flag_gems
    flag_gems.enable()

def event_p50(fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s,e in ev)

def enqueue_us(fn, iters=50, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    best = None
    for _ in range(5):
        t0=time.perf_counter()
        for _ in range(iters): fn()
        dt=(time.perf_counter()-t0)/iters*1e3
        torch.cuda.synchronize()
        best = dt if best is None else min(best, dt)
    return round(best,4)

g=torch.Generator(device="cuda").manual_seed(5)
out={}
for n in (1228800, 12288000):
    x=torch.randn(n,generator=g,device="cuda",dtype=torch.float32)
    out[str(n)]={"gelu_event_ms":round(event_p50(lambda:F.gelu(x)),4),
                 "gelu_enqueue_ms":enqueue_us(lambda:F.gelu(x)),
                 "add_event_ms":round(event_p50(lambda:x+x),4),
                 "add_enqueue_ms":enqueue_us(lambda:x+x)}
print(json.dumps({"mode":mode,"measure":out}))
'''

ENV = {"PYTHONPATH": "/tmp/opencode/FlagGems-base/src", "CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin:/usr/local/bin", "PPU_SDK": "/usr/local/PPU_SDK", "HOME": "/root"}
PY = sys.executable

procs = {}
for mode in ("native", "gems"):
    p = subprocess.run([PY, "-c", CHILD, mode], env=ENV, capture_output=True, text=True, timeout=900)
    procs[mode] = json.loads(p.stdout.strip().splitlines()[-1])

OUT = {"child_measure": {m: procs[m]["measure"] for m in procs}}

# ---- size sweep in isolated processes ----
SWEEP = r'''
import json, statistics, sys
import torch
import torch.nn.functional as F
mode = sys.argv[1]
if mode == "gems":
    import flag_gems
    flag_gems.enable()
def event_p50(fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return round(statistics.median(s.elapsed_time(e) for s,e in ev),4)
g=torch.Generator(device="cuda").manual_seed(5)
res={}
for n in (307200, 1228800, 4915200, 12288000, 49152000, 98304000):
    x=torch.randn(n,generator=g,device="cuda",dtype=torch.float32)
    res[str(n)]={"gelu":event_p50(lambda:F.gelu(x)),"add":event_p50(lambda:x+x)}
print(json.dumps(res))
'''
sweep = {}
for mode in ("native", "gems"):
    p = subprocess.run([PY, "-c", SWEEP, mode], env=ENV, capture_output=True, text=True, timeout=900)
    sweep[mode] = json.loads(p.stdout.strip().splitlines()[-1])

SIZES = [307200, 1228800, 4915200, 12288000, 49152000, 98304000]
fit_block = {}
for op in ("gelu", "add"):
    ys_n = [sweep["native"][str(n)][op] for n in SIZES]
    ys_g = [sweep["gems"][str(n)][op] for n in SIZES]
    A = np.vstack([np.ones(len(SIZES)), np.array(SIZES, dtype=float)]).T

    def fit(ys):
        coef, *_ = np.linalg.lstsq(A, np.array(ys, dtype=float), rcond=None)
        return {"intercept_us": round(coef[0] * 1e3, 2), "slope_ns_per_elem": round(coef[1] * 1e6, 4)}

    fit_block[op] = {
        "rows": [{"elements": n, "native_ms": a, "gems_ms": b} for n, a, b in zip(SIZES, ys_n, ys_g)],
        "fit_native": fit(ys_n),
        "fit_gems": fit(ys_g),
        "fit_range": "0.3M..98M, 6 points, per-process isolation",
    }
OUT["size_sweep"] = fit_block

# ---- direct entry vs aten entry (gems process, same impl verification) ----
CODE3 = r'''
import json, statistics, time
import torch
import torch.nn.functional as F
import flag_gems
flag_gems.enable()
def event_p50(fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return round(statistics.median(s.elapsed_time(e) for s,e in ev),4)
def enqueue_ms(fn, iters=50, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    best=None
    for _ in range(5):
        t0=time.perf_counter()
        for _ in range(iters): fn()
        dt=(time.perf_counter()-t0)/iters*1e3
        torch.cuda.synchronize()
        best=dt if best is None else min(best,dt)
    return round(best,4)
def kernels_of(fn, iters=5):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    return sorted({e.key for e in prof.key_averages() if e.device_time > 0})
out={}
for n in (1228800, 12288000):
    x=torch.randn(n,generator=torch.Generator(device="cuda").manual_seed(5),device="cuda",dtype=torch.float32)
    ka=kernels_of(lambda:F.gelu(x)); kd=kernels_of(lambda:flag_gems.gelu(x))
    out[str(n)]={"aten_kernels":[k[:60] for k in ka],"direct_kernels":[k[:60] for k in kd],
        "same_kernel_set":bool(set(ka)==set(kd)),
        "aten_event_ms":event_p50(lambda:F.gelu(x)),"direct_event_ms":event_p50(lambda:flag_gems.gelu(x)),
        "aten_enqueue_ms":enqueue_ms(lambda:F.gelu(x)),"direct_enqueue_ms":enqueue_ms(lambda:flag_gems.gelu(x))}
print(json.dumps(out))
'''
p = subprocess.run([PY, "-c", CODE3, "gems"], env=ENV, capture_output=True, text=True, timeout=900)
OUT["direct_vs_aten"] = json.loads(p.stdout.strip().splitlines()[-1])

OUT["method_notes"] = {
    "event": "median of 30 CUDA-event timed iterations after 5 warmup (JIT excluded); on idle GPU the event window includes host enqueue latency",
    "enqueue": "min over 5 batches of (host wall to enqueue 50 calls without sync)/50; measures HOST-side per-call cost only",
    "profiler": "kernel-name evidence only; profiler inflates device_time (v1 measured 0.58ms reported vs 0.07ms event for the same 1.2M gelu) so profiler numbers are NOT used for timing claims",
    "native_vs_gems": "strict per-process isolation: native process never imports flag_gems",
}
with open("results/probe5_overhead_attribution.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1))
