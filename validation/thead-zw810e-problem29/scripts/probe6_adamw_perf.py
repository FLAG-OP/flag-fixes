"""Probe 6 (review follow-up): AdamW stock vs fused on ZW810E —
device step time, kernel-launch count (profiler CUDA kernel instances),
peak memory, for (a) single [1024,1024] param, (b) 336-tensor BGE-M3-like
config (PR3 attribution口径). Correctness trajectories already archived in
results/adamw_check.json; this probe adds performance/launch evidence.
"""
import json
import statistics

import torch

OUT = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0)}


def build_params(kind):
    g = torch.Generator(device="cuda").manual_seed(9)
    if kind == "single":
        shapes = [(1024, 1024)]
    else:  # bge336: 336 tensors — 112 x [4096,1280] w, 112 x [1280] b, 112 x [1280,512]
        shapes = [(4096, 1280)] * 112 + [(1280,)] * 112 + [(1280, 512)] * 112
    return [torch.nn.Parameter(torch.randn(s, generator=g, device="cuda", dtype=torch.float32) * 0.02) for s in shapes]


def run(kind, opt_kind, steps=5):
    params = build_params(kind)
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01, fused=(opt_kind == "fused"))
    grads = [torch.randn_like(p) for p in params]
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        for p, gr in zip(params, grads):
            p.grad = gr
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        opt.step()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    # launch count: profile one step
    for p, gr in zip(params, grads):
        p.grad = gr
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        opt.step()
        torch.cuda.synchronize()
    evs = [ev for ev in prof.key_averages() if ev.device_time > 0]
    launches = sum(e.count for e in evs)
    return {"step_ms_median": round(statistics.median(times[1:]), 3), "step_ms_first": round(times[0], 3), "peak_mem_bytes": peak, "cuda_kernel_launches_per_step": launches}


for kind in ("single", "bge336"):
    OUT[kind] = {}
    for opt_kind in ("stock", "fused"):
        try:
            OUT[kind][opt_kind] = run(kind, opt_kind)
        except Exception as e:
            OUT[kind][opt_kind] = {"error": f"{type(e).__name__}: {e}"}

with open("results/probe6_adamw_perf.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1))
