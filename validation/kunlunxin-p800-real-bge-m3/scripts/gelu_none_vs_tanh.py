#!/usr/bin/env python3
"""v2: GELU none/tanh microbench corrected per review round-1.

Fixes vs v1 (which was retracted):
  - shapes come from the REAL entry's call inventory (gelu_call_inventory.json),
    all at the FFN intermediate width 4096 — v1 wrongly used flattened
    (2990, 1024) (attention width, wrong dim, wrong object);
  - forward and backward are timed SEPARATELY: backward-only uses a graph
    pre-built outside the timing loop with retain_graph, and x.grad is cleared
    every iteration (v1 double-counted forward and accumulated grads);
  - tanh vs none is reported as an observation; the tanh direction is NOT
    declared closed.

Per-step estimate = sum over the real call histogram of per-step-call-count x
(fwd_ms + bwd_ms), matching how the entry calls GELU (480 fwd + 480 bwd).
"""
import json
import torch
import flag_gems

flag_gems.only_enable(include=["gelu", "gelu_backward"])

INV = json.load(open("/workspace/evidence/gelu_call_inventory.json"))
STEPS = 6  # inventory collected over 6 optimizer steps
hist = []
for u in INV["unique_shapes"]:
    shape = tuple(u["shape"])
    if len(shape) == 3 and shape[-1] == 4096:
        hist.append((shape, u["calls"] / STEPS))  # per-step fwd call count
hist.sort(key=lambda t: -t[1])
total_calls = sum(c for _, c in hist)
print(f"histogram: {len(hist)} distinct shapes, {total_calls:.1f} fwd calls/step", flush=True)


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


out = {"hist_top": [{"shape": list(s), "calls_per_step": round(c, 1)} for s, c in hist[:6]],
       "total_fwd_calls_per_step": round(total_calls, 1)}
for ap in ("none", "tanh"):
    per_shape = {}
    for shape, calls in hist:
        x = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 2).requires_grad_(True)
        w = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        fwd = bench(lambda: torch.nn.functional.gelu(x, approximate=ap))
        y = torch.nn.functional.gelu(x, approximate=ap)  # graph built OUTSIDE timing
        def bwd_only():
            y.backward(w, retain_graph=True)  # backward-only, graph reused
            x.grad = None                      # no grad accumulation
        bwd = bench(bwd_only)
        per_shape[str(list(shape))] = {"calls_per_step": round(calls, 1),
                                       "fwd_ms": round(fwd, 4), "bwd_ms": round(bwd, 4)}
    step_total = sum(v["calls_per_step"] * (v["fwd_ms"] + v["bwd_ms"]) for v in per_shape.values())
    out[ap] = {"per_shape": per_shape, "est_step_ms": round(step_total, 1)}
    print(ap, "est_step_ms:", round(step_total, 1), flush=True)

# reference: the RETRACTED v1 object ((2990,1024), flattened attention width),
# measured with the SAME corrected methodology, for transparency only
x = (torch.randn(2990, 1024, device="cuda", dtype=torch.bfloat16) * 2).requires_grad_(True)
w = torch.randn_like(x)
f = bench(lambda: torch.nn.functional.gelu(x))
y = torch.nn.functional.gelu(x)
b = bench(lambda: (y.backward(w, retain_graph=True), setattr(x, "grad", None)))
out["v1_object_reference_2990x1024"] = {"fwd_ms": round(f, 4), "bwd_ms": round(b, 4),
                                        "note": "different object (flattened, width 1024); NOT comparable to the entry's (10,L,4096) calls"}
json.dump(out, open("/workspace/evidence/gelu_none_vs_tanh_v2.json", "w"), indent=1)
print("saved /workspace/evidence/gelu_none_vs_tanh_v2.json")
