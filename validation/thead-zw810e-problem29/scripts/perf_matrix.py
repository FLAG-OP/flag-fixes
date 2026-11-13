"""ZW810E problem29 performance matrix.

One process measures ONE config (native | gems_full | gems_selective) so
process-level isolation is preserved. Outer driver runs each config 3x.

Fixed shapes (BGE-M3 workload scale, fp32 12.25M row from PR3 attribution),
warmup then N CUDA-event timed iterations -> per-iter ms -> p50/p95 in JSON.
Memory: torch.cuda.max_memory_allocated reset around steady loop.
"""
import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, choices=["native", "gems_full", "gems_selective"])
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--out", required=True)
args = ap.parse_args()

if args.config != "native":
    import flag_gems

    if args.config == "gems_selective":
        flag_gems.only_enable(include=["gelu", "gelu_backward", "gelu_and_mul"])
    else:
        flag_gems.enable()

SH = [12288000]  # 12.25M
MM = (4096, 1280)


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in ev)
    p50 = statistics.median(ms)
    p95 = ms[int(round(0.95 * (len(ms) - 1)))]
    # percentiles are PER-PROCESS statistics of the 30 iterations;
    # cross-process aggregation (median of 3 process p50s) happens in reporting.
    return {"p50_ms": round(p50, 4), "p95_ms": round(p95, 4), "min_ms": round(ms[0], 4), "max_ms": round(ms[-1], 4), "peak_mem_bytes": torch.cuda.max_memory_allocated(), "samples_ms": [round(v, 4) for v in ms]}


def bench_set():
    g = torch.Generator(device="cuda").manual_seed(7)
    ops = {}
    errors = []

    for dt_name in ("float32", "bfloat16"):
        dt = getattr(torch, dt_name)
        x = (torch.randn(SH, generator=g, device="cuda", dtype=torch.float32) * 2).to(dt)
        ops[f"gelu_fwd_none/{dt_name}"] = lambda x=x: F.gelu(x)
        ops[f"gelu_fwd_tanh/{dt_name}"] = lambda x=x: F.gelu(x, approximate="tanh")
        xg = x.clone().requires_grad_(True)
        y = F.gelu(xg)
        gy = torch.autograd.grad(y.sum(), xg, retain_graph=False, create_graph=False)

        def gelu_fwd_bwd(xg=xg, gy=gy):
            xg.grad = None
            y2 = F.gelu(xg)
            y2.backward(gy[0])
            return None

        # NOTE: this times forward+backward together (label fixed per review round 2)
        ops[f"gelu_fwd_bwd_none/{dt_name}"] = gelu_fwd_bwd
        # round trip via fp32 (bf16->bf16 second cast would be a no-op; review round 2)
        mid = torch.float32 if dt_name == "bfloat16" else torch.bfloat16
        ops[f"cast_to/{dt_name}"] = lambda x=x, mid=mid: x.to(mid).to(dt)
        ops[f"pointwise_add/{dt_name}"] = lambda x=x: x + x
        ops[f"mul_scalar/{dt_name}"] = lambda x=x: x * 0.5

    a = torch.randn(*MM, generator=g, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(*MM, generator=g, device="cuda", dtype=torch.bfloat16)
    ops["gelu_and_mul/bf16"] = lambda: F.gelu(a) * b  # two-kernel path as modeling code writes it
    if args.config == "native":
        ops["gelu_and_mul_fused/bf16"] = lambda: F.gelu(a) * b
    else:
        import flag_gems as _fg
        ops["gelu_and_mul_fused/bf16"] = lambda: _fg.gelu_and_mul(a, b)
    x4 = torch.randn(*MM, generator=g, device="cuda", dtype=torch.float32)
    ops["findmax/fp32"] = lambda: torch.max(x4, dim=-1)
    w = torch.randn(2048, 1280, generator=g, device="cuda", dtype=torch.float32)
    ops["mm_transposed/fp32"] = lambda: x4 @ w.t()
    ops["transpose_copy/fp32"] = lambda: w.t().contiguous()
    q = torch.randn(8, 16, 512, 64, generator=g, device="cuda", dtype=torch.bfloat16)
    ops["sdpa_fwd/bf16"] = lambda: F.scaled_dot_product_attention(q, q, q)

    res = {}
    for name, fn in ops.items():
        try:
            r = timed(fn, args.iters, args.warmup)
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"}
            errors.append(name)
        res[name] = r
    res["_meta"] = {"op_errors": errors, "all_ok": not errors}
    return res


out = {"config": args.config, "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": __import__("triton").__version__, "iters": args.iters, "warmup": args.warmup, "results": bench_set()}
with open(args.out, "w") as f:
    json.dump(out, f, indent=1)
ok = out["results"]["_meta"]["all_ok"]
print("op_errors:", out["results"]["_meta"]["op_errors"], flush=True)
import sys
sys.exit(0 if ok else 1)
