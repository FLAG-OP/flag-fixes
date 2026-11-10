"""Benchmark the three AdamW implementations on a BGE-M3-shaped parameter set.

Reports median step time and per-step kernel launches for stock, foreach and
fused AdamW, plus ``build_optimizer``'s choice.  Writes JSON via ``--out``.
Informational only: exit 0 unless CUDA is missing (exit 2).
"""

import argparse
import json
import sys
import time

import torch

WARMUP = 5
ITERS = 30


def bge_m3_like_params():
    g = torch.Generator(device="cuda").manual_seed(0)
    params = []
    for _ in range(24):
        shapes = (
            ([1024, 1024],) * 4 + ((4096, 1024),) + ((1024, 4096),)
            + ((1024,),) * 4 + ((4096,),) + ((1024,),) + ((1024,), (1024,))
        )
        for shape in shapes:
            params.append(torch.nn.Parameter(
                torch.randn(*shape, generator=g, device="cuda", dtype=torch.bfloat16)
                * 0.02))
    return params


def bench(opt, n=ITERS):
    for _ in range(WARMUP):
        opt.step()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        opt.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
    times.sort()
    return times[len(times) // 2]


def launches(opt, params):
    from torch.profiler import ProfilerActivity, profile

    for p in params:
        p.grad = torch.randn_like(p) * 0.01
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        opt.step()
        torch.cuda.synchronize()
    evts = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    device_ms = sum(e.self_device_time_total for e in evts) / 1000
    return {"launches": sum(e.count for e in evts),
            "distinct_kernels": len(evts),
            "device_ms": round(device_ms, 3),
            "top_kernels": sorted(
                ({"name": e.key[:80], "count": e.count,
                  "ms": round(e.self_device_time_total / 1000, 3)} for e in evts),
                key=lambda item: -item["ms"])[:8]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device; exit 2 (skip is not a pass)", file=sys.stderr)
        return 2

    from omegaconf import OmegaConf
    from flagscale.train.train_retrieval import build_optimizer

    params = bge_m3_like_params()
    for p in params:
        p.grad = torch.randn_like(p) * 0.01
    cfg = OmegaConf.create({"name": "adamw", "lr": 2e-5, "weight_decay": 0.01})

    stock = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01, fused=False)
    foreach = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01, foreach=True)
    chosen = build_optimizer(cfg, params, allow_adafactor=False)

    rows = {}
    for label, opt in (("stock", stock), ("foreach", foreach), ("build_optimizer", chosen)):
        ms = bench(opt, n=a.iters)
        info = launches(opt, params)
        rows[label] = {
            "step_ms": round(ms, 3),
            "fused_default": bool(opt.defaults.get("fused")),
            **info,
        }
        print(f"{label:16s} {ms:8.3f} ms/step  launches {info['launches']:5d}  "
              f"device {info['device_ms']:7.3f} ms  fused={rows[label]['fused_default']}")

    payload = {
        "params": len(params),
        "iters": a.iters,
        "rows": rows,
        "note": "BGE-M3-shaped 336-tensor bf16 parameter set, KunlunXIN vendor torch",
    }
    if a.out:
        with open(a.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
