"""Reproduce the per-parameter AdamW decomposition on the KunlunXIN vendor torch.

Exit 0 requires ``flagscale.train.train_retrieval.build_optimizer`` to hand out
a fused AdamW (``defaults['fused']`` truthy) whose step is clearly cheaper than
a stock AdamW step on the same 336-tensor BGE-M3-shaped parameter set.
Baseline FlagScale returns stock AdamW, so the script exits 1 there.

Exit code 2 means "cannot run here" (no CUDA device): a skip is not a pass.
"""

import sys
import time

import torch

WARMUP = 5
ITERS = 20


def bge_m3_like_params(device="cuda", dtype=torch.bfloat16):
    """336 tensors: the transformer-block part of XLM-R large (no embedding table)."""
    g = torch.Generator(device=device).manual_seed(0)
    params = []
    for _ in range(24):
        shapes = (
            ([1024, 1024],) * 4          # q, k, v, o
            + ((4096, 1024),)            # ff in
            + ((1024, 4096),)            # ff out
            + ((1024,),) * 4             # q/k/v/o bias
            + ((4096,),)                 # ff1 bias
            + ((1024,),)                 # ff2 bias
            + ((1024,), (1024,))         # two layer norms (weight, bias)
        )
        for shape in shapes:
            params.append(torch.nn.Parameter(
                torch.randn(*shape, generator=g, device=device, dtype=dtype) * 0.02
            ))
    return params


def bench_step(opt, n=ITERS, warmup=WARMUP):
    for _ in range(warmup):
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


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device; exit 2 (skip is not a pass)", file=sys.stderr)
        return 2

    from omegaconf import OmegaConf
    from flagscale.train.train_retrieval import build_optimizer

    params = bge_m3_like_params()
    for p in params:
        p.grad = torch.randn_like(p) * 0.01

    cfg = OmegaConf.create({"name": "adamw", "lr": 2e-5, "weight_decay": 0.01})
    chosen = build_optimizer(cfg, params, allow_adafactor=False)
    fused = bool(chosen.defaults.get("fused"))
    t_chosen = bench_step(chosen)

    stock = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01, fused=False)
    t_stock = bench_step(stock)

    print(f"params: {len(params)} tensors")
    print(f"build_optimizer fused flag : {fused}")
    print(f"build_optimizer step       : {t_chosen:.3f} ms")
    print(f"stock AdamW step           : {t_stock:.3f} ms")
    print(f"ratio chosen/stock         : {t_chosen / t_stock:.3f}")

    if not fused:
        print(
            "FAIL: build_optimizer did not select fused AdamW "
            "(baseline behaviour; vendor torch decomposes AdamW into "
            f"per-parameter kernels, {t_stock:.1f} ms/step)",
            file=sys.stderr,
        )
        return 1
    if t_chosen > 0.75 * t_stock:
        print(
            f"FAIL: fused step {t_chosen:.3f} ms is not faster than stock "
            f"{t_stock:.3f} ms (expected < 0.75x)",
            file=sys.stderr,
        )
        return 1
    print("PASS: fused AdamW selected and faster than stock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
