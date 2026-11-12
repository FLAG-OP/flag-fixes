#!/usr/bin/env python3
"""Minimal repro: KunlunXIN GELU kernels are slower than native torch.

The baseline KunlunXIN GELU implementations square with libdevice `pow(x, 2.0)`
(`tl_extra_shim.pow`).  On XPU3 that lowers to a slow libdevice call (and in
some kernels makes `xpu3-elfconv` fail outright), which dominates the GELU
forward/backward time.  The fix replaces it with `x * x`.

This script fails on the unpatched kernel and passes after the fix:

    PYTHONPATH=<FlagGems>/src python3 tests/repro_gelu_pow.py

Pass criteria are deliberately loose (2x native torch) so they are not tied to
one machine's absolute latency; the baseline misses them by a wide margin.
Exit codes: 0 pass, 1 regression, 2 missing accelerator (skip is not a pass).
"""
import sys

import torch

DEVICE = "cuda"
SHAPE = (2990, 1024)  # BGE-M3 padded step (docs/retrieval 08-kunlunxin.md)
DTYPE = torch.bfloat16
LIMIT = 2.0  # FlagGems kernel must be within 2x of the native torch kernel


def bench(fn, warmup=20, iters=100, rounds=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return min(samples)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no accelerator available; not a pass", file=sys.stderr)
        return 2

    torch.manual_seed(0)
    x = torch.randn(*SHAPE, device=DEVICE, dtype=DTYPE)
    dy = torch.randn_like(x)

    import flag_gems  # imported after the native reference is materialised

    ref_bwd = torch.ops.aten.gelu_backward(dy, x)
    ref_tanh = torch.nn.functional.gelu(x, approximate="tanh")

    got_bwd = flag_gems.gelu_backward(dy, x)
    got_tanh = flag_gems.gelu(x, approximate="tanh")
    torch.cuda.synchronize()
    err_bwd = (got_bwd.float() - ref_bwd.float()).abs().max().item()
    err_tanh = (got_tanh.float() - ref_tanh.float()).abs().max().item()

    native_bwd = bench(lambda: torch.ops.aten.gelu_backward(dy, x))
    gems_bwd = bench(lambda: flag_gems.gelu_backward(dy, x))
    native_tanh = bench(
        lambda: torch.nn.functional.gelu(x, approximate="tanh"))
    gems_tanh = bench(lambda: flag_gems.gelu(x, approximate="tanh"))

    ratio_bwd = gems_bwd / native_bwd
    ratio_tanh = gems_tanh / native_tanh

    print(f"gelu_backward      native={native_bwd:.4f} ms  "
          f"flag_gems={gems_bwd:.4f} ms  ratio={ratio_bwd:.2f}x  "
          f"max_abs_err={err_bwd:.3e}")
    print(f"gelu(tanh) forward native={native_tanh:.4f} ms  "
          f"flag_gems={gems_tanh:.4f} ms  ratio={ratio_tanh:.2f}x  "
          f"max_abs_err={err_tanh:.3e}")

    failures = []
    if ratio_bwd > LIMIT:
        failures.append(f"gelu_backward is {ratio_bwd:.2f}x native (> {LIMIT}x)")
    if ratio_tanh > LIMIT:
        failures.append(
            f"gelu(tanh) forward is {ratio_tanh:.2f}x native (> {LIMIT}x)")
    if err_bwd > 0.05 or err_tanh > 0.05:
        failures.append("output differs from native torch by more than 0.05")

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1
    print("PASS: GELU kernels are within the native-torch budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
