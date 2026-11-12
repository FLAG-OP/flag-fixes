#!/usr/bin/env python3
"""GELU-family microbenchmark: native torch vs FlagGems (KunlunXIN).

Measures the operator entry points affected by this fix and validates every
FlagGems result against a reference computed with native torch *before*
FlagGems is enabled.  Output is JSON (stdout and optionally --json FILE).

Exit codes: 0 success, 2 missing accelerator/dependency (skip is not a pass),
1 benchmark or accuracy failure.
"""
import argparse
import json
import os
import platform
import sys
import traceback

import torch

DEVICE = "cuda"
SHAPE = (2990, 1024)          # BGE-M3 padded step: 2990 x hidden 1024
GATED_SHAPE = (2990, 4096)    # typical GeGLU intermediate


def bench(fn, warmup=20, iters=100, rounds=5):
    """Return best-of-`rounds` mean latency in milliseconds (CUDA events)."""
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


def errors(a, b):
    """Max abs error and max abs error relative to the reference magnitude."""
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    err = (a - b).abs().max().item()
    scale = b.abs().max().item()
    return err, err / max(scale, 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no accelerator available; not a pass", file=sys.stderr)
        return 2
    dtype = getattr(torch, args.dtype)

    torch.manual_seed(0)
    x = torch.randn(*SHAPE, device=DEVICE, dtype=dtype)
    dy = torch.randn_like(x)
    a = torch.randn(*GATED_SHAPE, device=DEVICE, dtype=dtype)
    b = torch.randn_like(a)
    # ---- native references, computed before FlagGems is enabled ----
    ref = {
        "gelu_none": torch.nn.functional.gelu(x),
        "gelu_tanh": torch.nn.functional.gelu(x, approximate="tanh"),
        "gelu_bwd_none": torch.ops.aten.gelu_backward(dy, x),
        "gelu_bwd_tanh": torch.ops.aten.gelu_backward(
            dy, x, approximate="tanh"),
        "gelu_and_mul_none": torch.nn.functional.gelu(a) * b,
        "gelu_and_mul_tanh": torch.nn.functional.gelu(
            a, approximate="tanh") * b,
    }
    native = {
        "gelu_none": bench(
            lambda: torch.nn.functional.gelu(x), iters=args.iters),
        "gelu_tanh": bench(
            lambda: torch.nn.functional.gelu(x, approximate="tanh"),
            iters=args.iters),
        "gelu_bwd_none": bench(
            lambda: torch.ops.aten.gelu_backward(dy, x), iters=args.iters),
        "gelu_bwd_tanh": bench(
            lambda: torch.ops.aten.gelu_backward(
                dy, x, approximate="tanh"), iters=args.iters),
        "gelu_and_mul_none": bench(
            lambda: torch.nn.functional.gelu(a) * b, iters=args.iters),
        "gelu_and_mul_tanh": bench(
            lambda: torch.nn.functional.gelu(a, approximate="tanh") * b,
            iters=args.iters),
    }

    import triton
    import flag_gems
    flag_gems.enable()

    out = {
        "device": torch.cuda.get_device_name(0),
        "vendor_env": os.environ.get("GEMS_VENDOR"),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "flag_gems": flag_gems.__file__,
        "dtype": args.dtype,
        "shape": list(SHAPE),
        "gated_shape": list(GATED_SHAPE),
        "iters": args.iters,
        "cases": {},
    }

    cases = {
        "gelu_fwd_none": (
            lambda: torch.nn.functional.gelu(x), "gelu_none"),
        "gelu_fwd_tanh": (
            lambda: torch.nn.functional.gelu(x, approximate="tanh"),
            "gelu_tanh"),
        "gelu_bwd_none": (
            lambda: torch.ops.aten.gelu_backward(dy, x), "gelu_bwd_none"),
        "gelu_bwd_tanh": (
            lambda: torch.ops.aten.gelu_backward(
                dy, x, approximate="tanh"), "gelu_bwd_tanh"),
        "gelu_and_mul_none": (
            lambda: flag_gems.gelu_and_mul(a, b), "gelu_and_mul_none"),
        "gelu_and_mul_tanh": (
            lambda: flag_gems.gelu_and_mul(a, b, approximate="tanh"),
            "gelu_and_mul_tanh"),
    }

    failed = False
    for name, (fn, ref_key) in cases.items():
        try:
            got = fn()
            torch.cuda.synchronize()
            err, rel = errors(got, ref[ref_key])
            atol, rtol = ((1e-4, 1e-4) if dtype == torch.float32
                          else (5e-2, 3e-2))
            if rel > rtol or err > atol + rtol * ref[ref_key].abs().max().item():
                failed = True
                print(f"ACCURACY FAIL: {name} rel={rel:.3e} err={err:.3e}",
                      file=sys.stderr)
            ms = bench(fn, iters=args.iters)
            native_ms = native[ref_key]
            out["cases"][name] = {
                "gems_ms": round(ms, 6),
                "native_ms": round(native_ms, 6),
                "speedup_vs_native": round(native_ms / ms, 3),
                "max_abs_err": err,
                "max_rel_err": rel,
            }
        except Exception as exc:  # report every broken entry point
            failed = True
            out["cases"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()

    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
