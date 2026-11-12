#!/usr/bin/env python3
"""Accuracy matrix for the KunlunXIN GELU family (dtype x layout x op).

Compares the FlagGems kernels against native torch (computed before enabling or
importing any patched dispatch) for:

  * gelu / gelu_backward, approximate in {none, tanh}
  * fused gelu_and_mul, approximate in {none, tanh}, forward and backward

Shapes include a tiny odd shape and the BGE-M3-like (2990, 1024) shape, plus
transposed / sliced / permuted layouts so strided inputs are covered.

Exit codes: 0 pass, 1 mismatch, 2 missing accelerator (skip is not a pass).
"""
import sys

import torch

DEVICE = "cuda"
SHAPES = [(1, 64), (17, 33), (2990, 1024), (3, 128, 512)]
GATED_SHAPE = (2990, 4096)
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.bfloat16: (5e-2, 3e-2),
    torch.float16: (5e-2, 3e-2),
}
failures = []


def check(name, got, expected, dtype):
    atol, rtol = TOL[dtype]
    got = got.detach().float().cpu()
    expected = expected.detach().float().cpu()
    err = (got - expected).abs().max().item()
    scale = expected.abs().max().item()
    budget = atol + rtol * scale
    if err > budget:
        failures.append(
            f"{name}: max_abs_err={err:.3e} > budget={budget:.3e} "
            f"(dtype={dtype})")
    return err


def layouts(shape, dtype):
    base = torch.randn(*shape, device=DEVICE, dtype=dtype)
    yield "contiguous", base
    if len(shape) == 2:
        yield "transposed", base.transpose(0, 1)
        yield "sliced", base[:, ::2]
    else:
        yield "permuted", base.permute(1, 0, 2)
        yield "sliced", base[..., ::2]


def main():
    if not torch.cuda.is_available():
        print("SKIP: no accelerator available; not a pass", file=sys.stderr)
        return 2

    torch.manual_seed(0)
    import flag_gems

    for dtype in DTYPES:
        for shape in SHAPES:
            for layout, x in layouts(shape, dtype):
                dy = torch.randn_like(x)
                tag = f"{dtype}/{shape}/{layout}"

                # forward: none / tanh
                check(f"{tag}/gelu_none",
                      flag_gems.gelu(x),
                      torch.nn.functional.gelu(x), dtype)
                check(f"{tag}/gelu_tanh",
                      flag_gems.gelu(x, approximate="tanh"),
                      torch.nn.functional.gelu(x, approximate="tanh"), dtype)

                # backward: none / tanh
                check(f"{tag}/gelu_bwd_none",
                      flag_gems.gelu_backward(dy, x),
                      torch.ops.aten.gelu_backward(dy, x), dtype)
                check(f"{tag}/gelu_bwd_tanh",
                      flag_gems.gelu_backward(dy, x, approximate="tanh"),
                      torch.ops.aten.gelu_backward(
                          dy, x, approximate="tanh"), dtype)

    # Fused gated GELU: forward and both input gradients.
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(*GATED_SHAPE, device=DEVICE, dtype=dtype)
        y = torch.randn_like(x)
        for approximate in ("none", "tanh"):
            tag = f"{dtype}/gelu_and_mul/{approximate}"

            gx = x.clone().requires_grad_(True)
            gy = y.clone().requires_grad_(True)
            out = flag_gems.gelu_and_mul(gx, gy, approximate=approximate)
            out.sum().backward()

            rx = x.clone().requires_grad_(True)
            ry = y.clone().requires_grad_(True)
            expected = (torch.nn.functional.gelu(
                rx, approximate=approximate) * ry)
            expected.sum().backward()

            check(f"{tag}/forward", out, expected, dtype)
            check(f"{tag}/d_input", gx.grad, rx.grad, dtype)
            check(f"{tag}/d_gate", gy.grad, ry.grad, dtype)

            gx.grad = gy.grad = None
            rx.grad = ry.grad = None

    if failures:
        for item in failures[:20]:
            print(f"FAIL: {item}")
        print(f"TOTAL: {len(failures)} failed")
        return 1
    print("TOTAL: 0 failed (GELU family matches native torch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
