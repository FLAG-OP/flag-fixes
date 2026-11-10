#!/usr/bin/env python3
"""Dispatch-level check: F.gelu / nn.GELU / autograd / gated MLP.

The unit fix lives in KunlunXIN operator files; this script verifies the paths
training code actually uses after `flag_gems.enable()`:

  * torch.nn.functional.gelu, approximate in {none, tanh}
  * torch.autograd.grad through the registered backward
  * torch.nn.GELU(approximate="tanh")
  * a gated MLP block built on flag_gems.gelu_and_mul
  * a non-contiguous input (transposed / sliced), which stresses strides

Native references are computed before FlagGems is enabled.

Exit codes: 0 pass, 1 mismatch, 2 missing accelerator (skip is not a pass).
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda"
SHAPE = (2990, 1024)
GATED_SHAPE = (2990, 4096)
DTYPE = torch.bfloat16
failures = []


def check(name, got, expected, atol=5e-2, rtol=3e-2):
    got = got.detach().float().cpu()
    expected = expected.detach().float().cpu()
    err = (got - expected).abs().max().item()
    scale = expected.abs().max().item()
    budget = atol + rtol * scale
    if err > budget:
        failures.append(
            f"{name}: max_abs_err={err:.3e} > budget={budget:.3e}")


class GatedMLP(nn.Module):
    """Minimal GeGLU-style block backed by the fixed vendor kernel."""

    def __init__(self, approximate):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        import flag_gems
        a, b = x.chunk(2, dim=-1)
        return flag_gems.gelu_and_mul(a, b, approximate=self.approximate)


class NativeGatedMLP(nn.Module):
    def __init__(self, approximate):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.gelu(a, approximate=self.approximate) * b


def main():
    if not torch.cuda.is_available():
        print("SKIP: no accelerator available; not a pass", file=sys.stderr)
        return 2

    torch.manual_seed(0)
    x = torch.randn(*SHAPE, device=DEVICE, dtype=DTYPE)
    dy = torch.randn_like(x)
    gated = torch.randn(*GATED_SHAPE, device=DEVICE, dtype=DTYPE)

    ref = {
        "gelu_none": F.gelu(x),
        "gelu_tanh": F.gelu(x, approximate="tanh"),
        "bwd_none": torch.ops.aten.gelu_backward(dy, x),
        "bwd_tanh": torch.ops.aten.gelu_backward(dy, x, approximate="tanh"),
        "gated_tanh": NativeGatedMLP("tanh")(gated),
        "gated_none": NativeGatedMLP("none")(gated),
    }
    xt = torch.randn(1024, 2990, device=DEVICE, dtype=DTYPE).transpose(0, 1)
    xs = x[:, :512]
    ref["non_contiguous"] = F.gelu(xt)
    ref["sliced"] = F.gelu(xs)
    xg = x.clone().requires_grad_(True)
    ref["autograd"] = torch.autograd.grad(
        F.gelu(xg, approximate="tanh").sum(), xg)[0]

    import flag_gems
    flag_gems.enable()

    check("F.gelu", F.gelu(x), ref["gelu_none"])
    check("F.gelu(tanh)", F.gelu(x, approximate="tanh"), ref["gelu_tanh"])
    check("nn.GELU(tanh)", nn.GELU(approximate="tanh")(x), ref["gelu_tanh"])
    check("aten.gelu_backward", torch.ops.aten.gelu_backward(dy, x),
          ref["bwd_none"])
    check("aten.gelu_backward(tanh)",
          torch.ops.aten.gelu_backward(dy, x, approximate="tanh"),
          ref["bwd_tanh"])

    xg = x.clone().requires_grad_(True)
    got = torch.autograd.grad(
        F.gelu(xg, approximate="tanh").sum(), xg)[0]
    check("autograd.grad", got, ref["autograd"])

    # Non-contiguous activation: transposed batch/seq and a last-dim slice.
    check("F.gelu(non-contiguous)", F.gelu(xt), ref["non_contiguous"])
    check("F.gelu(sliced)", F.gelu(xs), ref["sliced"])

    check("gated MLP tanh", GatedMLP("tanh")(gated), ref["gated_tanh"])
    check("gated MLP none", GatedMLP("none")(gated), ref["gated_none"])

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        print(f"TOTAL: {len(failures)} failed")
        return 1
    print("TOTAL: 0 failed (dispatch / autograd / gated MLP)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
