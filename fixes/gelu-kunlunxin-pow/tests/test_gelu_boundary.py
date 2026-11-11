"""Dump FlagGems gelu outputs for boundary inputs (base/fixed tree diff harness).

The pow->multiply patch must not change boundary semantics.  Since the erf
kernels themselves differ slightly from torch.erf at extreme tails (pre-
existing, unrelated to the patch), the meaningful check is PATCHED vs
UNPATCHED FlagGems outputs being bitwise identical.

Usage (run twice, diff the JSONs):
    PYTHONPATH=<FlagGems-base>/src  python test_gelu_boundary.py --out base.json
    PYTHONPATH=<FlagGems-fixed>/src python test_gelu_boundary.py --out fixed.json
    python - <<'EOF'  # compare
    import json
    a=json.load(open('base.json')); b=json.load(open('fixed.json'))
    assert a['outputs']==b['outputs']
    EOF

In-process sanity (always): NaN-in -> NaN-out; +0.0/-0.0 equal outputs.
"""

import argparse
import json

import torch
import torch.nn.functional as F

SPECIALS = [
    0.0, -0.0,
    float("inf"), float("-inf"), float("nan"),
    1e-300, -1e-300,
    5e-324, -5e-324,
    1e30, -1e30,
    1e-30, -1e-30,
    3.0, -3.0, 1.0, -1.0, 0.5, -0.5,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import flag_gems
    flag_gems.only_enable(include=["gelu", "gelu_backward"])

    outputs = {}
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        x = torch.tensor(SPECIALS, device="cuda", dtype=dtype)
        x.requires_grad_(True)
        dy = torch.ones_like(x)
        row = {}
        y = F.gelu(x)
        row["gelu_none_fwd"] = [t.item() for t in y.detach().float().flatten()]
        x.grad = None
        F.gelu(x).backward(dy)
        row["gelu_none_bwd"] = [t.item() for t in x.grad.float().flatten()]
        x.grad = None
        y = F.gelu(x, approximate="tanh")
        row["gelu_tanh_fwd"] = [t.item() for t in y.detach().float().flatten()]
        x.grad = None
        F.gelu(x, approximate="tanh").backward(dy)
        row["gelu_tanh_bwd"] = [t.item() for t in x.grad.float().flatten()]
        # sanity: NaN propagates, +/-0 equal
        nan_in = torch.tensor([float("nan")], device="cuda", dtype=dtype, requires_grad=True)
        nan_out = F.gelu(nan_in).detach()
        row["nan_propagates"] = bool(torch.isnan(nan_out.float()[0]))
        zz = torch.tensor([0.0, -0.0], device="cuda", dtype=dtype)
        row["pm_zero_equal"] = bool(torch.equal(F.gelu(zz)[0], F.gelu(zz)[1]))
        outputs[str(dtype)] = row
        for k, v in row.items():
            if k not in ("nan_propagates", "pm_zero_equal"):
                print(f"  {str(dtype):16s} {k:16s} {v[:6]}...")

    with open(a.out, "w") as handle:
        json.dump({"specials": SPECIALS, "outputs": outputs}, handle, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
