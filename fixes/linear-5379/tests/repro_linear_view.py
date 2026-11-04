"""Issue #5379 minimal repro: non-contiguous 3-D input through flag_gems linear.

Pre-fix  : RuntimeError "view size is not compatible with input tensor's size and
           stride ..." raised at src/flag_gems/ops/linear.py (non-zero exit).
Post-fix : output matches native torch, prints OK (exit 0).

Run:  PYTHONPATH=<FlagGems>/src python repro_linear_view.py
No CUDA device -> exit 2; a skip is never a pass.
"""
import sys

import torch

if not torch.cuda.is_available():
    print("SKIP: no CUDA device; a skip is not a pass", file=sys.stderr)
    raise SystemExit(2)

device = "cuda"

# The reported layout: attention-style transpose, innermost dim is not stride-1.
x = torch.randn(2, 64, 16, device=device).transpose(1, 2)  # -> (2, 16, 64)
w = torch.randn(32, 64, device=device)
b = torch.randn(32, device=device)
print(f"input {tuple(x.shape)} stride={x.stride()} contiguous={x.is_contiguous()}")
assert not x.is_contiguous(), "repro input must be non-contiguous"

# Native oracle, computed before FlagGems registers anything.
ref = torch.nn.functional.linear(x, w, b).cpu()

import flag_gems  # noqa: E402

print("flag_gems from:", flag_gems.__file__)
flag_gems.enable()

try:
    out = torch.nn.functional.linear(x, w, b)
except RuntimeError as error:
    print("ISSUE #5379 REPRODUCED:", error, file=sys.stderr)
    raise

out = out.cpu()
mse = ((out - ref) ** 2).mean().item()
print(f"out {tuple(out.shape)} mse={mse:.3e} max_diff={(out - ref).abs().max().item():.3e}")
assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), "output diverges from native torch"
print("OK: flag_gems linear accepts non-contiguous input and matches native torch")
