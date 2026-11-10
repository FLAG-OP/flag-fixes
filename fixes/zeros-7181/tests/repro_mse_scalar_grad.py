# Minimal reproduction for the round-4 finding: FlagGems mse_loss_backward
# does not broadcast a scalar (0-dim) grad_output. Autograd passes a 0-dim
# seed gradient for scalar losses, so under FULL flag_gems.enable() every
# element except index 0 of the gradient is corrupted (out-of-bounds load of
# the 1-element tensor). Native aten broadcasts correctly.
#
# The file ops/mse_loss_backward.py is identical in the pre-fix tree, so this
# is a pre-existing FlagGems bug outside the zeros-7181 fix scope (the 8
# fix targets do not include mse_loss_backward).
#
# Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<tree>/src python3 tests/repro_mse_scalar_grad.py
import torch

import flag_gems  # noqa: F401  (import alone does not register anything)

flag_gems.only_enable(include=["mse_loss_backward"])

torch.manual_seed(0)
x = torch.randn(8, 5, device="cuda")
t = torch.randn(8, 5, device="cuda")

# case 1: full-tensor grad_output -> exact
g = torch.randn(8, 5, device="cuda")
ref = 2 * (x - t) * g / x.numel()
fg = torch.ops.aten.mse_loss_backward(g, x, t, 1)
print(f"full-tensor grad_output, mean: max_abs={(fg - ref).abs().max():.3e}")

# case 2: scalar 0-dim grad_output (what autograd passes) -> corrupted
gs = torch.tensor(1.0, device="cuda")
refs = 2 * (x - t) * gs / x.numel()
fgs = torch.ops.aten.mse_loss_backward(gs, x, t, 1)
bad = int((fgs - refs).abs().gt(1e-4).sum())
print(f"scalar grad_output, mean: mismatched={bad}/{x.numel()}")
print("elem0 fg/ref:", fgs.flatten()[0].item(), refs.flatten()[0].item())
print("elem1 fg/ref:", fgs.flatten()[1].item(), refs.flatten()[1].item())

# reference check with reduction=none / sum stays exact
fg0 = torch.ops.aten.mse_loss_backward(g, x, t, 0)
print("reduction=0 max_abs:", (fg0 - 2 * (x - t) * g).abs().max().item())
fg2 = torch.ops.aten.mse_loss_backward(g, x, t, 2)
print("reduction=2 max_abs:", (fg2 - 2 * (x - t) * g).abs().max().item())
