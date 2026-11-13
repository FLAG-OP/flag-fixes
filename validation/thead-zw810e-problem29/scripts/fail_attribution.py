"""Issue #7 §D: positional attribution of the 15 hard-FAIL cases.

For gelu_none fp32 'large' (the worst grad-violation class): rerun the case,
capture violating positions, and evaluate BOTH implementations' formulas in
fp64 (from the same fp32 inputs):
  formula_gems(x) = scale2*x*exp(-(scale1*x)^2) + 0.5*erf(scale1*x) + 0.5
      scale1=0.7071067811, scale2=0.3989422803  (exact source constants)
  formula_aten(x) = 0.5*x*(1+erf(x*scale1)) + x*0.3989422804014327*exp(-0.5*x*x)
If gems-device ≈ formula_gems_fp64 (within fp32 ulp) and native-device ≈
formula_aten_fp64, the deviation is a finite-precision cancellation artifact
in the Phi(x)=0.5*(1+erf) term at |x|>~3.9 — NOT an algorithm/implementation
error. Report the |x| distribution of violating positions and absolute error
magnitudes.
"""
import json
import math

import torch
import torch.nn.functional as F

OUT = {}

# rebuild the exact case (deterministic; same seeds as accuracy_matrix v6)
import zlib

g = torch.Generator(device="cuda").manual_seed(zlib.crc32("large".encode()))
x = torch.randn(2048, 1280, generator=g, device="cuda", dtype=torch.float32) * 10

import flag_gems

# native phase
xv = x.clone().requires_grad_(True)
yn = F.gelu(xv)
yn.sum().backward()
native_grad = xv.grad.detach().clone()

flag_gems.enable()
xg = x.clone().requires_grad_(True)
yg = F.gelu(xg)
yg.sum().backward()
gems_grad = xg.grad.detach().clone()
torch.cuda.synchronize()

# fp64 oracle on device
oracle = None
xd = x.double()
oracle = (xd * 0.5 * (1.0 + torch.erf(xd / math.sqrt(2.0)) + 0.0))  # forward not needed
phi = 0.5 * (1.0 + torch.erf(xd / math.sqrt(2.0)))
pdf = torch.exp(-0.5 * xd * xd) / math.sqrt(2.0 * math.pi)
oracle_grad = (phi + xd * pdf).cpu()

n_c, g_c = native_grad.cpu(), gems_grad.cpu()
orc = oracle_grad
diff = (n_c.float() - g_c.float()).abs()
viol_mask = (diff > 0) & torch.isfinite(diff)
positions = torch.nonzero(viol_mask.reshape(-1), as_tuple=False).reshape(-1)
OUT["total_positions"] = int(x.numel())
OUT["bitwise_diff_positions"] = int(viol_mask.sum())

# gate-violation subset (per-position rel > max(4eps,1.5*native_rel)) — recompute
fin = torch.isfinite(orc)
denom = orc.abs().clamp_min(1e-300)
rel_n = ((n_c.double() - orc).abs() / denom).reshape(-1)
rel_g = ((g_c.double() - orc).abs() / denom).reshape(-1)
viol_gate = rel_g > torch.maximum(torch.full_like(rel_n, 4 * 1.1920929e-07), 1.5 * rel_n)
OUT["gate_violation_positions"] = int(viol_gate.sum())

# fp64 formula simulation at violating positions
vi = viol_gate.nonzero().reshape(-1)
sel = vi[torch.topk(rel_g[vi], k=min(8, vi.numel())).indices] if vi.numel() else vi[:0]
xs = x.double().cpu().reshape(-1)[sel]
scale1 = 0.7071067811
scale2 = 0.3989422803
fg = scale2 * xs * torch.exp(-(scale1 * xs) ** 2) + 0.5 * torch.erf(scale1 * xs) + 0.5
fa = 0.5 * xs * (1.0 + torch.erf(xs * (1 / math.sqrt(2)))) + xs * 0.3989422804014327 * torch.exp(-0.5 * xs * xs)
examples = []
for j in range(sel.numel()):
    i = int(sel[j])
    examples.append({
        "flat_index": i,
        "x": float(x.reshape(-1)[i]),
        "native_grad": float(n_c.reshape(-1)[i]),
        "gems_grad": float(g_c.reshape(-1)[i]),
        "oracle_grad": float(orc.reshape(-1)[i]),
        "formula_gems_fp64": float(fg[j]),
        "formula_aten_fp64": float(fa[j]),
        "gems_vs_formula_gems": float(abs(float(g_c.reshape(-1)[i]) - float(fg[j]))),
        "native_vs_formula_aten": float(abs(float(n_c.reshape(-1)[i]) - float(fa[j]))),
    })
OUT["examples"] = examples

# |x| distribution of violating positions
if vi.numel():
    import numpy as np
    xs_all = x.reshape(-1)[vi].abs().cpu().double().numpy()
    OUT["violation_absx_p50"] = float(np.median(xs_all))
    OUT["violation_absx_min"] = float(xs_all.min())
    OUT["violation_absx_max"] = float(xs_all.max())
    OUT["violation_abs_frac_over_3p9"] = float((xs_all > 3.9).mean())
# absolute error magnitude at violations
if vi.numel():
    OUT["violation_max_abs_err_native_vs_oracle"] = float((n_c.double() - orc).abs().reshape(-1)[vi].max())
    OUT["violation_max_abs_err_gems_vs_oracle"] = float((g_c.double() - orc).abs().reshape(-1)[vi].max())
    OUT["violation_max_abs_err_gems_vs_native"] = float(diff.reshape(-1)[vi].max())

with open("results/fail_attribution_gelu_bwd.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1))
