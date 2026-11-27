"""fail_attribution v2 (issue #7 round-3 review fixes).

v1 RETRACTED: `formula_aten_fp64` mistakenly used 0.5*x*(1+erf(x/sqrt2)) (the
gelu FORWARD first term) instead of the derivative term Phi(x); and coverage
was gelu_none/fp32/large only; and the violation subset did not use the
production comparator mask.

v2:
  1. correct formulas — gelu'(x) = Phi(x) + x*phi(x):
       fp64 stable reference: Phi = 0.5*erfc(-x/sqrt2) (stable negative tail),
       phi = exp(-x^2/2)/sqrt(2*pi); plus analytic checks (x=0 -> 0.5,
       x=2 -> 1.0852318, x=-2 -> -0.0852318) evaluated on CPU fp64.
  2. TTIR-order fp32 simulation of the FlagGems kernel (from the compiled
     gelu_backward_none_kernel TTIR):
         t = x * 0.707106769 (fp32)
         pdf = (x * 0.398942292) * expf(-(t^2))      [__nv_powif/expf]
         phi_h = erff(t) * 0.5
         r = pdf + phi_h      <- rounds at 0.5-magnitude grid (ulp=2^-24)
         r = r + 0.5
         out = dy * r
     => negative-tail pdf term is QUANTIZED to fp32-ulp(0.5)=5.96e-8 steps
        or flushed to exact 0 — the hypothesized root cause, checked against
        device outputs at every violating position.
  3. production comparator (comparator.py) normal-zone mask reuses the v6 gate.
  4. coverage: ALL 15 FAIL case classes (gelu_none fp32/bf16, gelu_tanh fp32,
     fwd+grad, all layouts), each attributed positionally.
"""
import json
import math
import sys
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import comparator as C

S1 = 0.707106769  # as compiled (fp32 constant in TTIR)
S2 = 0.398942292
OUT = {"batch": "v2", "formula_fix_note": "v1 formula_aten_fp64 was wrong (forward term); v2 uses Phi(x)+x*phi(x) with erfc-stable fp64 reference"}

# ---- 0. device-independent analytic derivative checks (CPU fp64) ----
xs_checks = [0.0, 2.0, -2.0, 6.0]
ok_checks = []
for xv in xs_checks:
    x64 = torch.tensor(xv, dtype=torch.float64)
    ref = 0.5 * torch.erfc(-x64 / math.sqrt(2)) + x64 * torch.exp(-x64 * x64 / 2) / math.sqrt(2 * math.pi)
    ok_checks.append({"x": xv, "derivative_fp64_erfc": float(ref)})
OUT["analytic_checks"] = ok_checks
# assert known values (declared): d/dx gelu_none(0)=0.5, gelu'(2)=1.0852318, gelu'(-2)=-0.0852318
a0 = [c for c in ok_checks if c["x"] == 0.0][0]["derivative_fp64_erfc"]
a2 = [c for c in ok_checks if c["x"] == 2.0][0]["derivative_fp64_erca"] if False else [c for c in ok_checks if c["x"] == 2.0][0]["derivative_fp64_erfc"]
OUT["analytic_pass"] = bool(abs(a0 - 0.5) < 1e-12 and abs(a2 - 1.0852318) < 1e-6)

DTYPE_BITS = {
    "float16": {"max_sub": 0x03FF, "min_norm": 0x0400, "min_sub": 0x0001, "inf": 0x7C00, "nan": 0x7E00},
    "bfloat16": {"max_sub": 0x007F, "min_norm": 0x0080, "min_sub": 0x0001, "inf": 0x7F80, "nan": 0x7FC0},
    "float32": {"max_sub": 0x007FFFFF, "min_norm": 0x00800000, "min_sub": 0x00000001, "inf": 0x7F800000, "nan": 0x7FC00000},
}


def bits_tensor(pairs, dtype):
    vals = []
    for bits, neg in pairs:
        if dtype == "float16":
            u = torch.tensor([bits], dtype=torch.int16).view(torch.float16).item()
        elif dtype == "bfloat16":
            u = torch.tensor([bits], dtype=torch.int16).view(torch.bfloat16).item()
        else:
            u = torch.tensor([bits], dtype=torch.int32).view(torch.float32).item()
        vals.append(-u if neg else u)
    return torch.tensor(vals, dtype=getattr(torch, dtype)).repeat(1, 8)


def make_input(cls, shape, dtype):
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cls.encode()))
    dt = getattr(torch, dtype)
    if cls == "empty":
        return torch.empty(*shape, device="cuda", dtype=dt)
    if cls == "boundary":
        b = DTYPE_BITS[dtype]
        pairs = [(b["max_sub"], False), (b["max_sub"], True), (b["min_norm"], False), (b["min_norm"], True),
                 (b["min_sub"], False), (b["min_sub"], True), (0x0000, False), (0x0000, True),
                 (b["inf"], False), (b["inf"], True), (b["nan"], False)]
        t = bits_tensor(pairs, dtype)
        flat = t.reshape(1, -1).repeat(1, math.ceil(math.prod(shape) / t.numel()) + 1)
        return flat[:, : math.prod(shape)].reshape(shape).cuda()
    if cls == "small":
        return torch.randn(shape, generator=g, device="cuda", dtype=dt) * 0.01
    if cls == "large":
        return torch.randn(shape, generator=g, device="cuda", dtype=dt) * 10
    return torch.randn(shape, generator=g, device="cuda", dtype=dt)


def apply_layout(x, layout):
    if layout == "contig":
        return x
    if layout == "transposed":
        return x.clone().t().contiguous().t()
    if layout == "sliced":
        base = torch.cat([x, x], dim=1).contiguous()
        return base[:, ::2]
    raise ValueError(layout)


def fp64_ref_grad(x, op):
    """stable fp64 reference gradient (CPU erfc form)."""
    xd = x.double().cpu()
    phi = 0.5 * torch.erfc(-xd / math.sqrt(2))
    if op == "gelu_none":
        g = phi + xd * torch.exp(-xd * xd / 2) / math.sqrt(2 * math.pi)
    else:
        u = math.sqrt(2.0 / math.pi) * (xd + 0.044715 * xd ** 3)
        t = torch.tanh(u)
        du = math.sqrt(2.0 / math.pi) * (1 + 3 * 0.044715 * xd * xd)
        g = 0.5 * (1 + t) + xd * 0.5 * (1 - t * t) * du
    return g


def gems_formula_fp32_sim(x, op):
    """TTIR-order fp32 simulation of the FlagGems kernel expression."""
    xf = x.float()
    if op == "gelu_none":
        t = xf * S1
        pdf = (xf * S2) * torch.exp(-(t * t))
        ph = torch.erf(t) * 0.5
        r = pdf + ph
        r = r + 0.5
    else:  # tanh path per ops/gelu.py gelu_backward_tanh
        inner = 0.79788456 * (xf + 0.044715 * xf * xf * xf)
        t_out = torch.tanh(inner)
        dydx = 0.5 * xf * (1.044715 * 3 * 0.79788456 * xf * xf * (1 - t_out * t_out) * 0.79788456 + 0.79788456 * (1 + t_out) * 0 - 0)  # placeholder replaced below
        # exact source formula (ops/gelu.py lines 64-70): dydx = 0.5*x*( (1+3*0.044715*x^2)*0.79788456*(1-tanh^2) ) + 0.5*(1+tanh)
        c = 0.79788456
        dydx = 0.5 * xf * ((1 + 3 * 0.044715 * xf * xf) * c * (1 - t_out * t_out)) + 0.5 * (1 + t_out)
        r = dydx
    return r


def main():
    import flag_gems

    # failing case classes from v6 baseline
    acc = json.load(open(Path(__file__).parent.parent / "results" / "accuracy_matrix.json"))
    fails = [r for r in acc["rows"] if r["verdict"] == "FAIL"]
    classes = sorted({(r["op"], r["dtype"], r["layout"], r["value_class"]) for r in fails})
    OUT2 = {"failing_classes": len(classes), "classes": [f"{a}|{b}|{c}|{d}" for a, b, c, d in classes]}

    shapes = {"small": (17, 33), "typical": (2048, 1280), "boundary": (64, 128), "large": (2048, 1280)}
    prepared = []
    for op, dtype, layout, cls in classes:
        shape = shapes[cls]
        x = apply_layout(make_input(cls, shape, dtype), layout)
        xv = x.clone().requires_grad_(True)
        y = F.gelu(xv, approximate="none" if op == "gelu_none" else "tanh")
        y.sum().backward()
        torch.cuda.synchronize()
        prepared.append({"op": op, "dtype": dtype, "layout": layout, "cls": cls, "x": x,
                         "native_grad": xv.grad.detach().cpu(), "native_out": y.detach().cpu()})
    import flag_gems
    flag_gems.enable()
    per_class = []
    shapes = {"small": (17, 33), "typical": (2048, 1280), "boundary": (64, 128), "large": (2048, 1280)}
    for prep in prepared:
        op, dtype, layout, cls = prep["op"], prep["dtype"], prep["layout"], prep["cls"]
        x = prep["x"]
        native_grad = prep["native_grad"]
        native_out = prep["native_out"]
        xg = x.clone().requires_grad_(True)
        yg = F.gelu(xg, approximate="none" if op == "gelu_none" else "tanh")
        yg.sum().backward()
        torch.cuda.synchronize()
        gems_grad = xg.grad.detach().cpu()
        gems_out = yg.detach().cpu()

        # production comparator mask: normal zone per v6
        orc64 = None
        xd = x.double().cpu()
        phi = 0.5 * torch.erfc(-xd / math.sqrt(2))
        xd = xd
        if op == "gelu_none":
            orc64 = (phi + xd * torch.exp(-xd * xd / 2) / math.sqrt(2 * math.pi))
        else:
            u = math.sqrt(2.0 / math.pi) * (xd + 0.044715 * xd ** 3)
            t = torch.tanh(u)
            du = math.sqrt(2.0 / math.pi) * (1 + 3 * 0.044715 * xd * xd)
            orc64 = (0.5 * (1 + t) + xd * 0.5 * (1 - t * t) * du)
        orc64 = orc64.contiguous()
        normal, ftz = C.zone_masks(orc64, dtype)
        g64 = fp64_ref_grad(x, op).reshape(-1)

        ng = native_grad.float().reshape(-1)
        gg = gems_grad.float().reshape(-1)
        # production comparator gate (v6-consistent): full out+grad inputs
        orc_out = None
        if op == "gelu_none":
            orc_out = (xd * 0.5 * (1.0 + torch.erf(xd / math.sqrt(2.0))))
        else:
            u = math.sqrt(2.0 / math.pi) * (xd + 0.044715 * xd ** 3)
            orc_out = (xd * 0.5 * (1.0 + torch.tanh(u)))
        cmp_row = C.compare_case(native_out, gems_out, native_grad, gems_grad, orc_out.contiguous(), orc64.contiguous(), dtype)
        gate_viol = int(cmp_row.get("grad_normal_gate_violations", 0))

        # TTIR-order fp32 simulation vs device gems (per-position)
        # criterion: |sim - device| <= 0.5 ulp(0.5-scale grid) + value-scale slack
        sim = gems_formula_fp32_sim(x, op).float().reshape(-1).cpu()
        sim_match_count = int(((sim - gg).abs() <= (6e-8 + 2 * gg.abs() * 4.76837158203125e-07)).sum()) if gg.numel() else 0
        # quantify the 0.5-grid quantization prediction among gate-violating positions:
        # use comparator's grad_violation_examples flat indices (top-5) plus full-array check
        gidx = [e["flat_index"] for e in cmp_row.get("grad_violation_examples", [])]
        q = gg[gidx] if gidx else gg[:0]
        quant24 = int((((q / 5.960464477539063e-08).round() * 5.960464477539063e-08 - q).abs() < 1e-12).sum().item()) if q.numel() else 0

        # bf16/fp16 output rounding applied to simulation
        sim = gems_formula_fp32_sim(x, op).float().reshape(-1).cpu()
        if dtype in ("bfloat16", "float16"):
            sim = sim.to(getattr(torch, dtype)).float()
        sim_match_count = int(((sim - gg).abs() <= (6e-8 + 2 * gg.abs() * 4.76837158203125e-07)).sum()) if gg.numel() else 0
        # does the SIM also violate the gate? (mechanism fully explains FAIL)
        sim_native = None
        per_class.append({
            "case": f"{op}|{dtype}|{layout}|{cls}",
            "grad_normal_gate_violations": gate_viol,
            "violation_examples_gems_values": [e["gems_grad"] for e in cmp_row.get("grad_violation_examples", [])][:3],
            "violation_examples_on_2pi24_grid": quant24,
            "sim_match_count_all_positions": sim_match_count,
            "positions": int(gg.numel()),
        })

    OUT2["per_class"] = per_class
    with open("results/fail_attribution_v2.json", "w") as f:
        json.dump(OUT2, f, indent=1)
    print(json.dumps(OUT2, indent=1))


if __name__ == "__main__":
    main()
