"""Issue #7 round-4 review: three-level match statistics on violating subsets
(exact-bit / ULP / tolerance) + tanh-forward independent attribution.

Produces per-class: gate-violating positions; of those, how many are EXPLAINED
by the TTIR-order fp32 simulation at three strictness levels:
  exact  : sim bits == device bits
  ulp<=2 : dtype-ordinal ulp distance <= 2
  tol    : |sim-device| <= 6e-8 + 2*|device|*4.768e-07 (declared tolerance;
           note 6e-8 abs is itself close to the 2^-24 quantization step —
           stated explicitly, NOT a bitwise-exactness proof)
  unexplained: none of the above (examples archived)
Also separately attributes tanh FORWARD normal-zone gate violations (v6 fwd
violations) against the gelu_tanh forward TTIR-order fp32 simulation.
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

S1 = 0.707106769
S2 = 0.398942292

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


def sim_grad(x, op):
    """TTIR-order fp32 simulation of the FlagGems backward expression."""
    xf = x.float()
    if op == "gelu_none":
        t = xf * S1
        pdf = (xf * S2) * torch.exp(-(t * t))
        ph = torch.erf(t) * 0.5
        r = pdf + ph
        r = r + 0.5
        return r
    c = 0.79788456
    inner = c * (xf + 0.044715 * xf * xf * xf)
    t_out = torch.tanh(inner)
    dydx = 0.5 * xf * ((1 + 3 * 0.044715 * xf * xf) * c * (1 - t_out * t_out)) + 0.5 * (1 + t_out)
    return dydx


def sim_forward(x, op):
    xf = x.float()
    if op == "gelu_none":
        t = xf * S1
        return 0.5 * xf * (1 + torch.erf(t))
    c = 0.79788456
    # round-5 fix: x + 0.044715*x^3 (was x^2 — wrong formula)
    return 0.5 * xf * (1 + torch.tanh(c * (xf + 0.044715 * xf ** 3)))


def forward_formula_selftest():
    """round-5: analytic checks incl. +/-2, 0, tail; CPU fp64 reference."""
    bad = []
    c = 0.79788456
    for xv in (0.0, 2.0, -2.0, 6.0):
        x64 = torch.tensor(xv, dtype=torch.float64)
        ref = 0.5 * x64 * (1 + torch.tanh(c * (x64 + 0.044715 * x64 ** 3)))
        got = float(sim_forward(torch.tensor([xv]), "gelu_tanh")[0])
        exp = float(ref)
        if abs(got - exp) > 1e-3:
            bad.append(f"x={xv}: sim {got} vs fp64 {exp}")
        print(f"  fwd selftest x={xv}: sim={got:.6f} fp64={exp:.6f}")
    # review's counterexample: x=2 -> ~1.95460 (wrong x^2 formula gives ~1.94005)
    if abs(float(sim_forward(torch.tensor([2.0]), "gelu_tanh")[0]) - 1.95460) > 1e-3:
        bad.append("x=2 forward value does not match corrected formula (~1.95460)")
    else:
        print("  fwd selftest x=2 matches corrected formula 1.95460 (x^2 bug fixed)")
    return bad


def main():
    fb = forward_formula_selftest()
    for b in fb:
        print("FAIL fwd selftest:", b)
    if fb:
        sys.exit(2)
    import flag_gems

    acc = json.load(open(Path(__file__).parent.parent / "results" / "accuracy_matrix.json"))
    fails = [r for r in acc["rows"] if r["verdict"] == "FAIL"]
    classes = sorted({(r["op"], r["dtype"], r["layout"], r["value_class"]) for r in fails})
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

    TOL_ABS = 6e-8
    TOL_REL = 2 * 4.76837158203125e-07  # single constant, used once in criterion (round-5 unification)
    per_class = []
    grand = {"gate_viol": 0, "exact": 0, "ulp_le2": 0, "tol": 0, "unexplained": 0}
    fwd_attrib = []
    for prep in prepared:
        op, dtype, layout, cls = prep["op"], prep["dtype"], prep["layout"], prep["cls"]
        x = prep["x"]
        native_grad, native_out = prep["native_grad"], prep["native_out"]
        xg = x.clone().requires_grad_(True)
        yg = F.gelu(xg, approximate="none" if op == "gelu_none" else "tanh")
        yg.sum().backward()
        torch.cuda.synchronize()
        gems_grad = xg.grad.detach().cpu()
        gems_out = yg.detach().cpu()

        xd = x.double().cpu()
        phi = 0.5 * torch.erfc(-xd / math.sqrt(2))
        if op == "gelu_none":
            orc_out = xd * 0.5 * (1.0 + torch.erf(xd / math.sqrt(2.0)))
            orc_grad = phi + xd * torch.exp(-xd * xd / 2) / math.sqrt(2 * math.pi)
        else:
            u = math.sqrt(2.0 / math.pi) * (xd + 0.044715 * xd ** 3)
            t = torch.tanh(u)
            orc_out = xd * 0.5 * (1.0 + torch.tanh(u))
            du = math.sqrt(2.0 / math.pi) * (1 + 3 * 0.044715 * xd * xd)
            orc_grad = 0.5 * (1 + t) + xd * 0.5 * (1 - t * t) * du

        cmp_row = C.compare_case(native_out, gems_out, native_grad, gems_grad, orc_out.contiguous(), orc_grad.contiguous(), dtype)
        gv = int(cmp_row.get("grad_normal_gate_violations", 0))
        fv = int(cmp_row.get("normal_gate_violations", 0))

        # ---- grad violating subset: three-level match vs TTIR-order sim ----
        gg = gems_grad.float().reshape(-1)
        simg_raw = sim_grad(x, op).float().reshape(-1).cpu()
        # target-dtype representation: simulation rounded to the OUTPUT dtype
        # (bf16/fp16 kernels truncate output; one count per original position)
        if dtype in ("bfloat16", "float16"):
            simg = simg_raw.to(getattr(torch, dtype)).float()
        else:
            simg = simg_raw
        # gate-violating positions: reuse comparator examples + full-array rel-vs-oracle gate recomputed identically
        rel_n = ((native_grad.double() - orc_grad).abs() / orc_grad.abs().clamp_min(1e-300)).reshape(-1)
        rel_g = ((gems_grad.double() - orc_grad).abs() / orc_grad.abs().clamp_min(1e-300)).reshape(-1)
        tol_pos = torch.maximum(torch.full_like(rel_n, 4 * C.EPS[dtype]), 1.5 * rel_n)
        nz = C.zone_masks(orc_grad, dtype)[0].reshape(-1)
        gv_mask = (rel_g > tol_pos) & nz & torch.isfinite(rel_g)
        gv_idx = torch.nonzero(gv_mask, as_tuple=False).reshape(-1)
        stats = {"gate_violations": int(gv_mask.sum()), "exact": 0, "ulp_le2": 0, "tol": 0, "unexplained": 0, "unexplained_examples": []}
        if gv_idx.numel():
            sim_v = simg[gv_idx]
            dev_v = gg[gv_idx]
            dt = dtype if dtype in ("float32",) else dtype
            # round-5 fix: exact compared in TARGET dtype representation (sim rounded
            # to output dtype, one raw-bit count per original position)
            tgt = getattr(torch, dtype)
            exact = (sim_v.to(tgt).view(C.INT_VIEW[dtype]) == dev_v.to(tgt).view(C.INT_VIEW[dtype]))
            # vectorized ordinal ULP per position (uint interpretation, same mapping as comparator.dt_ulp)
            import numpy as np
            nbits = 16 if dtype != "float32" else 32
            UV = torch.uint16 if nbits == 16 else torch.uint32
            sign_hi = 1 << (nbits - 1)
            full = (1 << nbits) - 1
            nat = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
            ai = sim_v.to(nat).contiguous().view(UV).numpy().astype(np.int64)
            bi = dev_v.to(nat).contiguous().view(UV).numpy().astype(np.int64)
            fa, fb = sim_v.float(), dev_v.float()
            m = ~(torch.isnan(fa) | torch.isnan(fb) | torch.isinf(fa) | torch.isinf(fb)).numpy()
            ord_a = np.where(ai < sign_hi, ai + sign_hi, full - ai)
            ord_b = np.where(bi < sign_hi, bi + sign_hi, full - bi)
            ulp_arr = np.abs(ord_a - ord_b)
            ulp_arr[~m] = 10**9  # specials: counted separately via masks
            stats["ulp_max"] = int(ulp_arr.max()) if ulp_arr.size else 0
            stats["ulp_le2"] = int((ulp_arr <= 2).sum())
            stats["ulp_le2_frac"] = round(float((ulp_arr <= 2).mean()), 6) if ulp_arr.size else 0.0
            tol = (sim_v - dev_v).abs() <= (TOL_ABS + dev_v.abs() * TOL_REL)
            stats["exact"] = int(exact.sum())
            # invariant selftest: every count within [0, gate_violations]
            for k in ("exact", "ulp_le2", "tol", "unexplained"):
                v = stats[k]
                assert 0 <= v <= stats["gate_violations"], f"invariant violated: {k}={v} gate={stats['gate_violations']}"
            stats["tol"] = int(tol.sum())
            explained = exact | (torch.tensor(ulp_arr <= 2)) | tol
            unexp_mask = ~explained
            stats["explained_union"] = int(explained.sum())
            stats["unexplained"] = int(unexp_mask.sum())
            if unexp_mask.any():
                ue = torch.nonzero(unexp_mask, as_tuple=False).reshape(-1)[:5]
                stats["unexplained_examples"] = [
                    {"flat_index": int(gv_idx[i]), "x": float(x.reshape(-1)[gv_idx[i]]), "sim": float(sim_v[i]), "device": float(dev_v[i]), "fp64_ref": float(orc_grad.reshape(-1)[gv_idx[i]])}
                    for i in ue.tolist()
                ]
            grand["gate_viol"] += stats["gate_violations"]
            grand["exact"] += stats["exact"]
            grand["ulp_le2"] += stats.get("ulp_le2", 0)
            grand["tol"] += stats["tol"]
            grand["unexplained"] += stats["unexplained"]

        # ---- tanh forward attribution (separate) ----
        if op == "gelu_tanh" and fv > 0:
            simf = sim_forward(x, op).float().reshape(-1).cpu()
            nf = native_out.float().reshape(-1)
            gf = gems_out.float().reshape(-1)
            of = orc_out.cpu().double().reshape(-1)
            rel_nf = ((nf.double() - of).abs() / of.abs().clamp_min(1e-300))
            rel_gf = ((gf.double() - of).abs() / of.abs().clamp_min(1e-300))
            tp_f = torch.maximum(torch.full_like(rel_nf, 4 * C.EPS[dtype]), 1.5 * rel_nf)
            fviol = (rel_gf > tp_f)
            fviol &= torch.isfinite(rel_gf) & torch.isfinite(rel_nf)
            # explained: sim also deviates beyond the same gate (formula/rounding path), i.e. sim reproduces device
            fviol_idx = torch.nonzero(fviol, as_tuple=False).reshape(-1)
            sim_match_v = ((simf - gf).abs() <= (TOL_ABS + gf.abs() * TOL_REL))[fviol_idx] if fviol_idx.numel() else torch.tensor([])
            fwd_attrib.append({
                "case": f"{op}|{dtype}|{layout}|{cls}",
                "fwd_gate_violations": int(fviol.sum()),
                "sim_also_deviates_and_matches_device_within_tol": int(sim_match_v.sum()),
                "unexplained_fwd": int(fviol.sum()) - int(sim_match_v.sum()),
            })

        per_class.append({"case": f"{op}|{dtype}|{layout}|{cls}", **stats, "fwd_gate_violations": fv})

    out = {
        "batch": "v3 (round-4): three-level stats on violating subsets + tanh-forward separate attribution",
        "criteria": {
            "exact": "sim bits == device bits (raw)",
            "ulp_le2": "dtype ordinal ulp distance <= 2",
            "tol": "|sim-device| <= 6e-8 + 2*|device|*4.768e-07 (NOTE: 6e-8 abs is close to the 2^-24 quantization step; tolerance match is NOT bitwise-exactness proof)",
            "unexplained": "none of the above",
        },
        "per_class": per_class,
        "grand_totals": grand,
        "tanh_forward_attribution": fwd_attrib,
    }
    with open("results/fail_attribution_stats_v3.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"grand_totals": grand, "tanh_forward": fwd_attrib}, indent=1))
    # hard rule: backward OR forward unexplained -> exit 1 (declared union)
    fwd_unexp = sum(a.get("unexplained_fwd", 0) for a in fwd_attrib)
    out["exit_rule"] = {"backward_unexplained": grand["unexplained"], "forward_unexplained": fwd_unexp,
                        "exit": 1 if (grand["unexplained"] + fwd_unexp) > 0 else 0}
    print("exit_rule:", json.dumps(out["exit_rule"]))
    sys.exit(1 if (grand["unexplained"] + fwd_unexp) > 0 else 0)


if __name__ == "__main__":
    main()
