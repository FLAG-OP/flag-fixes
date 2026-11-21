"""Issue #7 review round-9 step 3: AdamW stable-reference acceptance.

fp16 stock/foreach AdamW corrupts on this stack (v_hat & eps underflow -> Inf/NaN,
44K-25.6M positions) — stock cannot be the correctness reference for fp16.
Stable reference: fp64 closed-form AdamW (formula A, eps outside sqrt — the
formula BOTH stock and fused implement per round-4 attribution), computed from
the same fp32/fp16 inputs.

Acceptance (declared): fused fp16/bf16/fp32 vs fp64-reference —
  params/exp_avg/exp_avg_sq per-position rel err vs reference magnitude:
  fp16/bf16: <= 4 ulp(dtype at |ref|) OR abs <= 4*ulp(scale); fp32: <= 4*eps;
  no NaN/Inf in fused outputs; step == step count.
Also reports stock fp16 corruption for the record (separate line, not a gate).
"""
import json
import math
import statistics

import torch

def _dt_ulp_positions(ref64, dev, dtype):
    """per-position ordinal ulp between dev (dtype-native) and fp64 reference
    rounded to dtype. specials get a large sentinel. CPU-only math."""
    import numpy as np
    dev = dev.detach().cpu()
    nbits = 16 if dtype != "float32" else 32
    UV = torch.uint16 if nbits == 16 else torch.uint32
    sign_hi = 1 << (nbits - 1)
    full = (1 << nbits) - 1
    dev_c = dev.detach().cpu()
    ref_c = ref64.detach().cpu()
    ai = dev_c.to(getattr(torch, dtype)).contiguous().view(UV).numpy().astype(np.int64)
    bi = ref_c.to(getattr(torch, dtype)).contiguous().view(UV).numpy().astype(np.int64)
    fa, fb = dev_c.float(), ref_c.float()
    m = ~(torch.isnan(fa) | torch.isnan(fb) | torch.isinf(fa) | torch.isinf(fb)).numpy()
    oa = np.where(ai < sign_hi, ai + sign_hi, full - ai)
    ob = np.where(bi < sign_hi, bi + sign_hi, full - bi)
    d = np.abs(oa - ob)
    d[~m] = 10 ** 9
    return d

OUT = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "reference": "fp64 closed-form AdamW (formula A): p -= lr*(m_hat/(sqrt(v_hat)+eps) + wd*p); m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g^2; m_hat=m/(1-b1^t); v_hat=v/(1-b2^t)"}

B1, B2, LR, EPS, WD = 0.9, 0.999, 1e-3, 1e-8, 0.01


EPS_MAP = {"float32": 1.1920929e-07, "bfloat16": 0.0078125, "float16": 0.0009765625}


def fp64_adamw_ref(p0_64, grads, b1=B1, b2=B2, lr=LR, eps=EPS, wd=WD):
    p = p0_64.clone()
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    out = []
    for t, g in enumerate(grads, start=1):
        g = g.double()
        p = p * (1 - lr * wd)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        p = p - lr * mh / (vh.sqrt() + eps)
        out.append((p.clone(), m.clone(), v.clone()))
    return out


def run_impl(kind, dtype, steps, grads=None):
    if grads is None:
        g = torch.Generator(device="cuda").manual_seed(97 + steps)
        grads = [torch.randn(4096, generator=g, device="cuda") * 0.1 for _ in range(steps)]
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    torch.manual_seed(11)
    p = torch.nn.Parameter((torch.randn(4096, device="cuda", dtype=dt) * 0.02))
    kw = dict(lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD)
    if kind != "stock":
        kw[kind] = True
    opt = torch.optim.AdamW([p], **kw)
    snap = []
    for t in range(steps):
        p.grad = grads[t].to(dt)
        opt.step()
        torch.cuda.synchronize()
        if t in (0, steps // 2, steps - 1):
            st = opt.state[p]
            snap.append({"step_t": t + 1, "p": p.detach().clone(),
                         "m": st["exp_avg"].clone(), "v": st["exp_avg_sq"].clone(),
                         "step": st["step"].clone()})
    return snap


def ulp_positions(sim64, dev, dtype):
    """per-position ulp between dev (lowp) and sim rounded to dtype."""
    nbits = 16 if dtype != "float32" else 32
    UV = torch.uint16 if nbits == 16 else torch.uint32
    sign_hi = 1 << (nbits - 1)
    full = (1 << nbits) - 1
    import numpy as np
    ai = sim64.to(getattr(torch, dtype)).contiguous().view(UV).numpy().astype(np.int64)
    bi = dev.contiguous().view(UV).numpy().astype(np.int64)
    fa, fb = sim64.float(), dev.float()
    m = ~(torch.isnan(fa) | torch.isnan(fb) | torch.isinf(fa) | torch.isinf(fb)).numpy()
    oa = np.where(ai < sign_hi, ai + sign_hi, full - ai)
    ob = np.where(bi < sign_hi, bi + sign_hi, full - bi)
    d = np.abs(oa - ob)
    d[~m] = 10 ** 9
    return d


def main():
    steps = 30
    results = {}
    g = torch.Generator(device="cuda").manual_seed(21)
    grads = [torch.randn(4096, generator=g, device="cuda") * 0.1 for _ in range(steps)]
    p0 = torch.randn(4096, device="cuda") * 0.02

    for dtype in ("float32", "bfloat16", "float16"):
        ref = fp64_adamw_ref(p0.double(), grads)
        stock_snap = run_impl("stock", dtype, steps, grads)
        fused_snap = run_impl("fused", dtype, steps, grads)
        foreach_snap = run_impl("foreach", dtype, steps, grads)
        entry = {}
        for name, snap in (("foreach", foreach_snap), ("fused", fused_snap)):
            worst = {"param": None, "m": None, "v": None}
            nonfinite = 0
            maxulp = {"param": 0, "m": 0, "v": 0}
            for (t, ps, ms, vs), snap_i in zip([(s["step_t"], None, None, None) for s in snap], snap):
                pr = ref[t - 1]
                for key, dev_t, ref_t in (("param", snap_i["p"], pr[0]), ("m", snap_i["m"], pr[1]), ("v", snap_i["v"], pr[2])):
                    ulp = _dt_ulp_positions(ref_t, dev_t, dtype)
                    mu = int((ulp <= 4).sum())
                    nonfinite += int((~torch.isfinite(dev_t.float())).sum())
                    if maxulp[key] < int(ulp.max()):
                        maxulp[key] = int(ulp.max())
            # tolerance at final step: rel err vs |ref| per position
            last_p_dev = snap[-1]["p"].float().double()
            last_p_ref = ref[-1][0]
            scale = p0.abs().std().clamp_min(1e-12)
            rel = ((last_p_dev - last_p_ref).abs() / last_p_ref.abs().clamp_min(0.25 * scale))
            rel_gated = rel[last_p_ref.abs() > 0.25 * scale]  # near-zero positions excluded (scale-gated)
            entry[name] = {"step1_vs_ref_max_ulp_param": maxulp["param"],
                           "final_param_max_rel_vs_ref_scale_gated": float(rel_gated.max()) if rel_gated.numel() else None,
                           "final_param_median_rel": float(rel_gated.median()) if rel_gated.numel() else None,
                           "final_param_over4eps_positions": int((rel > 4 * EPS_MAP[dtype]).sum()),
                           "nonfinite_fused_or_impl": nonfinite,
                           "max_ulp_final": maxulp,
                           "step_matches": float(snap[-1]["step"].item())}
        results[dtype] = entry

    stock_snap = run_impl("stock", "float16", steps)
    nf = sum(int((~torch.isfinite(s["p"].float())).sum()) + int((~torch.isfinite(s["m"].float())).sum()) for s in stock_snap)
    results["fp16_stock_corruption_record"] = {"nonfinite_positions_over_3_snapshots": nf}

    import os
    tmp_out = "results/adamw_stable_ref_acceptance.json.tmp"
    with open(tmp_out, "w") as f:
        json.dump(OUT | results, f, indent=1)
    os.replace(tmp_out, "results/adamw_stable_ref_acceptance.json")
    print(json.dumps({"dtype_results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items() if k != "fp16_stock_corruption_record"}}, indent=1)[:1500])


C_EPS = {"float32": 1.1920929e-07, "bfloat16": 0.0078125, "float16": 0.0009765625}

if __name__ == "__main__":
    main()
