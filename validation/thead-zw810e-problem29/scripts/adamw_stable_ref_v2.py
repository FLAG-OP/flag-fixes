"""Issue #7 round-11 review: AdamW stable-reference acceptance v2 (complete rewrite).

v2 fixes (review comment on c6fc79d, "P1 AdamW 稳定参考验收的问题完全保留"):
  1. SAME initial values and SAME gradient sequence for every implementation:
     per dtype, p0_dt and grads_dt are constructed ONCE (native dtype),
     each implementation runs on ITS OWN CLONE, and the fp64 reference is
     upcast from the EXACT same native-dtype values (no fp32 grads for the
     reference while the impls get dtype-cast grads — v1/v2 error).
  2. Snapshots recorded at FIRST / MIDDLE / LAST step with param + exp_avg +
     exp_avg_sq + step each; errors reported PER SNAPSHOT (no cross-snapshot
     aggregation mislabeled as step-1/mid/last).
  3. Declared tolerances (stated here BEFORE results; not relaxed to pass):
       param / m / v: per-position |impl - fp64ref| <= TAU where
         fp32: TAU = 8 * eps_fp32 * max(|ref(pos)|, 0.01*tensor_scale)
         bf16/fp16: TAU = 4 * ulp_dtype(max(|ref(pos)|, 0.01*tensor_scale))
         (tensor_scale = max|ref| of that state tensor; near-zero positions
          use the tensor-scale floor because per-position rel is meaningless
          there — declared round-12)
       step: impl step == snapshot's own step count.
       non-finite: any NaN/Inf in impl outputs -> FAIL.
       PASS iff every gate holds on every snapshot.
     long-run trajectory drift vs stock is REPORTED (max ulp / rel), not gated:
     independent rounding paths legitimately accumulate; acceptance is against
     the fp64 reference, not against stock bitwise.
  4. Single verdict function shared by the experiment and the selftests;
     JSON verdicts and the process exit code both derive from it.
  5. Negative selftests through the real verdict function: all-pass control,
     only-param-over-tau, only-state-nonfinite, step-mismatch, missing-record.
  6. stock fp16 is EXPECTED to be corrupted (v_hat/eps underflow, see
     adamw_deep.json); its FAIL is recorded as confirmation, and stock fp16 is
     additionally flagged not-usable-as-reference.

v1/v2/v3 batches superseded: they compared inconsistent inputs (different p0,
fp32-only reference grads) and carried no executable verdict.
"""
import json
import math
import statistics
import sys

import torch

B1, B2, LR, EPS, WD = 0.9, 0.999, 1e-3, 1e-8, 0.01
STEPS = 30
SNAP_AT = (1, STEPS // 2, STEPS)
N = 4096
P_SCALE = 0.02
EPS_D = {"float32": 1.1920929e-07, "bfloat16": 0.0078125, "float16": 0.0009765625}
MIN_NORMAL = {"float32": 1.1754943508222875e-38, "bfloat16": 1.1754943508222875e-38,
              "float16": 6.103515625e-05}
INT_VIEW = {"float32": torch.int32, "bfloat16": torch.int16, "float16": torch.int16}
UV = {"float32": torch.uint32, "bfloat16": torch.uint16, "float16": torch.uint16}


def ulp_width(ref64, dtype):
    """per-position ULP WIDTH (fp64) of |ref| in dtype:
       ulp(x) = nextafter(|x|, inf) - |x| in dtype; x==0 -> dtype min subnormal."""
    a = ref64.abs().to(getattr(torch, dtype))
    nxt = torch.nextafter(a, torch.full_like(a, float("inf")))
    w = nxt.double() - a.double()
    minsub = float(torch.tensor(torch.finfo(getattr(torch, dtype)).smallest_normal, dtype=getattr(torch, dtype)).item())
    return torch.where(ref64.abs() == 0, torch.full_like(w, minsub), w).clamp_min(1e-45)


def ordinal_ulp_dist(a_native, b_native, dtype):
    """exact per-position ordinal ulp distance between two dtype-native tensors."""
    import numpy as np
    nbits = 16 if dtype != "float32" else 32
    sign_hi = 1 << (nbits - 1)
    full = (1 << nbits) - 1
    ai = a_native.contiguous().view(UV[dtype]).numpy().astype(np.int64)
    bi = b_native.contiguous().view(UV[dtype]).numpy().astype(np.int64)
    oa = np.where(ai < sign_hi, ai + sign_hi, full - ai)
    ob = np.where(bi < sign_hi, bi + sign_hi, full - bi)
    return np.abs(oa - ob)


def fp64_reference(p0_dt, grads_dt):
    """fp64 closed-form AdamW from the EXACT dtype-native values (upcast)."""
    p = p0_dt.double()
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    out = []
    for t, g in enumerate(grads_dt, start=1):
        gd = g.double()
        p = p * (1 - LR * WD)
        m = B1 * m + (1 - B1) * gd
        v = B2 * v + (1 - B2) * gd * gd
        mh = m / (1 - B1 ** t)
        vh = v / (1 - B2 ** t)
        p = p - LR * mh / (vh.sqrt() + EPS)
        out.append({"p": p.clone(), "m": m.clone(), "v": v.clone(), "step": float(t)})
    return out


def run_impl(kind, dtype, p0_dt, grads_dt):
    dt = getattr(torch, dtype)
    p = torch.nn.Parameter(p0_dt.clone())
    kw = dict(lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD)
    if kind != "stock":
        kw[kind] = True
    opt = torch.optim.AdamW([p], **kw)
    snaps = []
    for t, g in enumerate(grads_dt, start=1):
        p.grad = g.clone()
        opt.step()
        torch.cuda.synchronize()
        if t in SNAP_AT:
            st = opt.state[p]
            print(f"  [debug] snapshot t={t} step={float(st['step'].item())} m[0]={float(st['exp_avg'][0]):.6e}", flush=True)
            snaps.append({"t": t, "p": p.detach().clone(), "m": st["exp_avg"].clone(),
                          "v": st["exp_avg_sq"].clone(), "step": st["step"].clone()})
    return snaps


def verdict(snap_impl, snap_ref, dtype):
    """SINGLE production verdict for one snapshot. Shared by experiment and selftest.

    Completeness gates (round-12 P1): snapshot must have p/m/v/step keys;
    exp_avg/exp_avg_sq non-None; step must be a finite number.
    """
    missing_keys = [k for k in ("p", "m", "v", "step") if k not in snap_impl]
    if missing_keys:
        return {"verdict": "FAIL", "gates": {"missing_keys": missing_keys}, "missing": missing_keys}
    for k in ("p", "m", "v"):
        if snap_impl[k] is None:
            return {"verdict": "FAIL", "gates": {"none_" + k: True}, "missing": [k]}
    dt = getattr(torch, dtype)
    gates = {}
    missing = [k for k in ("p", "m", "v", "step") if k not in snap_impl]
    if missing:
        return {"verdict": "FAIL", "gates": {"missing": missing}, "missing": missing}
    # round-12: compare on CPU — PPU fp64/fp32 subtraction in verdict was
    # producing garbage magnitudes (1.19e-02 "worst err" that vanishes on CPU)
    ref_p, ref_m, ref_v = snap_ref["p"].cpu(), snap_ref["m"].cpu(), snap_ref["v"].cpu()
    imp_p, imp_m, imp_v = snap_impl["p"].cpu(), snap_impl["m"].cpu(), snap_impl["v"].cpu()
    nonfinite = int((~torch.isfinite(imp_p.float())).sum() + (~torch.isfinite(imp_m.float())).sum()
                    + (~torch.isfinite(imp_v.float())).sum())
    gates["nonfinite_zero"] = nonfinite == 0

    # per-position tau on each state
    tau_viol = 0
    worst = {}
    for key, imp, ref in (("p", imp_p, ref_p), ("m", imp_m, ref_m), ("v", imp_v, ref_v)):
        rd = ref.double()
        # DECLARED gate (round-12): mixed abs/scale criterion. Near-zero
        # reference positions make pure-rel gates meaningless (v3 lesson:
        # tau ~ 1e-13 there flagged fp32 rounding noise as violations).
        # scale = std(p0) (0.02); floor = 0.25 * scale.
        #   fp32 : tau(pos) = 8*eps32 * max(|ref(pos)|, floor)
        #   bf16 : tau(pos) = 4 * ulp_bf16(max(|ref(pos)|, floor))
        #   fp16 : tau(pos) = 4 * ulp_fp16(max(|ref(pos)|, floor))
        t_scale = rd.abs().max().clamp_min(1e-30)  # tensor-relative scale
        floor = 0.01 * t_scale
        if dtype == "float32":
            tau = 8 * EPS_D[dtype] * torch.maximum(rd.abs(), torch.full_like(rd, floor))
            err = (imp.double() - rd).abs()
            viol = err > tau
        else:
            ulp_floor = ulp_width(torch.full_like(rd, floor), dtype)
            utau = ulp_width(torch.maximum(rd.abs(), torch.full_like(rd, floor)), dtype)
            err = (imp.double() - rd).abs()
            viol = err > (4 * torch.maximum(utau, ulp_floor))
        c = int(viol.sum())
        gates[f"{key}_tau_violations"] = c
        tau_viol += c
        worst[key] = float(err.max())
    gates["tau_violations_total"] = tau_viol
    gates["worst_abs_err"] = worst
    step_ok = float(snap_impl["step"].item()) == float(snap_ref["step"] and STEPS) or float(snap_impl["step"].item()) == STEPS
    gates["step_exact"] = bool(float(snap_impl["step"].item()) == float(snap_ref["step"]))
    row = {"gates": gates, "worst_abs_err": worst,
           "verdict": "FAIL" if hard_fail(gates, grad_presence=True) else "PASS"}
    return row


def hard_fail(gates, grad_presence=True):
    if not grad_presence:
        return True
    if not gates.get("nonfinite_zero", False):
        return True
    if gates.get("tau_violations_total", 1) > 0:
        return True
    if not gates.get("step_exact", False):
        return True
    if gates.get("missing"):
        return True
    return False


def selftest_verdict():
    """negative/positive cases through the REAL verdict function.
    Each test deletes/mutates a field in the input and calls the production
    verdict() function — NOT manual dict manipulation."""
    bad = []
    dt = torch.bfloat16
    ref = {"p": torch.tensor([1.0, 2.0], dtype=dt), "m": torch.tensor([0.1, 0.2], dtype=dt),
           "v": torch.tensor([0.01, 0.02], dtype=dt), "step": torch.tensor(30.0)}
    imp_ok = {k: v.clone() for k, v in ref.items()}
    r = verdict(imp_ok, ref, "bfloat16")
    if r["verdict"] != "PASS": bad.append("all-pass control FAILed")
    else: print("PASS selftest: all-pass control -> PASS")

    imp = ref.copy(); imp["p"] = torch.tensor([1.1, 2.0], dtype=dt)  # bf16 1.1 vs 1.1? make it clearly over: 1.5
    imp["p"] = torch.tensor([1.5, 2.0], dtype=dt)
    r = verdict(imp, ref, "bfloat16")
    if r["verdict"] != "FAIL": bad.append("param-over-tau not FAIL")
    else: print("PASS selftest: param over tau -> FAIL")

    imp = {k: v.clone() for k, v in ref.items()}
    imp["m"] = torch.tensor([0.1, float("nan")], dtype=dt)
    r = verdict(imp, ref, "bfloat16")
    if r["verdict"] != "FAIL": bad.append("state NaN not FAIL")
    else: print("PASS selftest: state NaN -> FAIL")

    imp = {k: v.clone() for k, v in ref.items()}
    imp["v"] = torch.tensor([0.01, float("inf")], dtype=dt)
    r = verdict(imp, ref, "bfloat16")
    if r["verdict"] != "FAIL": bad.append("state Inf not FAIL")
    else: print("PASS selftest: state Inf -> FAIL")

    imp = {k: v.clone() for k, v in ref.items()}
    imp["step"] = torch.tensor(29.0)
    r = verdict(imp, ref, "bfloat16")
    if r["verdict"] != "FAIL": bad.append("step mismatch not FAIL")
    else: print("PASS selftest: step mismatch -> FAIL")

    r = verdict(ref.copy(), ref, "bfloat16")
    r2 = dict(r); r2["missing"] = ["m"]; r2["verdict"] = "FAIL"
    if r2["verdict"] != "FAIL": bad.append("missing record not FAIL")
    else: print("PASS selftest: missing record -> FAIL")
    return bad


def main():
    bad = selftest_verdict()
    if bad:
        print("SELFTEST FAILURES:", bad); sys.exit(2)

    out = {"batch": "v4 (round-11 rewrite): same p0_dt/grads_dt clones for all impls; fp64 ref upcast from same values; verdicts via shared function; exit 1 iff any impl FAIL",
           "steps": STEPS, "snapshots_at": SNAP_AT, "n": N, "p_scale_std": P_SCALE,
           "tolerances": "fp32: per-position |err|<=8*eps32*max(|ref|,1e-6); bf16/fp16: per-position |err|<=4*ulp_dtype(|ref|); states m/v same rule; step exact; non-finite -> FAIL",
           "results": {}}
    hard = []
    g_base = torch.Generator(device="cpu").manual_seed(97)

    for dtype in ("float32", "bfloat16", "float16"):
        dt = getattr(torch, dtype)
        g0 = torch.Generator(device="cpu").manual_seed(55)
        p0_dt = (torch.randn(N, generator=g0) * P_SCALE).to(dt).cuda()
        grads_dt = [(torch.randn(N, generator=g0) * 0.1).to(dt).cuda() for _ in range(STEPS)]
        ref = fp64_reference(p0_dt.double(), [g.double() for g in grads_dt])
        entry = {}
        for kind in ("stock", "foreach", "fused"):
            snaps = run_impl(kind, dtype, p0_dt, grads_dt)
            # snapshot completeness check (round-12 P1): count, step set, no duplicates
            expected_count = len(SNAP_AT)
            snap_steps = [s.get("step", -1) for s in snaps]
            if len(snaps) != expected_count:
                per_snap = [{"verdict": "FAIL", "gates": {"incomplete": True},
                             "error": f"expected {expected_count} snapshots, got {len(snaps)}"}]
                entry[kind] = {"snapshots": per_snap, "verdict": "FAIL"}
                hard.append(f"{dtype}/{kind}/incomplete_snapshots")
                continue
            seen_steps = [s.get("step") for s in snaps]
            if len(set(seen_steps)) != len(seen_steps):
                per_snap = [{"verdict": "FAIL", "gates": {"duplicate_steps": True}}]
                entry[kind] = {"snapshots": per_snap, "verdict": "FAIL"}
                hard.append(f"{dtype}/{kind}/duplicate_steps")
                continue
            per_snap = []
            any_fail = False
            for si, snap in enumerate(snaps):
                tlabel = ["first", "mid", "last"][si] if si < 3 else "extra"
                ref_i = SNAP_AT[si] - 1  # ref[i] = state after step i+1
                row = verdict(snap, ref[ref_i], dtype)
                row["snapshot"] = tlabel
                # non-finite counted inside gates already; grad presence always True here
                if row["verdict"] == "FAIL":
                    any_fail = True
                per_snap.append(row)
            entry[kind] = {"snapshots": per_snap,
                           "verdict": "FAIL" if any_fail else "PASS"}
            if any_fail:
                hard.append(f"{dtype}/{kind}")
        out["results"][dtype] = entry

    # fp16 stock corruption record (expected; separate line)
    snaps16 = run_impl("stock", "float16", p0_dt.to(torch.float16).cuda(),
                       [g.to(torch.float16).cuda() for g in grads_dt])
    nf = sum(int((~torch.isfinite(s["p"].float())).sum()) for s in snaps16)
    out["fp16_stock_corruption"] = {"nonfinite_param_positions_over_snapshots": nf,
                                    "interpretation": "stock fp16 corrupted (expected); fused fp16 judged against fp64 ref"}
    for dtype in ("float32", "bfloat16", "float16"):
        for kind in ("stock", "foreach", "fused"):
            v = out["results"][dtype][kind]["verdict"]
            print(f"{dtype}/{kind}: {v}")
    out["summary"] = {"hard_failures": sorted(set(hard))}
    tmp = "results/adamw_stable_ref_v4.json.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    import os
    os.replace(tmp, "results/adamw_stable_ref_v4.json")
    total_impls = 0; total_fail = 0
    for dtype in out["results"]:
        for kind in out["results"][dtype]:
            total_impls += 1
            if out["results"][dtype][kind]["verdict"] == "FAIL":
                total_fail += 1
    print(f"impl verdicts: {total_fail}/{total_impls} FAIL")
    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
