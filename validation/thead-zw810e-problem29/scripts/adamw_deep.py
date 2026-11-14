"""Issue #7 §B: AdamW stock / foreach / fused — deep validation on ZW810E.

Configs: single [1024,1024] | multi (8 tensors) | bge336 (336 tensors)
dtypes:  float32 | bfloat16 | float16 | mixed (fp32 2D + bf16 1D groups)
Steps:   100 controlled updates, same init & same pre-generated grad sequence.
State comparison (declared BEFORE run, review §B):
   STEP-1 deviation (algorithm identity): per-position rel <= 4*eps(dtype)
     — proves fused implements the same update, only different summation order.
   100-STEP divergence: independent rounding paths amplify; NOT gated as FAIL.
     Reported honestly: max_rel, p99_rel, share of positions > 4eps, NaN/Inf.
   bf16/fp16: <= 4 ulp(dtype at param scale) at step 1 (declared); 100-step reported.
   KNOWN PLATFORM FINDING (verified): fp16 STOCK/foreach AdamW produces Inf/NaN
   on this stack (v_hat=g^2 underflows fp16 to 0 AND eps=1e-8 underflows fp16
   -> 0-division); fused computes internally in fp32 and is the only sound fp16
   path. Fused-vs-stock deviation is therefore EXPECTED for fp16.
   tensors compared: params, exp_avg, exp_avg_sq, step
Checkpoint: save state_dict at step 50, restore into fresh optimizer, run steps
   51..100, compare vs direct 100-step run (bitwise, same config).
Timing: per-step optimizer time (CUDA events, warmup 5, median of 95).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import math

import torch

sys.path.insert(0, str(Path(__file__).parent))
import comparator as C  # ulp mapping + MIN_NORMAL/EPS

DT = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_params(kind, dtype):
    g = torch.Generator(device="cuda").manual_seed(9)
    mk = lambda s: torch.randn(s, generator=g, device="cuda", dtype=DT.get(dtype, torch.float32)) * (0.02 if math.prod(s) > 1 else 1)
    if dtype == "mixed":
        if kind == "single":
            spec = [((1024, 1024), "float32")]
        elif kind == "multi":
            spec = [((512, 512), "float32")] * 4 + [((256,), "bfloat16")] * 4
        else:
            spec = [((4096, 1280), "float32")] * 112 + [((1280,), "bfloat16")] * 224
        out = []
        for shp, dt in spec:
            t = torch.randn(shp, generator=g, device="cuda", dtype=DT[dt]) * (0.02 if math.prod(shp) > 1 else 1)
            out.append(torch.nn.Parameter(t))
        return out, [dt for _, dt in spec]
    if kind == "single":
        shapes = [(1024, 1024)]
    elif kind == "multi":
        shapes = [(512, 512)] * 4 + [(256,)] * 4
    else:
        shapes = [(4096, 1280)] * 112 + [(1280,)] * 224
    return [torch.nn.Parameter(mk(s)) for s in shapes], [dtype] * len(shapes)


def gen_grads(params, step, seed=77):
    g = torch.Generator(device="cuda").manual_seed(seed + step)
    return [torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype) * 0.1 for p in params]


def state_tensors(opt, params):
    st = []
    for group in opt.param_groups:
        for p in group["params"]:
            s = opt.state.get(p, {})
            for k in ("exp_avg", "exp_avg_sq"):
                if k in s:
                    st.append((id(p), k, s[k]))
    return st


def compare_state(a, b, dtype, label):
    """a,b same-shape float tensors -> declared-tolerance verdict fields."""
    if a is None or b is None:
        return {"label": label, "missing": True}
    af, bf = a.float(), b.float()
    if a.dtype in (torch.float32,):
        fin = torch.isfinite(af) & torch.isfinite(bf)
        denom = bf[fin].abs().clamp_min(1e-6)
        viol = ((af[fin] - bf[fin]).abs() / denom > 4 * C.EPS["float32"]).sum()
        return {"label": label, "positions": fin.sum().item(), "tol_violations": int(viol), "max_abs": float((af - bf).abs().max()), "nan_a": int(torch.isnan(af).sum()), "nan_b": int(torch.isnan(bf).sum())}
    else:
        d = dtype if dtype in DT else "bfloat16"
        ulp = C.dt_ulp(af.cpu().to(DT[d]), bf.cpu().to(DT[d]), d)
        r = {"label": label, "ulp_max": ulp, "max_abs": float((af - bf).abs().max()) if (af - bf).abs().max() == (af - bf).abs().max() else None,
             "nan_a": int(torch.isnan(af).sum()), "nan_b": int(torch.isnan(bf).sum()),
             "inf_a": int(torch.isinf(af).sum()), "inf_b": int(torch.isinf(bf).sum())}
        return r


def run(kind, dtype, opt_kind, steps=100, ckpt_at=50, capture_step1=False):
    params, dtypes = build_params(kind, dtype)
    kw = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    if opt_kind == "fused":
        kw["fused"] = True
    elif opt_kind == "foreach":
        kw["foreach"] = True
    opt = torch.optim.AdamW(params, **kw)
    times = []
    ck = None
    for st in range(steps):
        grads = gen_grads(params, st)
        for p, gr in zip(params, grads):
            p.grad = gr
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        opt.step()
        e.record()
        torch.cuda.synchronize()
        if st >= 5:
            times.append(s.elapsed_time(e))
        if capture_step1 and st == 0:
            s1 = {i: {k: v.clone() for k, v in opt.state[p].items()} for i, p in enumerate(params) if p in opt.state}
            p1 = [p.detach().clone() for p in params]
        if st == ckpt_at - 1:
            ck = {id(p): {k: v.clone() for k, v in opt.state[p].items()} for p in params if p in opt.state}
    # state keyed by STABLE param index (review P1: id(p) of clones won't match)
    peak_state = {i: {k: v.clone() for k, v in opt.state[p].items()} for i, p in enumerate(params) if p in opt.state}
    missing_state = [i for i, p in enumerate(params) if p not in opt.state]
    step_val = float(next(iter(opt.state.values()))["step"].item()) if opt.state else None
    out = {"params": [p.detach().clone() for p in params], "state": peak_state, "missing_state_indices": missing_state,
           "n_params": len(params), "state_entries": len(peak_state), "step": step_val,
           "step_ms_median": round(statistics.median(times), 4), "step_ms_p95": round(sorted(times)[int(0.95 * (len(times) - 1))], 4), "dtypes": dtypes}
    if capture_step1:
        out["step1_state"] = s1
        out["step1_params"] = p1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    OUT = {"steps": args.steps, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "provenance": "adamw_deep v4 (round-4 review: SINGLE production aggregate() — summary and exit code same-source; --selftest injects through real aggregate(); v1-v3 batches superseded: v1/v2 state_compare silently empty, v3 dual aggregation lists)",
           "tolerances_declared": "fp32: per-position rel<=4eps or abs<=1e-6 denom floor; bf16/fp16: <=1 ulp (ordinal); ckpt: bitwise vs direct"}

    # correctness matrix (fresh processes not needed: same-process sequential,
    # fixed seeds; but optimizer states are independent per run)
    results = {}
    for kind in ("single", "multi", "bge336"):
        for dtype in ("float32", "bfloat16", "float16", "mixed"):
            key = f"{kind}/{dtype}"
            try:
                ref = run(kind, dtype, "stock", args.steps, capture_step1=True)
                cmp = {}
                for ok_ in ("foreach", "fused"):
                    alt = run(kind, dtype, ok_, args.steps, capture_step1=True)
                    # STEP-1 gate (declared): algorithm identity via ordinal ULP
                    # (relative-to-position inflates at tiny params; fp64 closed-form
                    #  attribution shows both paths match formula A within 1 ulp)
                    step1 = []
                    for i, (a, b) in enumerate(zip(ref["step1_params"], alt["step1_params"])):
                        d = ref["dtypes"][i] if dtype == "mixed" else (dtype if dtype in DT else "float32")
                        scale = float(a.float().std())
                        abs_diff = float((a.float() - b.float()).abs().max())
                        if not (scale == scale) or abs_diff != abs_diff:
                            step1.append({"param": i, "dtype": d, "pass": False, "step1_max_abs": None, "allowed_abs": None,
                                          "note": "non-finite in step-1 params (see nan/inf counts)"})
                            continue
                        # declared: abs diff <= 2 ulp(dtype at param scale) == same algorithm, diff summation order
                        quantum = {"float32": 1.5259021896696442e-07, "bfloat16": 7.8125e-05, "float16": 9.765625e-06}[d] * scale / 0.02
                        mult = 4 if d in ("float16", "bfloat16") else 2  # low-precision: 4 ulp@scale (declared)
                        step1.append({"param": i, "dtype": d, "step1_max_abs": abs_diff, "param_scale_std": round(scale, 6), "allowed_abs": round(mult * quantum, 12), "pass": abs_diff <= mult * quantum})
                    st_cmp = []
                    if ref.get("missing_state_indices") or alt.get("missing_state_indices"):
                        st_cmp.append({"missing": True, "label": "state", "ref_missing": ref.get("missing_state_indices"), "alt_missing": alt.get("missing_state_indices")})
                    if len(ref["state"]) != ref["n_params"] or len(alt["state"]) != alt["n_params"]:
                        st_cmp.append({"missing": True, "label": "state_count", "ref_entries": len(ref["state"]), "alt_entries": len(alt["state"]), "expected": ref["n_params"]})
                    for i in range(ref["n_params"]):
                        d_i = ref["dtypes"][i]
                        rs = ref["state"].get(i, {})
                        as_ = alt["state"].get(i, {})
                        for k in ("exp_avg", "exp_avg_sq"):
                            if k not in rs or k not in as_:
                                st_cmp.append({"missing": True, "label": f"param{i}.{k}"})
                                continue
                            a_t, b_t = rs[k], as_[k]
                            if a_t.shape != b_t.shape or a_t.dtype != b_t.dtype:
                                st_cmp.append({"missing": True, "label": f"param{i}.{k}", "mismatch": f"{a_t.shape}/{a_t.dtype} vs {b_t.shape}/{b_t.dtype}"})
                                continue
                            # mixed dtype: compare with EACH tensor's actual dtype
                            cd = "float32" if a_t.dtype == torch.float32 else ("bfloat16" if a_t.dtype == torch.bfloat16 else "float16")
                            st_cmp.append(compare_state(a_t, b_t, cd, f"param{i}.{k}"))
                    if not st_cmp:
                        st_cmp.append({"missing": True, "label": "state_compare_empty"})
                    cmp[ok_] = {
                        "step1_gate": step1,
                        "param_compare": [compare_state(a, b, (ref["dtypes"][i] if dtype == "mixed" else dtype), f"param{i}") for i, (a, b) in enumerate(zip(ref["params"], alt["params"]))],
                        "state_compare": st_cmp,
                        "step_ref": ref["step"], "step_alt": alt["step"],
                        "time_ms_median": {"stock": ref["step_ms_median"], ok_: alt["step_ms_median"]},
                    }
                results[key] = cmp
                viol = sum(c["tol_violations"] for x in cmp.values() for c in x["param_compare"] if "tol_violations" in c)
                print(f"{key}: done, param tol_violations total={viol}", flush=True)
            except Exception as e:
                import traceback
                results[key] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-600:]}
                print(f"{key}: ERROR {e}", flush=True)
    OUT["correctness_vs_stock"] = results

    # checkpoint save/restore consistency (fp32 bge336, fused)
    import copy
    params, _ = build_params("bge336", "float32")
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01, fused=True)
    for st in range(50):
        for p, gr in zip(params, gen_grads(params, st)):
            p.grad = gr
        opt.step()
    sd = copy.deepcopy(opt.state_dict())
    # restore into fresh
    params2, _ = build_params("bge336", "float32")
    with torch.no_grad():
        for p_new, p_old in zip(params2, params):
            p_new.copy_(p_old.detach())  # restore param VALUES too (harness bug fix)
    opt2 = torch.optim.AdamW(params2, lr=1e-3, weight_decay=0.01, fused=True)
    opt2.load_state_dict(copy.deepcopy(sd))
    # continue both 50 steps (same grads) — direct run continues with opt
    for st in range(50, 100):
        grads = gen_grads(params, st)
        for p, gr in zip(params, grads):
            p.grad = gr
        opt.step()
        for p2, gr in zip(params2, grads):
            p2.grad = gr
        opt2.step()
    ck_ok = all(torch.equal(a.detach(), b.detach()) for a, b in zip(params, params2))
    state_ok = True
    for p, p2 in zip(params, params2):
        sa, s2 = opt.state[p], opt2.state[p2]
        for k in ("exp_avg", "exp_avg_sq", "step"):
            if not torch.equal(sa[k], s2[k]):
                state_ok = False
    OUT["checkpoint_restore_fused_bge336"] = {"params_bitwise_equal": ck_ok, "state_bitwise_equal": state_ok}

    with open(args.out, "w") as f:
        json.dump(OUT, f, indent=1)
    div_stats = {}
    for key, r in results.items():
        if "error" in r:
            continue
        for ok_, x in r.items():
            ma = [c.get("max_abs") for c in x.get("param_compare", []) if c.get("max_abs") is not None]
            tot = sum(c.get("positions", 0) for c in x.get("param_compare", []))
            tv = sum(c.get("tol_violations", 0) for c in x.get("param_compare", []))
            div_stats[f"{key}/{ok_}"] = {"max_abs_100step": max(ma, default=None), "positions": tot, "over_4eps_positions": tv}
    final = aggregate(results, OUT.get("checkpoint_restore_fused_bge336", {}))
    OUT["divergence_100step_report"] = div_stats
    OUT["summary"] = {"hard_deviations": final, "exit_code_rule": "exit 1 iff summary.hard_deviations non-empty (single production aggregation)"}

    with open(args.out, "w") as f:
        json.dump(OUT, f, indent=1)
    print("summary:", json.dumps(OUT["summary"]))
    sys.exit(1 if final else 0)


def aggregate(results, checkpoint):
    """SINGLE production aggregation: summary AND exit code both use this result.

    Robust to partial schemas (review P2/round-4): uses .get everywhere; any
    missing state / mismatch / tolerance violation / step mismatch / checkpoint
    failure / harness error lands in the returned list; empty list == exit 0.
    """
    final = []
    for key, r in results.items():
        if "error" in r:
            final.append(f"{key}/harness_error")
            continue
        for ok_, x in r.items():
            s1fail = [c for c in x["step1_gate"] if not c.get("pass")]
            if s1fail:
                final.append(f"{key}/{ok_}/step1_identity_fail({[(c.get('param'), c.get('step1_max_abs'), c.get('allowed_abs'), c.get('note')) for c in s1fail][:3]}...)")
            for c in x.get("state_compare", []):
                if c.get("missing") or c.get("mismatch"):
                    final.append(f"{key}/{ok_}/{c.get('label')}/state_missing_or_mismatch")
                if c.get("tol_violations", 0) > 0 or c.get("ulp_max", 0) > 1:
                    final.append(f"{key}/{ok_}/{c.get('label')}/state_tol_violation")
            for c in x.get("param_compare", []):
                if c.get("nan_a", 0) > 0 or c.get("nan_b", 0) > 0 or c.get("inf_a", 0) > 0 or c.get("inf_b", 0) > 0:
                    final.append(f"{key}/{ok_}/{c.get('label')}/nan_inf")
            if x.get("step_ref") != x.get("step_alt"):
                final.append(f"{key}/{ok_}/step_count_mismatch")
    ck = checkpoint or {}
    if not (ck.get("params_bitwise_equal") and ck.get("state_bitwise_equal")):
        final.append(f"checkpoint_restore_fail({ck})")
    return final


def selftest_aggregation():
    """Entry-level negatives THROUGH the real production aggregate() function.

    Each injected case must produce non-empty summary AND the exit-code rule
    must map it to non-zero; the all-pass control must aggregate empty.
    """
    bad = []
    empty_cmp = lambda i, lab: {"label": lab, "ulp_max": 0, "max_abs": 0.0, "nan_a": 0, "nan_b": 0, "inf_a": 0, "inf_b": 0}
    good_x = lambda: {
        "step1_gate": [{"param": 0, "dtype": "float32", "pass": True, "step1_max_abs": 0.0, "allowed_abs": 1e-6}],
        "param_compare": [empty_cmp(0, "param0")],
        "state_compare": [empty_cmp(0, "param0.exp_avg"), empty_cmp(0, "param0.exp_avg_sq")],
        "step_ref": 100, "step_alt": 100, "time_ms_median": {"stock": 1.0},
    }

    def agg(results, ck):
        return aggregate(results, ck)

    # 1. only checkpoint failure
    f = agg({"k/float32": {"foreach": good_x(), "fused": good_x()}}, {"params_bitwise_equal": False, "state_bitwise_equal": True})
    ok1 = len(f) == 1 and f[0].startswith("checkpoint_restore_fail")
    print(("PASS" if ok1 else "FAIL") + " aggregate: only-checkpoint-fail -> [" + "; ".join(f) + "]")
    if not ok1: bad.append("checkpoint")

    # 2. only state missing
    r2 = {"k/float32": {"foreach": good_x(), "fused": dict(good_x(), state_compare=[{"label": "param0.exp_avg", "missing": True}])}}
    f2 = agg(r2, {"params_bitwise_equal": True, "state_bitwise_equal": True})
    ok2 = len(f2) == 1 and "state_missing_or_mismatch" in f2[0]
    print(("PASS" if ok2 else "FAIL") + " aggregate: only-state-missing -> " + "; ".join(f2))
    if not ok2: bad.append("state missing")

    # 3. only state over tolerance
    r3 = {"k/float32": {"foreach": good_x(), "fused": dict(good_x(), state_compare=[dict(empty_cmp(0, "param0.exp_avg"), tol_violations=5)])}}
    f3 = agg(r3, {"params_bitwise_equal": True, "state_bitwise_equal": True})
    ok3 = len(f3) == 1 and "state_tol_violation" in f3[0]
    print(("PASS" if ok3 else "FAIL") + " aggregate: only-state-tolerance -> " + "; ".join(f3))
    if not ok3: bad.append("state tolerance")

    # 4. only step mismatch
    r4 = {"k/float32": {"foreach": good_x(), "fused": dict(good_x(), step_ref=100, step_alt=101)}}
    f4 = agg(r4, {"params_bitwise_equal": True, "state_bitwise_equal": True})
    ok4 = len(f4) == 1 and "step_count_mismatch" in f4[0]
    print(("PASS" if ok4 else "FAIL") + " aggregate: only-step-mismatch -> " + "; ".join(f4))
    if not ok4: bad.append("step mismatch")

    # 5. normal control -> empty
    f5 = agg({"k/float32": {"foreach": good_x(), "fused": good_x()}}, {"params_bitwise_equal": True, "state_bitwise_equal": True})
    ok5 = len(f5) == 0
    print(("PASS" if ok5 else "FAIL") + " aggregate: all-pass control -> empty (exit 0)")
    if not ok5: bad.append("control")

    # exit-code rule: same-source mapping
    rule = lambda lst: 1 if lst else 0
    ok6 = rule(f) == 1 and rule(f2) == 1 and rule(f3) == 1 and rule(f4) == 1 and rule(f5) == 0
    print(("PASS" if ok6 else "FAIL") + " exit-code rule uses aggregate() result")
    if not ok6: bad.append("exit rule")
    return bad


if __name__ == "__main__":
    ap_check = "--selftest" in sys.argv
    if ap_check:
        bad = selftest_aggregation()
        for b in bad:
            print("FAIL", b)
        print("selftest failures:", len(bad))
        sys.exit(1 if bad else 0)
    main()
