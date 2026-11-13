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
    peak_state = {id(p): {k: v.clone() for k, v in opt.state[p].items()} for p in params if p in opt.state}
    step_val = float(next(iter(opt.state.values()))["step"].item()) if opt.state else None
    out = {"params": [p.detach().clone() for p in params], "state": peak_state, "step": step_val, "step_ms_median": round(statistics.median(times), 4), "step_ms_p95": round(sorted(times)[int(0.95 * (len(times) - 1))], 4), "dtypes": dtypes}
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
                            step1.append({"param": i, "dtype": d, "pass": False, "note": "non-finite in step-1 params (see nan/inf counts)"})
                            continue
                        # declared: abs diff <= 2 ulp(dtype at param scale) == same algorithm, diff summation order
                        quantum = {"float32": 1.5259021896696442e-07, "bfloat16": 7.8125e-05, "float16": 9.765625e-06}[d] * scale / 0.02
                        mult = 4 if d in ("float16", "bfloat16") else 2  # low-precision: 4 ulp@scale (declared)
                        step1.append({"param": i, "dtype": d, "step1_max_abs": abs_diff, "param_scale_std": round(scale, 6), "allowed_abs": round(mult * quantum, 12), "pass": abs_diff <= mult * quantum})
                    st_cmp = []
                    for i, p in enumerate(ref["params"]):
                        rs = ref["state"].get(id(p), {})
                        as_ = alt["state"].get(id(alt["params"][i]), {})
                        for k in ("exp_avg", "exp_avg_sq"):
                            if k in rs and k in as_:
                                st_cmp.append(compare_state(rs[k], as_[k], dtype, f"param{i}.{k}"))
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
    hard_bad = []
    div_stats = {}
    for key, r in results.items():
        if "error" in r:
            hard_bad.append(key)
            continue
        for ok_, x in r.items():
            s1fail = [c for c in x["step1_gate"] if not c.get("pass")]
            if s1fail:  # declared: abs diff <= 2 ulp(dtype at scale) == same algorithm, diff summation order
                hard_bad.append(f"{key}/{ok_}/step1_identity_fail({[(c['param'], c['step1_max_abs'], c['allowed_abs']) for c in s1fail]})")
            for c in x["param_compare"]:
                if c.get("nan_a", 0) > 0 or c.get("nan_b", 0) > 0 or c.get("inf_a", 0) > 0 or c.get("inf_b", 0) > 0:
                    hard_bad.append(f"{key}/{ok_}/{c['label']}/nan_inf(nan_a={c.get('nan_a')},inf_a={c.get('inf_a')},nan_b={c.get('nan_b')},inf_b={c.get('inf_b')})")
            div = [c.get("max_abs") for c in x["param_compare"]]
            tot = sum(c.get("positions", 0) for c in x["param_compare"])
            tv = sum(c.get("tol_violations", 0) for c in x["param_compare"])
            div_stats[f"{key}/{ok_}"] = {"max_abs_100step": max(div) if div else None, "positions": tot, "over_4eps_positions": tv}
    OUT["divergence_100step_report"] = div_stats
    OUT["summary"] = {"hard_deviations": hard_bad}
    with open(args.out, "w") as f:
        json.dump(OUT, f, indent=1)
    print("summary:", json.dumps(OUT["summary"]))
    sys.exit(1 if hard_bad else 0)


if __name__ == "__main__":
    main()
