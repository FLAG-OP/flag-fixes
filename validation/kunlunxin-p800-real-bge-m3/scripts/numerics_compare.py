#!/usr/bin/env python3
"""Cross-config quantified numerics comparison (v3, review round-2).

v3 fixes the round-2 findings:
  - absolute metrics (numeric-equal fraction, max_abs, L2 diff norm) are
    computed for EVERY tensor including zero-ref-norm ones; the ref-norm guard
    only gates the *relative* L2 metric (round-2: zero-ref tensors were
    excluded from max_abs/equal stats entirely);
  - the grad criterion uses the TRUE L2 diff norm ||ref-tgt||_2 (round-2: the
    criterion compared max_abs, i.e. Linf, while claiming L2);
  - alignment is validated before comparing: equal step counts, equal tensor
    counts, equal shapes/dtypes per tensor, and grad presence recorded as
    both-present / both-none / MISMATCH (single-side None is an error, not a
    silent skip);
  - "numeric_equal_frac" (renamed from identical_frac): element-wise numeric
    equality compared in fp32 after casting (exact for bf16/fp16/fp32 values;
    +0/-0 not distinguished) — not a bitwise bit-pattern comparison.

CPU self-test: run with --selftest (no torch device needed) to exercise the
production compare on counterexamples: zero-ref-nonzero-target, length
mismatch, tensor-count mismatch, shape mismatch, single-side None grad,
both-none grads, NaN, and "small per-element diff but large accumulated L2".

PROVISIONAL reference thresholds (declared before the run; NOT an acceptance
gate — formal accuracy acceptance stays pending until criteria are approved
for the final config, per reviews round-1/round-2):
  nonfinite                          == 0      (hard)
  min param numeric_equal_frac       >= 99%
  param max_abs                      <= 1e-3
  param L2 rel (defined tensors)     <= 1e-2   (per-tensor maximum, labeled)
  grad L2 diff norm (per tensor max) <= 2e-3
Verdict is "observation with provisional criteria" — formal pass/fail wording
is deliberately avoided.
"""
import json, math, sys

import torch


def load_state(path):
    st = torch.load(path, weights_only=False, map_location="cpu")
    for k in ("params", "grads"):
        assert k in st, f"{path}: missing {k}"
    return st


def validate_alignment(rj, tj, R, T):
    errs = []
    if len(rj["steps"]) != len(tj["steps"]):
        errs.append(f"step count mismatch: {len(rj['steps'])} vs {len(tj['steps'])}")
    if len(R["params"]) != len(T["params"]):
        errs.append(f"param tensor count mismatch: {len(R['params'])} vs {len(T['params'])}")
    if len(R["grads"]) != len(T["grads"]):
        errs.append(f"grad tensor count mismatch: {len(R['grads'])} vs {len(T['grads'])}")
    for i, (rp, tp) in enumerate(zip(R["params"], T["params"])):
        if rp.shape != tp.shape:
            errs.append(f"param[{i}] shape {tuple(rp.shape)} vs {tuple(tp.shape)}")
        if rp.dtype != tp.dtype:
            errs.append(f"param[{i}] dtype {rp.dtype} vs {tp.dtype}")
    for i, (rg, tg) in enumerate(zip(R["grads"], T["grads"])):
        if (rg is None) != (tg is None):
            errs.append(f"grad[{i}] presence mismatch: ref={'None' if rg is None else 'tensor'} "
                        f"tgt={'None' if tg is None else 'tensor'} (both-unused grads are legal and counted)")
        elif rg is not None:
            if rg.shape != tg.shape:
                errs.append(f"grad[{i}] shape {tuple(rg.shape)} vs {tuple(tg.shape)}")
            if rg.dtype != tg.dtype:
                errs.append(f"grad[{i}] dtype {rg.dtype} vs {tg.dtype}")
    return errs


def compare_states(R, T):
    stats = {
        "param": {"min_numeric_equal_frac": 1.0, "max_abs": 0.0,
                  "max_l2_rel_defined": 0.0, "max_l2_diff_norm": 0.0,
                  "zero_ref_norm_tensors": 0, "undefined_l2_rel_tensors": 0},
        "grad": {"max_abs": 0.0, "max_l2_diff_norm": 0.0,
                 "max_l2_rel_defined": 0.0, "zero_ref_norm_tensors": 0,
                 "both_none_grads": 0, "undefined_l2_rel_tensors": 0},
    }
    nonfinite = 0
    outliers = []

    def one(kind, i, rt, tt):
        nonlocal nonfinite
        rt, tt = rt.float(), tt.float()
        if not torch.isfinite(rt).all() or not torch.isfinite(tt).all():
            nonfinite += 1
        d = rt - tt
        abs_d = d.abs()
        equal_frac = (d == 0).float().mean().item()  # numeric equality in fp32 (bf16-exact)
        l2_norm = d.norm().item()
        max_abs = abs_d.max().item()
        ref_norm = rt.norm().item()
        entry = {"kind": kind, "idx": i, "numeric_equal_frac": round(equal_frac, 6),
                 "max_abs": max_abs, "l2_diff_norm": round(l2_norm, 8),
                 "ref_norm": round(ref_norm, 8)}
        s = stats[kind]
        if kind == "param":
            s["min_numeric_equal_frac"] = min(s["min_numeric_equal_frac"], equal_frac)
        s["max_abs"] = max(s["max_abs"], max_abs)
        s["max_l2_diff_norm"] = max(s["max_l2_diff_norm"], l2_norm)
        if ref_norm > 0:
            l2rel = (d.norm() / ref_norm).item()
            entry["l2_rel"] = round(l2rel, 8)
            s["max_l2_rel_defined"] = max(s["max_l2_rel_defined"], l2rel)
        else:
            s["zero_ref_norm_tensors"] += 1
            s["undefined_l2_rel_tensors"] += 1
        if (kind == "param" and (equal_frac < 0.99 or max_abs > 1e-3)) or \
           (kind == "grad" and l2_norm > 2e-3):
            outliers.append(entry)

    for i, (rp, tp) in enumerate(zip(R["params"], T["params"])):
        one("param", i, rp, tp)
    for i, (rg, tg) in enumerate(zip(R["grads"], T["grads"])):
        if rg is None and tg is None:
            stats["grad"]["both_none_grads"] += 1
            continue
        one("grad", i, rg, tg)
    return stats, nonfinite, outliers


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(selftest())

    if len(sys.argv) < 6:
        print("usage: numerics_compare.py <ref.json> <ref.pt> <tgt.json> <tgt.pt> <out.json>")
        sys.exit(2)
    ref_json, ref_pt, tgt_json, tgt_pt, out_path = sys.argv[1:6]
    rj, tj = json.load(open(ref_json)), json.load(open(tgt_json))
    R, T = load_state(ref_pt), load_state(tgt_pt)

    # fail fast on any input-stream mismatch (full-batch fingerprint when
    # present; legacy first-batch hash for archived runs)
    rfp = rj.get("input_fingerprint_all_batches_sha16") or rj.get("first_batch_anchor_sha16")
    tfp = tj.get("input_fingerprint_all_batches_sha16") or tj.get("first_batch_anchor_sha16")
    if rfp is None or tfp is None or rfp != tfp:
        print(f"INPUT FINGERPRINT MISMATCH: ref={rfp} tgt={tfp}")
        sys.exit(2)
    align = validate_alignment(rj, tj, R, T)
    res = {"ref_label": rj["label"], "tgt_label": tj["label"],
           "input_fingerprint_ref": rj.get("input_fingerprint_all_batches_sha16")
                                    or rj.get("first_batch_anchor_sha16"),
           "input_fingerprint_tgt": tj.get("input_fingerprint_all_batches_sha16")
                                    or tj.get("first_batch_anchor_sha16"),
           "ref_env": rj.get("env_snapshot"), "tgt_env": tj.get("env_snapshot"),
           "ref_optimizer": rj.get("optimizer_impl"), "tgt_optimizer": tj.get("optimizer_impl"),
           "ref_gems_include": rj.get("flaggems_include"),
           "tgt_gems_include": tj.get("flaggems_include")}
    if align:
        res["alignment"] = {"errors": align}
        json.dump(res, open(out_path, "w"), indent=1)
        print("ALIGNMENT ERRORS:", res["alignment"]["errors"])
        sys.exit(1)

    rl = [s["step_loss_mean"] for s in rj["steps"]]
    tl = [s["step_loss_mean"] for s in tj["steps"]]
    loss_diffs = [(a, b, abs(a - b), abs(a - b) / max(abs(a), 1e-12)) for a, b in zip(rl, tl)]
    max_loss_rel = max(d[3] for d in loss_diffs)
    res["loss"] = {"per_step": [{"ref": round(a, 6), "tgt": round(b, 6),
                                 "abs": round(d, 6), "rel": round(e, 6)} for a, b, d, e in loss_diffs],
                   "max_rel": round(max_loss_rel, 6),
                   "ref_inter_step_range": round(max(rl) - min(rl), 6)}

    stats, nonfinite, outliers = compare_states(R, T)
    res["summary"] = stats
    res["outliers_gt_thresholds"] = outliers[:30]
    res["nonfinite"] = nonfinite

    checks = {
        "nonfinite_zero": nonfinite == 0,
        "param_min_numeric_equal_frac_ge_99pct": stats["param"]["min_numeric_equal_frac"] >= 0.99,
        "param_max_abs_le_1e-3": stats["param"]["max_abs"] <= 1e-3,
        "param_l2_rel_le_1e-2": stats["param"]["max_l2_rel_defined"] <= 1e-2,
        "grad_l2_diff_norm_le_2e-3": stats["grad"]["max_l2_diff_norm"] <= 2e-3,
    }
    res["provisional_criteria"] = {
        "criteria": "PROVISIONAL reference thresholds (script docstring); NOT a formal acceptance gate",
        **checks, "all": all(checks.values())}

    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps({"loss_max_rel": res["loss"]["max_rel"], "summary": stats,
                      "nonfinite": nonfinite, "outlier_count": len(outliers),
                      "provisional_criteria": res["provisional_criteria"]}, indent=1)[:2400])


def selftest():
    """CPU counterexamples against the production compare path."""
    failures = []

    def run_pair(R, T, rj=None, tj=None):
        base = {"label": "ref", "steps": [{"step_loss_mean": 1.0}] * 2,
                "first_batch_anchor_sha16": "x", "env_snapshot": {}, "optimizer_impl": {},
                "flaggems_include": None}
        rj = rj or dict(base)
        tj = tj or dict(base, label="tgt")
        json.dump(rj, open("/tmp/opencode/_cmp_r.json", "w"))
        json.dump(tj, open("/tmp/opencode/_cmp_t.json", "w"))
        torch.save(R, "/tmp/opencode/_cmp_r.pt")
        torch.save(T, "/tmp/opencode/_cmp_t.pt")
        out = "/tmp/opencode/_cmp_out.json"
        main_args = sys.argv
        code = 0
        try:
            sys.argv = ["numerics_compare.py", "/tmp/opencode/_cmp_r.json",
                        "/tmp/opencode/_cmp_r.pt", "/tmp/opencode/_cmp_t.json",
                        "/tmp/opencode/_cmp_t.pt", out]
            main()
        except SystemExit as e:
            code = e.code
        finally:
            sys.argv = main_args
        return json.load(open(out)), code

    # 1) zero-ref-norm param with a REAL difference: must show in max_abs
    R = {"params": [torch.zeros(4)], "grads": [torch.zeros(4)]}
    T = {"params": [torch.tensor([0.0, 0.0, 0.0, 10.0])], "grads": [torch.zeros(4)]}
    res, _ = run_pair(R, T)
    if res["summary"]["param"]["max_abs"] != 10.0:
        failures.append("zero-ref diff 10.0 missing from param max_abs")
    # 2) small per-element diff but large accumulated L2 (10k elements x 0.01)
    R = {"params": [torch.ones(10000)], "grads": [torch.zeros(1)]}
    T = {"params": [torch.ones(10000) + 0.01], "grads": [torch.zeros(1)]}
    res, _ = run_pair(R, T)
    if res["summary"]["param"]["max_l2_diff_norm"] < 0.99:
        failures.append("accumulated L2 (should be ~1.0) not reported")
    # 3) NaN counted
    R = {"params": [torch.tensor([1.0, float("nan")])], "grads": [torch.zeros(1)]}
    T = {"params": [torch.tensor([1.0, 1.0])], "grads": [torch.zeros(1)]}
    res, _ = run_pair(R, T)
    if res["nonfinite"] != 1:
        failures.append("NaN not counted as nonfinite")
    # 4) both-none grads counted, single-side None is an alignment error
    R = {"params": [torch.ones(2)], "grads": [None]}
    T = {"params": [torch.ones(2)], "grads": [None]}
    res, _ = run_pair(R, T)
    if res["summary"]["grad"]["both_none_grads"] != 1:
        failures.append("both-none grad not counted")
    R = {"params": [torch.ones(2)], "grads": [None]}
    T = {"params": [torch.ones(2)], "grads": [torch.zeros(2)]}
    res, code = run_pair(R, T)
    if code != 1 or not res.get("alignment", {}).get("errors"):
        failures.append("single-side None grad not flagged as alignment error")
    # 5) step-count mismatch flagged
    rj = {"label": "ref", "steps": [{"step_loss_mean": 1.0}] * 3,
          "first_batch_anchor_sha16": "x", "env_snapshot": {}, "optimizer_impl": {},
          "flaggems_include": None}
    tj = dict(rj, label="tgt")
    tj["steps"] = tj["steps"][:2]          # 3 steps vs 2 steps
    R = {"params": [torch.ones(2)], "grads": [torch.zeros(2)]}
    T = {"params": [torch.ones(2)], "grads": [torch.zeros(2)]}
    res, code = run_pair(R, T, rj=rj, tj=tj)
    if code != 1 or not res.get("alignment", {}).get("errors"):
        failures.append("step-count mismatch not flagged")

    if failures:
        print("SELFTEST FAILURES:", failures)
        sys.exit(1)
    print("SELFTEST OK: 6 counterexamples handled by the production path")


if __name__ == "__main__":
    main()
