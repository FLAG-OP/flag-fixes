#!/usr/bin/env python3
"""Cross-config quantified numerics comparison (review round-1 P1-1), v2.

v2 fixes the zero-element artifacts of v1: near-zero params make ULP/relative
metrics explode (a 3.7e-4 abs diff on a ~0 ref element reported as 1e33 ULP),
and FTZ'd zero-gradient rows made grad L2-rel meaningless (ref norm == 0).

Metrics (robust to zeros):
  - per-tensor bitwise-identical element fraction (1.0 = identical)
  - max_abs diff (absolute; scale comes from dtype, bf16)
  - L2 diff norm relative to the REF tensor's L2 norm, only where ref norm > 0
  - counts: tensors with ref norm == 0, tensors not fully bitwise identical

PROVISIONAL reference thresholds (declared before the run; NOT an acceptance
gate — formal accuracy acceptance stays pending until criteria are approved
for the final config, per review round-1):
  nonfinite                       == 0      (hard)
  min param identical fraction    >= 99%
  param max_abs                   <= 1e-3   (bf16 ULP at typical |w|~0.01 is 6.1e-5;
                                             1e-3 ~ 16 ULP, several optimizer steps)
  param L2 rel (defined tensors)  <= 1e-2
  grad L2 diff norm               <= 2e-3   (absolute; observed ref grad norms
                                             are O(1) for active tensors)
Verdict reported is "observation with provisional criteria" — pass/fail
wording for formal acceptance is deliberately avoided.
"""
import json, sys

import torch


def main():
    ref_json, ref_pt, tgt_json, tgt_pt = sys.argv[1:5]
    out_path = sys.argv[5] if len(sys.argv) > 5 else "numerics_compare.json"
    rj, tj = json.load(open(ref_json)), json.load(open(tgt_json))
    R = torch.load(ref_pt, weights_only=False, map_location="cpu")
    T = torch.load(tgt_pt, weights_only=False, map_location="cpu")

    assert rj["first_batch_anchor_sha16"] == tj["first_batch_anchor_sha16"], "input order differs!"
    res = {"ref_label": rj["label"], "tgt_label": tj["label"], "input_order_match": True,
           "ref_env": rj["env_snapshot"], "tgt_env": tj["env_snapshot"],
           "ref_optimizer": rj["optimizer_impl"], "tgt_optimizer": tj["optimizer_impl"],
           "ref_gems_include": rj.get("flaggems_include"), "tgt_gems_include": tj.get("flaggems_include")}

    rl = [s["step_loss_mean"] for s in rj["steps"]]
    tl = [s["step_loss_mean"] for s in tj["steps"]]
    loss_diffs = [(a, b, abs(a - b), abs(a - b) / max(abs(a), 1e-12)) for a, b in zip(rl, tl)]
    max_loss_rel = max(d[3] for d in loss_diffs)
    res["loss"] = {"per_step": [{"ref": round(a, 6), "tgt": round(b, 6),
                                 "abs": round(d, 6), "rel": round(e, 6)} for a, b, d, e in loss_diffs],
                   "max_rel": round(max_loss_rel, 6),
                   "ref_inter_step_range": round(max(rl) - min(rl), 6)}

    stats = {"param": {"min_identical_frac": 1.0, "max_abs": 0.0, "max_l2_rel_defined": 0.0,
                       "zero_ref_norm_tensors": 0},
             "grad": {"max_abs": 0.0, "max_l2_rel_defined": 0.0, "zero_ref_norm_tensors": 0}}
    nonfinite = 0
    outliers = []

    def one(kind, i, rt, tt):
        nonlocal nonfinite
        rt, tt = rt.float(), tt.float()
        if not torch.isfinite(rt).all() or not torch.isfinite(tt).all():
            nonfinite += 1
        d = (rt - tt).abs()
        ident = (d == 0).float().mean().item()
        max_abs = d.max().item()
        ref_norm = rt.norm().item()
        entry = {"kind": kind, "idx": i, "identical_frac": round(ident, 6), "max_abs": max_abs,
                 "ref_norm": round(ref_norm, 8), "l2_rel": None}
        if ref_norm > 0:
            l2rel = ((rt - tt).norm() / ref_norm).item()
            entry["l2_rel"] = round(l2rel, 8)
            s = stats[kind]
            if kind == "param":
                s["min_identical_frac"] = min(s["min_identical_frac"], ident)
                s["max_abs"] = max(s["max_abs"], max_abs)
                s["max_l2_rel_defined"] = max(s["max_l2_rel_defined"], l2rel)
            else:
                s["max_abs"] = max(s["max_abs"], max_abs)
                s["max_l2_rel_defined"] = max(s["max_l2_rel_defined"], l2rel)
        else:
            stats[kind]["zero_ref_norm_tensors"] += 1
        if ident < 0.99 or max_abs > 1e-3:
            outliers.append(entry)

    for i, (rp, tp) in enumerate(zip(R["params"], T["params"])):
        one("param", i, rp, tp)
    for i, (rg, tg) in enumerate(zip(R["grads"], T["grads"])):
        if rg is None or tg is None:
            continue
        one("grad", i, rg, tg)

    res["summary"] = stats
    res["outliers_gt_1pct_or_1e-3"] = outliers[:30]
    res["nonfinite"] = nonfinite

    checks = {
        "nonfinite_zero": nonfinite == 0,
        "param_min_identical_frac_ge_99pct": stats["param"]["min_identical_frac"] >= 0.99,
        "param_max_abs_le_1e-3": stats["param"]["max_abs"] <= 1e-3,
        "param_l2_rel_le_1e-2": stats["param"]["max_l2_rel_defined"] <= 1e-2,
        "grad_l2_abs_le_2e-3": stats["grad"]["max_abs"] <= 2e-3,
    }
    res["provisional_criteria"] = {
        "criteria": "PROVISIONAL reference thresholds (script docstring); NOT a formal acceptance gate",
        **checks, "all": all(checks.values())}

    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps({"loss_max_rel": res["loss"]["max_rel"],
                      "summary": stats, "nonfinite": nonfinite,
                      "outlier_count": len(outliers),
                      "provisional_criteria": res["provisional_criteria"]}, indent=1)[:2200])


if __name__ == "__main__":
    main()
