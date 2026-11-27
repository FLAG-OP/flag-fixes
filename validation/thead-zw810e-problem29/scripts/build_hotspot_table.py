"""Rebuild the hotspot table from linear_capture_r28.json on CPU (review of
cca10d4 P2): group per-shape rows by MODULE CLASS (the old table keyed on
input shape alone merged QKV with FFN-up). Derived (M,K,N) per Linear
semantics; input dtype (pre-autocast) vs GEMM operand dtype (autocast)
distinguished. Reconcile equality asserted, total ms checked."""
import json, collections, sys

d = json.load(open("results/linear_capture_r28.json"))
STEPS = d["capture_steps"]

def classify(name):
    if "attention.self" in name and any(t in name for t in ("query", "key", "value")):
        return "attn_qkv"
    if "attention.output.dense" in name:
        return "attn_out"
    if "intermediate.dense" in name:
        return "ffn_up"
    if ".output.dense" in name and "encoder.layer" in name:
        return "ffn_down"
    if "pooler" in name:
        return "pooler"
    return "other"

# per (class, input shape, stride, dtype, w_shape) aggregation
groups = collections.defaultdict(lambda: {"calls_sum": 0, "ms_sum": 0.0, "modules": set(),
                                          "stride": None, "dtype": None, "w_shape": None, "w_stride": None})
mod_calls_total = 0
mod_ms_total = 0.0
for name, v in d["modules"].items():
    cls = classify(name)
    mod_calls_total += v["calls_per_capture_step"]
    mod_ms_total += v["fwd_ms_per_step"]
    for r in v["per_shape"]:
        key = (cls, tuple(r["shape"]), tuple(r["stride"]), r["dtype"], tuple(v["w_shape"]))
        g = groups[key]
        g["calls_sum"] += r["calls_sum"]
        g["ms_sum"] += r["fwd_ms_per_step"] * STEPS
        g["modules"].add(name)
        g["stride"] = r["stride"]; g["dtype"] = r["dtype"]
        g["w_shape"] = tuple(v["w_shape"]); g["w_stride"] = v["w_stride"]

# reconcile assertions (like-for-like units: both over ALL capture steps)
grp_calls = sum(g["calls_sum"] for g in groups.values())
grp_ms_total = sum(g["ms_sum"] for g in groups.values())
mod_calls_all_steps = mod_calls_total * STEPS
mod_ms_all_steps = mod_ms_total * STEPS
assert grp_calls == mod_calls_all_steps, f"calls mismatch {grp_calls} vs {mod_calls_all_steps}"
assert abs(grp_ms_total - mod_ms_all_steps) < 0.5, f"ms mismatch {grp_ms_total} vs {mod_ms_all_steps}"
grp_ms = grp_ms_total / STEPS

rows = []
for (cls, shp, stride, dtype, w), g in groups.items():
    if len(shp) == 3:
        B, S, K_in = shp; M = B * S; K = K_in
    else:
        M, K = shp
    N = w[0]
    assert w[1] == K, f"K mismatch {w} vs input {shp} in {cls}"
    rows.append({"class": cls, "in_shape": list(shp), "in_stride": list(stride),
                 "in_dtype": dtype, "w_shape": list(w), "w_stride": list(g["w_stride"]),
                 "gemm_dtype": "bf16 (autocast operand; Linear input pre-autocast)",
                 "M": M, "K": K, "N": N,
                 "calls_per_step": round(g["calls_sum"] / STEPS, 2),
                 "fwd_ms_per_step": round(g["ms_sum"] / STEPS, 3),
                 "modules": len(g["modules"])})
rows.sort(key=lambda r: -r["fwd_ms_per_step"])

out = {"note": "hotspot table rebuilt per module CLASS (input-shape-only key merged QKV with FFN-up - fixed); M=B*S flattened; (M,K,N) derived from Linear semantics; in_dtype = pre-autocast input dtype recorded by wrapper; gemm operand dtype = bf16 via autocast",
       "capture_source": "results/linear_capture_r28.json",
       "capture_steps": STEPS,
       "reconcile": {"calls": {"per_class_sum_all_steps": grp_calls, "module_total_all_steps": mod_calls_all_steps},
                     "ms_all_steps": {"per_class_sum": round(grp_ms_total, 3), "module_total": round(mod_ms_all_steps, 3)},
                     "asserted": True},
       "rows": rows}
json.dump(out, open("results/linear_hotspots_r29.json", "w"), indent=1)
print(f"{'class':<10}{'(M,K,N)':<22}{'calls/step':>10}{'ms/step':>9}{'modules':>8}")
for r in rows[:10]:
    print(f"{r['class']:<10}{str((r['M'],r['K'],r['N'])):<22}{r['calls_per_step']:>10}{r['fwd_ms_per_step']:>9}{r['modules']:>8}")
tot = sum(r["fwd_ms_per_step"] for r in rows)
print(f"TOTAL ms/step {tot:.3f} == module total {mod_ms_total:.3f}; calls {grp_calls} == {mod_calls_total}")
