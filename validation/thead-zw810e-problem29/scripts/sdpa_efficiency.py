"""Review 8190910 item 1: SDPA efficiency with correct per-step units.
FLOPs computed PER STEP from the r35 trace-pass fingerprints, then averaged
(matching per-step-normalized device ms from the r35 trace analysis).
Previous version summed 3 steps of FLOPs against per-step ms -> ~3x overstated."""
import json

meta = json.load(open("results/training_trace_r36_meta.json"))
fp = meta["batch_fingerprints"]["fingerprints"]
d35 = json.load(open("results/training_trace_analysis_r36.json"))
attn = {k: v for k, v in d35["inventory_per_step"].items() if k.startswith("sdpa_attention")}
H, D = 16, 64

rows = []
per_step_tflop = {}
for step_key in sorted(fp.keys()):
    tf = {"forward": 0.0, "backward": 0.0}
    micros = fp[step_key]
    for mi, micro in enumerate(micros):
        for gname, g in micro.items():
            B, S = g["sequences"], g["max_len"]
            f = 4 * B * H * S * S * D / 1e12
            b = 6 * B * H * S * S * D / 1e12
            tf["forward"] += f; tf["backward"] += b
            rows.append({"step": step_key, "micro": mi, "group": gname,
                          "B": B, "H": H, "S": S, "D": D,
                          "fwd_tflop": round(f, 4), "bwd_tflop": round(b, 4),
                          "note": "padded S (mask positions computed)"})
    per_step_tflop[step_key] = {k: round(v, 4) for k, v in tf.items()}
mean_f = sum(v["forward"] for v in per_step_tflop.values()) / len(per_step_tflop)
mean_b = sum(v["backward"] for v in per_step_tflop.values()) / len(per_step_tflop)
ms_f = attn.get("sdpa_attention/forward", {}).get("device_ms_per_step")
ms_b = attn.get("sdpa_attention/backward", {}).get("device_ms_per_step")
kern_f = attn.get("sdpa_attention/forward", {}).get("count_per_step")
kern_b = attn.get("sdpa_attention/backward", {}).get("count_per_step")
out = {"H": H, "D": D,
        "formula": "fwd TFLOP = 4*B*H*S^2*D / 1e12 (QK^T + PV); bwd = 6*B*H*S^2*D / 1e12 (dQ,dK,dV); S = padded max_len",
        "units": "FLOPs computed PER STEP from r35 fingerprints then averaged over steps; device ms are per-step-normalized trace analysis - SAME units (review 8190910 item 1 correction)",
        "dropout": {"config_hidden_dropout_prob": 0.1,
                     "module_level_dropout_active": True,
                     "sdpa_internal_dropout_flag": "UNKNOWN (not visible in kernel names)"},
        "recompute_in_backward": "UNKNOWN (no counter; formula covers dQ/dK/dV grads only)",
        "per_step_tflop": per_step_tflop,
        "per_group_rows": rows,
        "sdpa_kernels_per_step": {"forward": kern_f, "backward": kern_b},
        "sdpa_device_ms_per_step": {"forward": ms_f, "backward": ms_b},
        "tflops": {"forward": round(mean_f / (ms_f * 1e-3), 2) if ms_f else None,
                    "backward": round(mean_b / (ms_b * 1e-3), 2) if ms_b else None}}
json.dump(out, open("results/sdpa_efficiency_r36.json", "w"), indent=1)
print("per-step TFLOP:", per_step_tflop)
print("mean TFLOP/step:", round(mean_f,3), round(mean_b,3), "-> TFLOP/s:",
      out["tflops"], "| kernels/step:", kern_f, kern_b)
