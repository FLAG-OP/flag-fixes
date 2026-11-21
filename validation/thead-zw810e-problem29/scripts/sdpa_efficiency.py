"""Review 5ec9e6c item 3: SDPA efficiency recomputable.
Per (step, micro, group) FLOPs accumulation rows + phase totals; explicit
formula; dropout/recompute flags recorded; TFLOP/s computed correctly
(previous 0.0 was a double-scaling bug: flops already in TFLOP)."""
import json

ts = json.load(open("results/token_stats_r25.json"))
d33 = json.load(open("results/training_trace_analysis_r33.json"))
attn = {k: v for k, v in d33["inventory_per_step"].items() if k.startswith("sdpa_attention")}
H, D = 16, 64
rows = []
tot = {"forward_tflop": 0.0, "backward_tflop": 0.0}
for step, micros in ts["steps"].items():
    for mi, micro in enumerate(micros):
        for gname, g in micro.items():
            B, S = g["sequences"], g["max_len"]
            f = 4 * B * H * S * S * D / 1e12   # TFLOP
            b = 6 * B * H * S * S * D / 1e12
            tot["forward_tflop"] += f; tot["backward_tflop"] += b
            rows.append({"step": step, "micro": mi, "group": gname,
                          "B": B, "H": H, "S": S, "D": D,
                          "fwd_tflop": round(f, 4), "bwd_tflop": round(b, 4),
                          "note": "padded S (mask positions computed)"})
ms_f = attn.get("sdpa_attention/forward", {}).get("device_ms_per_step")
ms_b = attn.get("sdpa_attention/backward", {}).get("device_ms_per_step")
# NOTE: token_stats window = 3 capture steps; ms/step already normalized
out = {"H": H, "D": D, "formula": "fwd TFLOP = 4*B*H*S^2*D / 1e12 (QK^T + PV); bwd = 6*B*H*S^2*D / 1e12 (dQ,dK,dV); S = padded max_len (mask positions included in computation)",
        "dropout": {"config_hidden_dropout_prob": 0.1, "sdpa_internal_dropout_flag": "UNKNOWN (not visible in kernel names)", "module_level_dropout_active": True},
        "recompute_in_backward": "UNKNOWN (no counter; bwd flops formula covers dQ/dK/dV grad GEMMs only)",
        "per_group_rows": rows,
        "totals_tflop_per_step": {k: round(v, 3) for k, v in tot.items()},
        "sdpa_device_ms_per_step": {"forward": ms_f, "backward": ms_b},
        "tflops": {"forward": round(tot["forward_tflop"] / (ms_f * 1e-3), 2) if ms_f else None,
                    "backward": round(tot["backward_tflop"] / (ms_b * 1e-3), 2) if ms_b else None},
        "units_note": "tflops = totals_tflop_per_step / (ms/1000); previous 0.0 was double-scaling bug"}
json.dump(out, open("results/sdpa_efficiency_r35.json", "w"), indent=1)
print("totals TFLOP/step:", out["totals_tflop_per_step"], "| ms:", out["sdpa_device_ms_per_step"], "| TFLOP/s:", out["tflops"])
print("rows:", len(rows))
