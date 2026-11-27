"""Review cf0cc15 item 1: SDPA FLOPs with FULL BGE-M3 layer count (24 layers).
Per step: 4 micros x 5 groups x 24 layers = 480 forward SDPA calls.
Per group padded S:
  forward  = 2 matmuls (QK^T + PV)  = 4*B*H*S^2*D per layer
  backward = 4 matmuls (dV,dP,dQ,dK) = 8*B*H*S^2*D per layer
  (softmax/dropout/possible-recompute listed separately, not in FLOPs)
Three-step mean -> TFLOP/step; same-window device ms -> TFLOP/s."""
import json

meta = json.load(open("results/training_trace_r36_meta.json"))
fp = meta["batch_fingerprints"]["fingerprints"]
d36 = json.load(open("results/training_trace_analysis_r36.json"))
attn = {k: v for k, v in d36["inventory_per_step"].items() if k.startswith("sdpa_attention")}
H, D, LAYERS = 16, 64, 24

rows = []
per_step_tflop = {}
for step_key in sorted(fp.keys()):
    tf = {"forward": 0.0, "backward": 0.0}
    micros = fp[step_key]
    for mi, micro in enumerate(micros):
        for gname, g in micro.items():
            B, S = g["sequences"], g["max_len"]
            f = LAYERS * 4 * B * H * S * S * D / 1e12   # 2 matmuls x 24 layers
            b = LAYERS * 8 * B * H * S * S * D / 1e12   # 4 matmuls x 24 layers
            tf["forward"] += f; tf["backward"] += b
            rows.append({"step": step_key, "micro": mi, "group": gname,
                          "B": B, "H": H, "S": S, "D": D, "layers": LAYERS,
                          "fwd_tflop": round(f, 4), "bwd_tflop": round(b, 4),
                          "note": "padded S; matrix-multiply FLOPs only (softmax/dropout/recompute separate)"})
    per_step_tflop[step_key] = {k: round(v, 4) for k, v in tf.items()}
mean_f = sum(v["forward"] for v in per_step_tflop.values()) / len(per_step_tflop)
mean_b = sum(v["backward"] for v in per_step_tflop.values()) / len(per_step_tflop)
ms_f = attn.get("sdpa_attention/forward", {}).get("device_ms_per_step")
ms_b = attn.get("sdpa_attention/backward", {}).get("device_ms_per_step")
kern_f = attn.get("sdpa_attention/forward", {}).get("count_per_step")
kern_b = attn.get("sdpa_attention/backward", {}).get("count_per_step")
out = {"H": H, "D": D, "layers": LAYERS, "groups_per_micro": 5, "micros_per_step": 4,
        "sdpa_calls_per_step": {"forward": LAYERS * 5 * 4, "backward": None},
        "formula": "fwd = 24 layers x 4*B*H*S^2*D (2 matmuls: QK^T, PV); bwd = 24 layers x 8*B*H*S^2*D (4 matmuls: dV,dP,dQ,dK); S = padded max_len",
        "scope_note": "TFLOP/s = model matrix-multiply FLOPs / attention-stage device time; softmax, dropout, possible recompute NOT in FLOPs (listed separately)",
        "units": "FLOPs PER STEP from r36 fingerprints, averaged over 3 steps; device ms per-step-normalized",
        "dropout": {"config_hidden_dropout_prob": 0.1,
                     "module_level_dropout_active": True,
                     "sdpa_dropout_variant": "kernel names contain _dropout (variant confirmed via trace); actual probability and recompute require call-args/implementation evidence - UNKNOWN"},
        "recompute_in_backward": "UNKNOWN",
        "per_step_tflop": per_step_tflop,
        "per_group_rows": rows,
        "sdpa_kernels_per_step": {"forward": kern_f, "backward": kern_b},
        "sdpa_device_ms_per_step": {"forward": ms_f, "backward": ms_b},
        "tflops": {"forward": round(mean_f / (ms_f * 1e-3), 2) if ms_f else None,
                    "backward": round(mean_b / (ms_b * 1e-3), 2) if ms_b else None}}
json.dump(out, open("results/sdpa_efficiency_r36.json", "w"), indent=1)
print("per-step TFLOP:", per_step_tflop)
print("mean:", round(mean_f, 3), round(mean_b, 3), "TFLOP/step -> TFLOP/s:", out["tflops"])
