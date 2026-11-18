#!/usr/bin/env python3
"""Aggregate the real BGE-M3 evidence into the issue-#9 scorecard tables.

Reads:
  results/segments/{native,gelu,gemm,glue,all}_r{1,2,3}.json  (segments harness)
  results/numerics/*.json                                     (trajectories)
  results/gemm/*.json                                         (GEMM env sweep)
Writes:
  results/summary.json + prints markdown tables
"""
import glob, json, os, statistics, sys

EV = "/workspace/evidence"
SEG = sorted(glob.glob(f"{EV}/results-clean/segments/*_r*.json"))
NUM = sorted(glob.glob(f"{EV}/results/numerics/*.json"))
GEMM = sorted(glob.glob(f"{EV}/results-clean/gemm/*.json"))

def agg_segments():
    rows = {}
    for p in SEG:
        name = os.path.basename(p).rsplit("_r", 1)[0]
        rep = int(os.path.basename(p).rsplit("_r", 1)[1].split(".")[0])
        d = json.load(open(p))
        rows.setdefault(name, {})[rep] = d
    out = {}
    for name, reps in rows.items():
        walls = [reps[r]["wall_ms"] for r in sorted(reps)]
        dev = [reps[r]["per_step_ms"]["forward_pure_compute"] + reps[r]["per_step_ms"]["backward"] for r in sorted(reps)]
        tok = [reps[r]["per_step_ms"]["tokenize_seen_inside_forward"] for r in sorted(reps)]
        data = [reps[r]["per_step_ms"]["data"] for r in sorted(reps)]
        fwd = [reps[r]["per_step_ms"]["forward"] for r in sorted(reps)]
        bwd = [reps[r]["per_step_ms"]["backward"] for r in sorted(reps)]
        clip = [reps[r]["per_step_ms"]["clip_step"] for r in sorted(reps)]
        kern = [reps[r].get("kernel_ms_per_step", 0.0) for r in sorted(reps)]
        thr = [reps[r]["texts_per_step"] / (reps[r]["wall_ms"] / 1000) for r in sorted(reps)]
        out[name] = {
            "reps": len(walls),
            "wall_ms_per_rep": [round(w, 1) for w in walls],
            # 3-process statistics: median + range (NOT a step-level p95)
            "wall_ms_p50": round(statistics.median(walls), 1),
            "wall_ms_range": [round(min(walls), 1), round(max(walls), 1)],
            "device_ms_p50": round(statistics.median(dev), 1),
            "forward_ms_median": round(statistics.median(fwd), 1),
            "backward_ms_median": round(statistics.median(bwd), 1),
            "clip_step_ms_median": round(statistics.median(clip), 1),
            "tokenize_ms_p50": round(statistics.median(tok), 1),
            "dataloader_ms_p50": round(statistics.median(data), 2),
            "kernel_ms_p50": round(statistics.median(kern), 1),
            "texts_per_s_p50": round(statistics.median(thr), 1),
        }
    return out

GELU_PAT = ("gelu_highprecision", "gelu_bwd_highprecision")
def kernel_family_table(name_reps):
    """kernel-name -> per-step ms & calls, averaged over profiled reps of a config."""
    agg = {}
    n = 0
    for rep, d in name_reps.items():
        n += 1
        for k, v in d.get("ops", {}).items():
            a = agg.setdefault(k, [0.0, 0])
            a[0] += v["ms"] / max(1, d.get("profiled_steps", 3))
            a[1] += v["calls"] / max(1, d.get("profiled_steps", 3))
    return {k: {"ms_per_step": round(v[0] / max(1, n), 1), "calls_per_step": round(v[1] / max(1, n), 1)} for k, v in agg.items()}

def fam_sum(table, *keys):
    return round(sum(v["ms_per_step"] for k, v in table.items() if any(x in k for x in keys)), 1)

def main():
    seg = agg_segments()
    fams = {}
    for name in ("native", "gelu", "gemm", "glue", "all"):
        reps = {}
        for p in SEG:
            base = os.path.basename(p)
            if base.startswith(name + "_r"):
                r = int(base.rsplit("_r", 1)[1].split(".")[0])
                reps[r] = json.load(open(p))
        if reps:
            fams[name] = kernel_family_table(reps)

    fams_out = {}
    for name, t in fams.items():
        fams_out[name] = {
            "gelu_vendor_ms": fam_sum(t, *GELU_PAT),
            "gelu_triton_ms": fam_sum(t, "gelu_none_kernel", "gelu_backward_none_kernel"),
            "gemm_fc_ms": fam_sum(t, "fc_bf16", "fc_fp16", "xblas"),
            "attention_ms": fam_sum(t, "flash_attn", "flash_attention"),
            "cast_te_ms": fam_sum(t, "cast_te"),
            "transpose_ms": fam_sum(t, "transpose", "sdnn"),
            "tensor_calc_ms": fam_sum(t, "tensor_calc_basic"),
            "layernorm_ms": fam_sum(t, "layer_norm", "ln_"),
        }

    num = {}
    for p in NUM:
        if "compare_" in os.path.basename(p):
            continue
        d = json.load(open(p))
        if "steps" not in d:
            continue
        num[os.path.basename(p)[:-5]] = {
            "losses": [s["step_loss_mean"] for s in d["steps"]],
            "param_final": d["param_sha_16_final"],
            "grad_final": d["grad_sha_16_final"],
            "params_nonfinite": sum(s["params_nonfinite"] for s in d["steps"]),
            "max_mem_GB": d["max_memory_allocated_GB"],
            "env_snapshot": d.get("env_snapshot"),
            "optimizer_impl": d.get("optimizer_impl"),
            "first_batch_anchor_sha16": d.get("first_batch_anchor_sha16"),
            "loss_rel_vs_first_native": None,
        }
    base = num.get("native_r3", {}).get("losses") or num.get("native_r1", {}).get("losses")
    if base:
        for v in num.values():
            a, b = v["losses"], base
            if len(a) == len(b):
                v["loss_rel_vs_first_native"] = round(max(abs(x - y) / max(abs(y), 1e-9) for x, y in zip(a, b)), 6)

    gemm = {}
    for p in GEMM:
        d = json.load(open(p))
        name = os.path.basename(p)[:-5]
        proc = d["results"]["proc"]
        gemm[name] = proc

    summary = {"segments": seg, "families_ms_per_step": fams_out, "numerics": num, "gemm_sweep": gemm}
    json.dump(summary, open(f"{EV}/results-clean/summary.json", "w"), indent=1)

    print("== segments (3 indep processes, median wall) ==")
    base_wall = seg.get("native", {}).get("wall_ms_p50")
    for name in ("native", "gemm", "gelu", "glue", "all"):
        if name not in seg:
            continue
        s = seg[name]
        delta = f"{(s['wall_ms_p50']/base_wall-1)*100:+.1f}%" if base_wall and name != "native" else "-"
        print(f"{name:8} wall_p50={s['wall_ms_p50']:9.1f}ms ({delta:>7}) kernel={s['kernel_ms_p50']:8.1f} "
              f"dev={s['device_ms_p50']:8.1f} tok={s['tokenize_ms_p50']:6.1f} thr={s['texts_per_s_p50']:6.1f} texts/s "
              f"reps={s['reps']}")

    print("\n== kernel families (ms/step) ==")
    hdr = ["config", "gelu_vend", "gelu_tri", "gemm", "attn", "cast_te", "transp", "tcalc", "ln"]
    print(("%-8s" + "%10s" * 8) % tuple(hdr))
    for name, f in fams_out.items():
        print(("%-8s" + "%10.1f" * 8) % (name, f["gelu_vendor_ms"], f["gelu_triton_ms"], f["gemm_fc_ms"],
                                         f["attention_ms"], f["cast_te_ms"], f["transpose_ms"],
                                         f["tensor_calc_ms"], f["layernorm_ms"]))

    print("\n== numerics ==")
    for name, v in num.items():
        print(f"{name:12} final_loss={v['losses'][-1]:.6f} relmax_vs_native_r1={v['loss_rel_vs_first_native']} "
              f"nonfinite={v['params_nonfinite']} mem={v['max_mem_GB']}GB param={v['param_final']}")

    print("\n== gemm sweep ==")
    for name, v in gemm.items():
        shapes = " ".join(f"{k.replace('/', '_')}={d['ms']}ms/{d['tflops']}TF"
                          for k, d in v["shapes"].items())
        print(f"{name:22} {shapes}")

if __name__ == "__main__":
    main()
