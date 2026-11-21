"""Parse chrome trace from capture_training_trace.py (review 2026-11-21):
- kernel inventory (name/count/device ms/stream)
- phase attribution via correlation -> cpu_op -> rf marker containment
- critical path: union of kernel busy intervals per step window
- GEMM efficiency vs 139.2 TFLOP/s peak (FLOPs from measured per-shape calls)
- HBM logical-byte estimate vs 2765 GB/s (marked derived; counters unavailable)"""
import argparse, bisect, collections, gzip, json, statistics

def classify(name):
    n = name.lower()
    if n.startswith("gemm_ktype") or "gemm" in n[:24]: return "gemm"
    if "gelu" in n: return "gelu"
    if "attention" in n or "fmha" in n or "sdpa" in n or "flash" in n: return "sdpa_attention"
    if "layernorm" in n or "layer_norm" in n: return "layernorm"
    if "reduce" in n or "reduction" in n: return "reduction"
    if "memcpy" in n or "copy" in n[:12]: return "copy"
    if "adamw" in n or "adam" in n or "foreach" in n or "lars" in n: return "optimizer"
    if "elementwise" in n or "fillfunc" in n or "indexfunc" in n or "catfunc" in n: return "elementwise"
    if "norm" in n: return "norm"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--shapes", default="results/linear_capture_r28.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with gzip.open(a.trace, "rt") as f:
        t = json.load(f)
    evs = t["traceEvents"]

    kernels = [e for e in evs if e.get("cat") == "kernel"]
    cpu_ops = [e for e in evs if e.get("cat") == "cpu_op"]
    markers = [e for e in evs if e.get("name", "").startswith("rf_s")]

    # marker ranges sorted by ts
    markers.sort(key=lambda e: e["ts"])
    m_ts = [e["ts"] for e in markers]
    def marker_of(ts):
        i = bisect.bisect_right(m_ts, ts) - 1
        if i < 0: return "outside"
        m = markers[i]
        return m["name"] if ts <= m["ts"] + m.get("dur", 0) else "between"

    # attribution: ac2g 's' flow events carry flow id == kernel args.correlation.
    # REVIEW CORRECTION (6ba3bfd): ac2g timestamps are HOST SUBMISSION times
    # (not device execution); backward kernels may execute after their marker
    # drain asynchronously AFTER their rf marker window closed (async queue).
    # Attribution therefore uses the NEAREST PRECEDING marker START (submission
    # phase), which covers every kernel; the device-vs-CPU lag is reported
    # separately as cross-step overlap (review requirement: overlap via interval
    # union).
    corr2marker = {}
    for e in evs:
        if e.get("cat") == "ac2g" and e.get("ph") == "s":
            cid = e.get("id")
            if cid not in corr2marker:
                i = bisect.bisect_right(m_ts, e["ts"]) - 1
                corr2marker[cid] = markers[i]["name"] if i >= 0 else "before_first_marker"
    kern_rows = []
    unattributed = 0
    for e in kernels:
        c = (e.get("args") or {}).get("correlation")
        phase = corr2marker.get(c)
        if phase is None:
            phase = "unattributed"
            unattributed += 1
        step = phase.split("_")[1] if phase.startswith("rf_s") else "?"
        # review 6d1a05b item 1: keep forward/backward/clip_optimizer separate
        # (previous rsplit only yielded m0..m3, merging fwd/bwd of same micro)
        if phase.startswith("rf_s"):
            parts = phase.split("_")
            kind = parts[2]          # forward | backward | clip_optimizer
            micro = parts[3] if len(parts) > 3 else ""
        else:
            kind = phase
            micro = ""
        kern_rows.append({"name": e["name"], "dur_us": e.get("dur", 0), "ts": e["ts"],
                           "stream": (e.get("args") or {}).get("stream"),
                           "phase": kind, "micro": micro, "step": step, "marker": phase,
                           "corr": c})
    attributed = sum(1 for k in kern_rows if not k["marker"].startswith(("outside", "unattributed", "before_first")))
    # inventory per class+phase
    inv = collections.defaultdict(lambda: {"count": 0, "ms": 0.0, "streams": set()})
    for k in kern_rows:
        cls = classify(k["name"])
        key = (cls, k["phase"])
        inv[key]["count"] += 1
        inv[key]["ms"] += k["dur_us"] / 1000
        inv[key]["streams"].add(k["stream"])
    inventory = {f"{cls}/{ph}": {"count": v["count"], "device_ms_3steps": round(v["ms"], 2),
                                  "streams": sorted(v["streams"])}
                  for (cls, ph), v in sorted(inv.items(), key=lambda kv: -kv[1]["ms"])}
    n_steps = 3
    inventory_per_step = {f"{cls}/{ph}": {"count_per_step": round(v["count"] / n_steps, 1),
                                           "device_ms_per_step": round(v["ms"] / n_steps, 2)}
                           for (cls, ph), v in sorted(inv.items(), key=lambda kv: -kv[1]["ms"])}
    # per-step totals + critical path (busy union) per step window
# review 6ba3bfd item 2: TRUE step boundaries [first marker start, last
    # marker end]; last step extends to its synchronize tail (last kernel end,
    # bounded) so the busy union only counts kernels inside the real step.
    step_windows = {}
    first_start = {}
    last_end = {}
    for e in markers:
        s = e["name"].split("_")[1]
        first_start.setdefault(s, e["ts"])
        last_end[s] = max(last_end.get(s, 0), e["ts"] + e.get("dur", 0))
    order = sorted(first_start.items(), key=lambda kv: kv[1])
    all_kernel_end = max((e["ts"] + e.get("dur", 0) for e in kernels), default=0)
    for idx, (s, t0) in enumerate(order):
        t1 = last_end[s]
        if idx + 1 < len(order):
            t1 = min(t1, order[idx + 1][1])          # cap at next step start
        else:
            t1 = max(t1, min(all_kernel_end, t0 + 5_000_000))  # sync tail
        step_windows[s] = {"t0": t0, "t1": t1}
    # report cross-step drain overlap: kernels attributed to step N executing after step N+1 window started
    drain_overlap = collections.Counter()
    by_step = collections.defaultdict(list)
    for k in kern_rows:
        by_step[k["step"]].append(k)
        # drain: kernel of step N executing inside step N+1 window
        if k["step"] in step_windows and k["ts"] >= step_windows[k["step"]]["t1"]:
            drain_overlap[k["step"]] += k["dur_us"]
    rt = [e for e in evs if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    # review 5ec9e6c item 4: include gpu_memcpy/gpu_memset in busy union;
    # report HtoD/DtoD/DtoH/memset counts+durations separately
    memops = [e for e in evs if e.get("cat") in ("gpu_memcpy", "gpu_memset")]
    def mem_kind(e):
        n = e.get("name", "")
        if "memset" in n.lower(): return "memset"
        if "HtoD" in n or "htod" in n.lower(): return "HtoD"
        if "DtoH" in n or "dtoh" in n.lower(): return "DtoH"
        if "DtoD" in n or "dtod" in n.lower(): return "DtoD"
        return "other_memcpy"
    mem_by = collections.defaultdict(lambda: {"count": 0, "ms": 0.0})
    for e in memops:
        w0, w1 = None, None
        for st, w in step_windows.items():
            if w["t0"] <= e["ts"] <= w["t1"]: w0, w1 = st, w; break
        if w0 is None: continue
        k = mem_kind(e)
        if k == "memset":
            mem_by[f"memset"][ "count"] += 1; mem_by["memset"]["ms"] += e.get("dur", 0) / 1000
        else:
            mem_by[k]["count"] += 1; mem_by[k]["ms"] += e.get("dur", 0) / 1000
        by_step[w0].append({"name": e["name"], "dur_us": e.get("dur", 0), "ts": e["ts"],
                             "stream": (e.get("args") or {}).get("stream"),
                             "phase": "memcpy", "micro": "", "step": w0, "marker": "rf_memcpy", "corr": None})
    crit = {}
    for s, w in step_windows.items():
        ivs = sorted((k["ts"], k["ts"] + k["dur_us"]) for k in by_step.get(s, []))
        union = 0.0; cur_s = cur_e = None
        for a_, b_ in ivs:
            if cur_s is None or a_ > cur_e:
                if cur_s is not None: union += cur_e - cur_s
                cur_s, cur_e = a_, b_
            else:
                cur_e = max(cur_e, b_)
        if cur_s is not None: union += cur_e - cur_s
        wall = (w["t1"] - w["t0"]) / 1000
        launches = sum(1 for e in rt if w["t0"] <= e["ts"] <= w["t1"])
        syncs = sum(1 for e in rt if w["t0"] <= e["ts"] <= w["t1"] and "Synchronize" in e.get("name", ""))
        crit[s] = {"window_ms": round(wall, 1), "device_busy_union_ms": round(union / 1000, 1),
                    "busy_union_scope": "kernels + gpu_memcpy + gpu_memset intervals",
                    "exposed_idle_ms": round(wall - union / 1000, 1),
                    "busy_pct": round(union / 1000 / wall * 100, 1),
                    "cpu_launches_in_window": launches,
                    "sync_calls_in_window": syncs,
                    "memcpy_events_in_window": {k: v for k, v in mem_by.items()},
                    "kernels": len(by_step.get(s, [])),
                    "drain_into_next_step_ms": round(drain_overlap.get(s, 0) / 1000, 1)}
    # GEMM FLOPs from measured per-shape calls (same seed window as r28)
    try:
        shapes = json.load(open(a.shapes))
        STEPS = shapes["capture_steps"]
        flops_fwd = 0.0
        calls_fwd = 0
        for k, v in shapes["modules"].items():
            w = v["w_shape"]; N, K = w
            for r in v["per_shape"]:
                sh = r["shape"]; M = sh[0] * sh[1] if len(sh) == 3 else sh[0]
                calls = r["calls_sum"] / STEPS
                flops_fwd += 2 * M * K * N * calls
                calls_fwd += calls
        gemm_flops_per_step = 3 * flops_fwd  # fwd + dX + dW
        gemm_ms = sum(v["ms"] for (cls, ph), v in inv.items() if cls == "gemm") / len(step_windows)
        tflops = gemm_flops_per_step / (gemm_ms * 1e-3) / 1e12
        gemm_eff = {"units_correction": "previous 111.98 was TFLOP/step (not PFLOP); flops from r28 shapes x3, kernel ms from r32 trace - CROSS-WINDOW ESTIMATE, superseded by same-window measurement in results/gemm_efficiency_r33.json (see measure_gemm_efficiency.py)",
                     "estimate_tflops_cross_window": round(tflops, 1),
                     "estimate_utilization_pct_of_1392": round(tflops / 139.2 * 100, 1),
                     "flops_per_step_tflop": round(gemm_flops_per_step / 1e12, 2),
                     "gemm_device_ms_per_step": round(gemm_ms, 1),
                     "peak_tflops": 139.2,
                     "vendor_measured_reference_tflops": 135.7}
        # HBM logical bytes (derived)
        bytes_per_step = 0.0
        for k, v in shapes["modules"].items():
            w = v["w_shape"]; N, K = w
            for r in v["per_shape"]:
                sh = r["shape"]; M = sh[0] * sh[1] if len(sh) == 3 else sh[0]
                calls = r["calls_sum"] / STEPS
                bytes_fwd = (M * K + K * N + M * N) * 2 * calls
                bytes_per_step += bytes_fwd * 3  # fwd + dX + dW operands
        hbm = {"logical_bytes_per_step_1e9": round(bytes_per_step / 1e9, 2),
                "est_bandwidth_GBs": round(bytes_per_step / (gemm_ms * 1e-3) / 1e9, 1),
                "hbm_peak_GBs": 2765,
                "utilization_pct": round(bytes_per_step / (gemm_ms * 1e-3) / 1e9 / 2765 * 100, 1),
                "note": "DERIVED logical operand bytes (fwd+dX+dW); HBM counters not available on this stack - missing item documented"}
    except FileNotFoundError:
        gemm_eff = {"error": "shapes file missing"}; hbm = {}
    out = {"trace": a.trace, "kernels_total": len(kern_rows),
            "attributed_to_phase_pct": round(attributed / max(1, len(kern_rows)) * 100, 1),
            "inventory_3steps": inventory, "inventory_per_step": inventory_per_step,
            "critical_path_per_step": crit,
            "gemm_efficiency": gemm_eff, "hbm_logical_estimate": hbm}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"kernels {len(kern_rows)}, attributed {out['attributed_to_phase_pct']}%")
    for key, v in list(inventory_per_step.items())[:14]:
        print(f"  {key:<28} {v['count_per_step']:>8} calls/step {v['device_ms_per_step']:>9.1f} ms/step")
    print("critical path per step:", json.dumps(crit))
    print("GEMM:", json.dumps(gemm_eff))
    if hbm: print("HBM:", json.dumps(hbm))

if __name__ == "__main__":
    main()
