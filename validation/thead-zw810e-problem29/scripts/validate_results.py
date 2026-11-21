"""Offline validation (review 8190910 final item):
1. SDPA units: per-step FLOPs vs per-step ms (no 3-step sum against per-step ms)
2. copy/memset per-step conservation vs totals
3. final_table rows consistent with source JSONs
4. every efficiency TFLOP/s <= 139.2 peak
Exit 0 iff all checks pass."""
import json, sys

FAILS = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" [{detail}]" if detail else ""))
    if not ok: FAILS.append(name)

# --- 1. SDPA units ---
sd = json.load(open("results/sdpa_efficiency_r35.json"))
per_step = sd["per_step_tflop"]
mean_f = sum(v["forward"] for v in per_step.values()) / len(per_step)
mean_b = sum(v["backward"] for v in per_step.values()) / len(per_step)
tf_f = sd["tflops"]["forward"]; tf_b = sd["tflops"]["backward"]
check("SDPA tflops recomputed from per-step means", abs(tf_f - mean_f/(sd["sdpa_device_ms_per_step"]["forward"]*1e-3)) < 0.05
      and abs(tf_b - mean_b/(sd["sdpa_device_ms_per_step"]["backward"]*1e-3)) < 0.05)
check("SDPA per-step FLOPs are per-step (not 3-step sums)",
      max(v["forward"] for v in per_step.values()) < 0.3)
# group-row totals == step totals
rows_f = sum(r["fwd_tflop"] for r in sd["per_group_rows"])
steps_f = sum(v["forward"] for v in per_step.values())
check("SDPA group rows sum == per-step sum", abs(rows_f - steps_f) < 0.01, f"{rows_f:.3f} vs {steps_f:.3f}")

# --- 2. copy/memset conservation ---
an = json.load(open("results/training_trace_analysis_r35.json"))
tot = an["memcpy_total_events"]
per_step_sum = 0
for s, c in an["critical_path_per_step"].items():
    for k, v in c["memcpy_events_per_step"].items():
        per_step_sum += v["count"]
check("memcpy per-step counts sum == total events", per_step_sum == tot, f"{per_step_sum} vs {tot}")

# --- 3. final table vs sources ---
ft = json.load(open("results/final_table_r35.json"))
ke = json.load(open("results/kernel_efficiency_r36.json"))
import collections as _c
ke_ms = _c.defaultdict(list)
for _r in ke["rows"]:
    ke_ms[(_r["module_class"], _r["phase"], _r["calls_per_step"])].append(_r["device_ms"])
ok3 = True; matched = 0
for r in ft["rows"]:
    if r["source"].startswith("kernel_efficiency"):
        toks = r["stage"].split()
        cls = toks[1]
        phase = "forward" if "fwd" in toks else "backward"
        cand = ke_ms.get((cls, phase, r["calls"]), [])
        if not cand or not any(abs(c - r["ms"]) <= 0.2 for c in cand):
            ok3 = False
            print("  fail row:", r["stage"], r["calls"], r["ms"], cand)
        else:
            matched += 1
check("final_table ms consistent with kernel_efficiency_r36", ok3 and matched > 0, f"{matched} rows matched")

# --- 4. TFLOP/s ceiling ---
over = []
for r in ft["rows"]:
    if r.get("tflops") is not None and r["tflops"] > 139.2:
        over.append(f"{r['stage']}={r['tflops']}")
check("all efficiency rows <= 139.2 TFLOP/s", not over, "; ".join(over))
for r in ke["rows"]:
    if r.get("tflops") is not None and r["tflops"] > 139.2:
        over.append(f"ke:{r['module_class']}/{r['phase']}={r['tflops']}")
check("kernel_efficiency rows <= 139.2 TFLOP/s", not (len(over) > 0 and any(o.startswith('ke:') for o in over)), "; ".join(o for o in over if o.startswith('ke:')))

# --- pairing asserted ---
sw = json.load(open("results/samewindow_noprofiler_r36.json"))
check("no-profiler pairing asserted (text-hash)", sw["pairing_assertions"]["asserted"] is True)
ke_fp = json.load(open("results/kernel_efficiency_r36.json"))["fingerprint_verification"]
check("kernel-efficiency fingerprint asserted", ke_fp["asserted"] is True)

print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILURES: ' + str(FAILS)}")
sys.exit(0 if not FAILS else 1)
