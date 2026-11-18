#!/usr/bin/env bash
# 5 configs x 3 independent processes of the segments harness (sequential),
# with a hard completeness gate: EXACT enumeration of config x rep paths
# (review round-3 P1: the previous "{cfg}_r*.json" glob also matched the
# .env.json sidecars, so a fully successful 15-run matrix counted 6 files per
# config and could never pass).
set -uo pipefail
EV=/workspace/evidence
FAILS=()

for cfg in native gemm gelu glue all; do
  for rep in 1 2 3; do
    SEG_OUT=$EV/results-clean/segments SEG_LOG=$EV/logs-clean/segments \
      bash $EV/run_ablation.sh $cfg $rep || FAILS+=("$cfg r$rep")
  done
done

# completeness gate: exact main-file paths config_r{1,2,3}.json, each with the
# required numeric fields, each with a matching .env.json sidecar whose config
# name matches; sidecars alone never count as results
python3 - "$EV/results-clean/segments" <<'PY' || FAILS+=("json-validation")
import glob, json, math, os, sys
base = sys.argv[1]
need = {"wall_ms", "kernel_ms_per_step", "per_step_ms", "ops"}
numeric = {"wall_ms", "kernel_ms_per_step"}
bad = []
sidecar_dangling = [f for f in sorted(glob.glob(f"{base}/*.json.env.json"))
                    if not os.path.exists(f[:-len(".env.json")])]
for cfg in ("native", "gemm", "gelu", "glue", "all"):
    for rep in (1, 2, 3):
        f = f"{base}/{cfg}_r{rep}.json"
        if not os.path.exists(f):
            bad.append((f, "missing main JSON"))
            continue
        try:
            d = json.load(open(f))
        except Exception as e:
            bad.append((f, f"unparseable: {e}")); continue
        missing = [k for k in need if k not in d]
        if missing:
            bad.append((f, f"missing {missing}")); continue
        for k in numeric:
            v = d[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                bad.append((f, f"{k} not a positive finite number: {v!r}"))
        ps = d.get("per_step_ms", {})
        for k, v in ps.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0:
                bad.append((f, f"per_step_ms.{k} not a finite non-negative number: {v!r}"))
        sc = f + ".env.json"
        if not os.path.exists(sc):
            bad.append((f, "missing .env.json provenance sidecar")); continue
        try:
            m = json.load(open(sc))
        except Exception as e:
            bad.append((sc, f"unparseable: {e}")); continue
        if m.get("flagscale_root") is None or "env" not in m:
            bad.append((sc, "sidecar missing config metadata"))
if sidecar_dangling:
    bad.extend((f, "sidecar without main JSON") for f in sidecar_dangling)
if bad:
    for b in bad:
        print("VALIDATION FAIL:", b)
    sys.exit(1)
print("segments JSON validation OK: 5 configs x 3 reps (exact paths) + sidecars")
PY

if [ ${#FAILS[@]} -gt 0 ]; then
  echo "SEGMENTS_MATRIX_FAILED: ${FAILS[*]}"
  exit 1
fi
echo SEGMENTS_MATRIX_DONE
