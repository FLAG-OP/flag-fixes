#!/usr/bin/env bash
# 5 configs x 3 independent processes of the segments harness (sequential),
# with a hard completeness gate: every (config, rep) must produce a JSON with
# the required fields; any failure (or stale/missing artifact) exits nonzero.
set -uo pipefail
EV=/workspace/evidence
FAILS=()

for cfg in native gemm gelu glue all; do
  for rep in 1 2 3; do
    SEG_OUT=$EV/results-clean/segments SEG_LOG=$EV/logs-clean/segments \
      bash $EV/run_ablation.sh $cfg $rep || FAILS+=("$cfg r$rep")
  done
done

# completeness gate: exactly 3 fresh valid JSONs per config, each with the
# required fields and a matching .env.json provenance sidecar
python3 - <<'PY' || FAILS+=("json-validation")
import glob, json, os, sys
need = {"wall_ms", "kernel_ms_per_step", "per_step_ms", "ops"}
bad = []
for cfg in ("native", "gemm", "gelu", "glue", "all"):
    files = sorted(glob.glob(f"/workspace/evidence/results-clean/segments/{cfg}_r*.json"))
    if len(files) != 3:
        bad.append((cfg, f"expected 3 reps, got {len(files)}"))
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            bad.append((f, f"unparseable: {e}")); continue
        missing = [k for k in need if k not in d]
        if missing:
            bad.append((f, f"missing {missing}"))
        if not os.path.exists(f + ".env.json"):
            bad.append((f, "missing .env.json provenance sidecar"))
if bad:
    for b in bad:
        print("VALIDATION FAIL:", b)
    sys.exit(1)
print("segments JSON validation OK: 5 configs x 3 reps + provenance sidecars")
PY

if [ ${#FAILS[@]} -gt 0 ]; then
  echo "SEGMENTS_MATRIX_FAILED: ${FAILS[*]}"
  exit 1
fi
echo SEGMENTS_MATRIX_DONE
