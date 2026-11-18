#!/usr/bin/env bash
# One ablation config, one rep: run the ORIGINAL FlagScale segments harness
# (examples/retrieval/profile_step_segments.py via segments_with_fixes.py).
# Configs come from the shared ablation_configs.sh (same source as the
# numerics runner). Failure semantics (review round-2 P1): the training
# command is passed AS ARGUMENTS to run_one so timeout/log-redirect/JSON
# validation always wrap it; any failure is recorded and propagated via
# exit code (fail-fast off; matrix aggregates).
set -uo pipefail
EV=/workspace/evidence
# shellcheck source=ablation_configs.sh
source $EV/ablation_configs.sh
OUT=${SEG_OUT:-$EV/results-clean/segments}
LOG=${SEG_LOG:-$EV/logs-clean/segments}
FAILS_FILE=${SEG_FAILS:-$EV/logs-clean/ablation_failures.txt}
mkdir -p "$OUT" "$LOG" "$(dirname "$FAILS_FILE")"

export TRITON_CACHE_DIR=/workspace/evidence/.triton
export FLAGGEMS_CACHE_DIR=/workspace/evidence/.flaggems
export TORCH_EXTENSIONS_DIR=/workspace/evidence/.torch_ext
export KLX_USE_AUTOTUNE=0
export XPU_ENABLE_PROFILER_TRACING=1
export XPU_CUPTI_ENABLE_DEVICE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
unset USE_FLAGGEMS

CFG=$1
REP=$2

IFS='|' read -r pp fs env <<< "$(config_$CFG)"
include=""
if config_uses_gems "$CFG"; then include=$GELU_INCLUDE; fi

run_one () {
  local name=$1 rep=$2; shift 2
  local out=$OUT/${name}_r${rep}.json
  local log=$LOG/${name}_r${rep}.log
  local hist=$OUT/history
  mkdir -p "$hist"
  # isolate stale artifacts: a previous run's JSON/sidecar must never be able
  # to pass for this run's result (review round-3 P2). They are moved aside
  # before training and only restored there, never republished.
  for f in "$out" "$out.env.json"; do
    [ -e "$f" ] && mv "$f" "$hist/${name}_r${rep}_$(date +%Y%m%d-%H%M%S)_$(basename "$f")"
  done
  echo "=== [$name rep$rep] $(date +%H:%M:%S) -> $out"
  timeout 3000 "$@" > "$log" 2>&1
  local rc=$?
  echo "rc=$rc"
  tail -6 "$log" | sed 's/^/    /'
  # publish gate: a FRESH artifact must exist and carry the required numeric
  # fields (positive finite wall/kernel, finite non-negative segments)
  python3 - "$out" <<'PY' || rc=1
import json, math, sys
try:
    d = json.load(open(sys.argv[1]))
    for k in ("wall_ms", "kernel_ms_per_step", "per_step_ms", "ops"):
        assert k in d, f"missing {k}"
    for k in ("wall_ms", "kernel_ms_per_step"):
        v = d[k]
        assert isinstance(v, (int, float)) and not isinstance(v, bool), f"{k} not numeric: {v!r}"
        assert math.isfinite(v) and v > 0, f"{k} not positive finite: {v!r}"
    for k, v in d.get("per_step_ms", {}).items():
        assert isinstance(v, (int, float)) and not isinstance(v, bool), f"per_step_ms.{k} not numeric: {v!r}"
        assert math.isfinite(v) and v >= 0, f"per_step_ms.{k} not finite non-negative: {v!r}"
except Exception as e:
    print(f"JSON VALIDATION FAILED: {e}")
    sys.exit(1)
PY
  # config provenance sidecar is written only if training AND the publish
  # gate both passed (never for a failed/missing artifact)
  if [ $rc -eq 0 ]; then
    python3 - "$out" "$pp" "$fs" "$env" "$include" <<'PY' || rc=1
import json, sys
out, pp, fs, env, include = sys.argv[1:6]
meta = {"pythonpath": pp, "flagscale_root": fs,
        "env": dict(kv.split("=", 1) for kv in env.split() if "=" in kv),
        "flaggems_include": [s for s in include.split(",") if s]}
try:
    json.dump(meta, open(out + ".env.json", "w"), indent=1)
except Exception as e:
    print(f"ENV META FAILED: {e}")
    sys.exit(1)
PY
  fi
  if [ $rc -ne 0 ]; then
    echo "!!! $name rep$rep FAILED rc=$rc"
    echo "$name rep$rep rc=$rc" >> "$FAILS_FILE"
  fi
  return $rc
}

rc_total=0
case $CFG in
  native)
    run_one native "$REP" env CUDA_VISIBLE_DEVICES=${CARD:-1} PYTHONPATH=$pp \
      python3 $EV/segments_with_fixes.py --conf $CONF --steps 3 --skip 2 --device cuda \
      --flagscale-root $fs --out $OUT/native_r${REP}.json || rc_total=1
    ;;
  gemm)
    run_one gemm "$REP" env CUDA_VISIBLE_DEVICES=${CARD:-1} $env PYTHONPATH=$pp \
      python3 $EV/segments_with_fixes.py --conf $CONF --steps 3 --skip 2 --device cuda \
      --flagscale-root $fs --out $OUT/gemm_r${REP}.json || rc_total=1
    ;;
  glue)
    run_one glue "$REP" env CUDA_VISIBLE_DEVICES=${CARD:-1} $env PYTHONPATH=$pp \
      python3 $EV/segments_with_fixes.py --conf $CONF --steps 3 --skip 2 --device cuda \
      --flagscale-root $fs --out $OUT/glue_r${REP}.json || rc_total=1
    ;;
  gelu)
    run_one gelu "$REP" env CUDA_VISIBLE_DEVICES=${CARD:-1} $env PYTHONPATH=$pp \
      python3 $EV/segments_with_fixes.py --conf $CONF --steps 3 --skip 2 --device cuda \
      --flagscale-root $fs --flaggems-include $include --out $OUT/gelu_r${REP}.json || rc_total=1
    ;;
  all)
    run_one all "$REP" env CUDA_VISIBLE_DEVICES=${CARD:-1} $env PYTHONPATH=$pp \
      python3 $EV/segments_with_fixes.py --conf $CONF --steps 3 --skip 2 --device cuda \
      --flagscale-root $fs --flaggems-include $include --out $OUT/all_r${REP}.json || rc_total=1
    ;;
  *) echo "unknown config $CFG"; exit 2;;
esac

exit $rc_total
