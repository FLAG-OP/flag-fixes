#!/usr/bin/env bash
# Numerics trajectory runs — configs sourced from the SAME definition as the
# perf ablation (ablation_configs.sh, review round-1 P1-1; round-2: state/
# compare paths unified and compared results gate the run).
# revision branch covers native / gelu_only / all_fixes (=perf all env) /
# gemm_env (=perf D env), saves final tensors and runs all three quantified
# compares; any missing artifact or failed compare -> nonzero exit.
set -uo pipefail
EV=/workspace/evidence
source $EV/ablation_configs.sh
NUMDIR=$EV/results/numerics
STATDIR=$NUMDIR/state
mkdir -p "$NUMDIR" "$EV/logs/numerics" "$STATDIR"
export TRITON_CACHE_DIR=/workspace/evidence/.triton
export FLAGGEMS_CACHE_DIR=/workspace/evidence/.flaggems
export TORCH_EXTENSIONS_DIR=/workspace/evidence/.torch_ext
export KLX_USE_AUTOTUNE=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
FAILS=()

num () {
  local name=$1 cfgname=$2
  IFS='|' read -r pp fs env <<< "$(config_$cfgname)"
  local include=""
  if config_uses_gems "$cfgname"; then include=$GELU_INCLUDE; fi
  echo "=== numerics $name (config=$cfgname env='$env' include='$include') $(date +%H:%M:%S)"
  env CUDA_VISIBLE_DEVICES=${CARD:-2} $env \
    PYTHONPATH=$pp \
    python3 $EV/numerics_check.py --label $name --steps 6 \
      --flagscale-root $fs \
      $( [ -n "$include" ] && echo --flaggems-include $include ) \
      --save-final $STATDIR/$name.pt \
      --out $NUMDIR/$name.json > $EV/logs/numerics/$name.log 2>&1
  local rc=$?
  echo "rc=$rc"; grep -E "^step|saved" $EV/logs/numerics/$name.log | tail -8 | sed 's/^/    /'
  [ -s $NUMDIR/$name.json ] || { echo "MISSING JSON $name"; rc=1; }
  [ -s $STATDIR/$name.pt ] || { echo "MISSING STATE $name"; rc=1; }
  [ $rc -ne 0 ] && FAILS+=("$name")
  return $rc
}

cmp () {
  local tgt=$1 out=$2.json
  echo "=== compare $tgt vs native_r3 -> $out"
  CUDA_VISIBLE_DEVICES= python3 $EV/numerics_compare.py \
    $NUMDIR/native_r3.json $STATDIR/native_r3.pt \
    $NUMDIR/$tgt.json $STATDIR/$tgt.pt \
    $NUMDIR/$out > $EV/logs/numerics/$out.log 2>&1
  local rc=$?
  echo "rc=$rc"; tail -12 $EV/logs/numerics/$out.log | grep -v "XCCL\|SYMBOL" | sed 's/^/    /'
  [ -s $NUMDIR/$out ] || { echo "MISSING COMPARE $out"; rc=1; }
  [ $rc -ne 0 ] && FAILS+=("$out")
  return $rc
}

WHAT=${1:-revision}
if [ "$WHAT" = "revision" ]; then
  num native_r3 native
  num gelu_only_v2 gelu
  num all_fixes_v2 all
  num gemm_env_v2 gemm
  cmp gelu_only_v2 compare_gelu_vs_native
  cmp all_fixes_v2 compare_all_vs_native
  cmp gemm_env_v2 compare_gemm_vs_native
else
  num native_r1 native
  num native_r2 native
  num gelu_only_v2 gelu
  num all_fixes_v2 all
fi

if [ ${#FAILS[@]} -gt 0 ]; then echo "NUMERICS_FAILED: ${FAILS[*]}"; exit 1; fi
echo NUMERICS_DONE
