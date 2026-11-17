#!/usr/bin/env bash
# Real BGE-M3 ablation matrix on KunlunXIN P800 (issue #9 sections A+E).
# Each config = 3 independent processes; each process = skip 2 + 3 timed steps
# (median), the problem29 methodology, on the ORIGINAL FlagScale entry
# (examples/retrieval/profile_step_segments.py via segments_with_fixes.py).
#
# Configs (ablation, gains are NOT assumed additive):
#   native : stock entry, no FlagGems, no vendor env
#   gemm   : + XMLIR_MATMUL_FAST_MODE/FAST_TUNE (vendor GEMM fast algo env)
#   gelu   : + patched FlagGems via flaggems_include=[gelu,gelu_backward]
#            (FLAGSCALE_ADAMW_FUSED=0 forces stock AdamW so only GELU differs)
#   glue   : + XMLIR_ENABLE_LINEAR_FC_FUSION=0 (only locally switchable glue
#            knob; PR3 attribution says net-negative — measured, not assumed)
#   all    : gelu + adamw-fused(auto) + GEMM env, FlagScale integration path
set -uo pipefail
ROOT=/workspace/evidence
FS=${FS_TREE:-/workspace/FlagScale}            # flagscale tree with selective-enable patch
FS_STOCK=${FS_STOCK_TREE:-/workspace/FlagScale-stock} # pristine flagscale tree (native baseline)
FG=${FG_TREE:-/workspace/FlagGems-fixed/src}    # patched FlagGems tree
CONF=/workspace/FlagScale/examples/retrieval/conf/train/bge_m3.yaml
OUT=${SEG_OUT:-$ROOT/results-clean/segments}
LOG=${SEG_LOG:-$ROOT/logs-clean/segments}
mkdir -p "$OUT" "$LOG"

# Keep cache dirs off $HOME and constant across configs.
export TRITON_CACHE_DIR=/workspace/evidence/.triton
export FLAGGEMS_CACHE_DIR=/workspace/evidence/.flaggems
export TORCH_EXTENSIONS_DIR=/workspace/evidence/.torch_ext
export KLX_USE_AUTOTUNE=0          # constant across the whole matrix
export XPU_ENABLE_PROFILER_TRACING=1   # kunlunxin torch.profiler requirement (problem29 §3.2)
export XPU_CUPTI_ENABLE_DEVICE=0       # logical device 0, not physical card id
unset USE_FLAGGEMS                  # legacy env read by nothing; avoid surprises
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

CFG=$1
REP=$2

run_one () {
  local name=$1 tag=$2; shift 2
  local out=$OUT/${name}_r${REP}.json
  local log=$LOG/${name}_r${REP}.log
  echo "=== [$name rep$REP] $(date +%H:%M:%S) -> $out"
  timeout 3000 "$@" > "$log" 2>&1
  local rc=$?
  echo "rc=$rc"
  tail -6 "$log" | sed 's/^/    /'
  if [ $rc -ne 0 ]; then echo "!!! $name rep$REP FAILED rc=$rc"; fi
}

case $CFG in
  native)
    run_one native $REP env -u GEMS_VENDOR CUDA_VISIBLE_DEVICES=${CARD:-1} \
      PYTHONPATH=$FS_STOCK \
      python3 $ROOT/segments_with_fixes.py --conf $CONF --steps 3 --skip 2 --device cuda \
      --flagscale-root $FS_STOCK --out $OUT/native_r${REP}.json
    ;;
  gemm)
    run_one gemm $REP env -u GEMS_VENDOR CUDA_VISIBLE_DEVICES=${CARD:-1} \
      PYTHONPATH=$FS_STOCK XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1 \
      python3 $ROOT/segments_with_fixes.py --conf $CONF --steps 3 --skip 2 --device cuda \
      --flagscale-root $FS_STOCK --out $OUT/gemm_r${REP}.json
    ;;
  glue)
    run_one glue $REP env -u GEMS_VENDOR CUDA_VISIBLE_DEVICES=${CARD:-1} \
      PYTHONPATH=$FS_STOCK XMLIR_ENABLE_LINEAR_FC_FUSION=0 \
      python3 $ROOT/segments_with_fixes.py --conf $CONF --steps 3 --skip 2 --device cuda \
      --flagscale-root $FS_STOCK --out $OUT/glue_r${REP}.json
    ;;
  gelu)
    run_one gelu $REP env CUDA_VISIBLE_DEVICES=${CARD:-1} FLAGSCALE_ADAMW_FUSED=0 \
      GEMS_VENDOR=kunlunxin PYTHONPATH=$FS:$FG \
      python3 $ROOT/segments_with_fixes.py --conf $CONF --steps 3 --skip 2 --device cuda \
      --flagscale-root $FS --flaggems-include gelu,gelu_backward --out $OUT/gelu_r${REP}.json
    ;;
  all)
    run_one all $REP env CUDA_VISIBLE_DEVICES=${CARD:-1} \
      GEMS_VENDOR=kunlunxin PYTHONPATH=$FS:$FG \
      XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1 \
      python3 $ROOT/segments_with_fixes.py --conf $CONF --steps 3 --skip 2 --device cuda \
      --flagscale-root $FS --flaggems-include gelu,gelu_backward --out $OUT/all_r${REP}.json
    ;;
  *) echo "unknown config $CFG"; exit 2;;
esac
