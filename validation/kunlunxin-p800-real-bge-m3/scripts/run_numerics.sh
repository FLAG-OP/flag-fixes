#!/usr/bin/env bash
# Numerics trajectory runs (issue #9 B/E): fixed seed 42, 6 optimizer steps on
# the real data via the entry building blocks; card 2 (CARD=2), while the perf
# matrix owns card 1.
set -uo pipefail
EV=/workspace/evidence
FS=/workspace/FlagScale
FS_STOCK=/workspace/FlagScale-stock
FG=/workspace/FlagGems-fixed/src
mkdir -p $EV/results/numerics $EV/logs/numerics
export TRITON_CACHE_DIR=/workspace/evidence/.triton
export FLAGGEMS_CACHE_DIR=/workspace/evidence/.flaggems
export TORCH_EXTENSIONS_DIR=/workspace/evidence/.torch_ext
export KLX_USE_AUTOTUNE=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

num () {
  name=$1; fsroot=$2; include=$3; shift 3
  echo "=== numerics $name $(date +%H:%M:%S)"
  env "$@" CARD_VISIBLE=ignored CUDA_VISIBLE_DEVICES=${CARD:-2} \
    PYTHONPATH=$([ -n "$include" ] && echo "$FS:$FG" || echo "$fsroot") \
    python3 $EV/numerics_check.py --label $name --steps 6 \
      --flagscale-root $fsroot $( [ -n "$include" ] && echo --flaggems-include $include ) \
      --out $EV/results/numerics/$name.json > $EV/logs/numerics/$name.log 2>&1
  rc=$?
  echo "rc=$rc"; grep -E "^step|saved" $EV/logs/numerics/$name.log | tail -8 | sed 's/^/    /'
}

num native_r1  $FS_STOCK ""  FLAGSCALE_ADAMW_FUSED=auto
num native_r2  $FS_STOCK ""  FLAGSCALE_ADAMW_FUSED=auto
num gelu_only  $FS       "gelu,gelu_backward" FLAGSCALE_ADAMW_FUSED=0
num all_fixes  $FS       "gelu,gelu_backward" FLAGSCALE_ADAMW_FUSED=auto
NUM_GEMM=${NUM_GEMM:-0}
if [ "$NUM_GEMM" = 1 ]; then
  num gemm_env  $FS_STOCK ""  FLAGSCALE_ADAMW_FUSED=auto XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1
fi
echo NUMERICS_DONE
