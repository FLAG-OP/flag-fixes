#!/usr/bin/env bash
# GEMM env sweep at real BGE-M3 shapes — one process per env config.
set -uo pipefail
EV=/workspace/evidence
mkdir -p $EV/results-clean/gemm $EV/logs-clean/gemm
run () {
  name=$1; shift
  echo "=== gemm $name $(date +%H:%M:%S)"
  env CUDA_VISIBLE_DEVICES=${CARD:-1} KLX_USE_AUTOTUNE=0 "$@" \
    python3 $EV/bench_gemm_shapes.py --out $EV/results-clean/gemm/$name.json \
    > $EV/logs-clean/gemm/$name.log 2>&1
  rc=$?
  echo "rc=$rc"; tail -2 $EV/logs-clean/gemm/$name.log | sed 's/^/    /'
}
run base env -u XMLIR_MATMUL_FAST_MODE -u XMLIR_MATMUL_FAST_TUNE
run matmul_fast XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1
run matmul_fast_mode_only XMLIR_MATMUL_FAST_MODE=1
run klx_autotune KLX_USE_AUTOTUNE=1
run xblas_addmm XMLIR_ENABLE_XBLAS_ADDMM=1
run fc_autotune_bool XMLIR_FC_AUTOTUNE_BOOL=1
run bmm_dispatch_0 XMLIR_BMM_DISPATCH_VALUE=0
echo SWEEP_DONE
