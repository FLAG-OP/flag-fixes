#!/usr/bin/env bash
# 5 configs x 3 independent processes of the segments harness (sequential).
set -uo pipefail
EV=/workspace/evidence
for cfg in native gemm gelu glue all; do
  for rep in 1 2 3; do
    SEG_OUT=$EV/results-clean/segments SEG_LOG=$EV/logs-clean/segments bash $EV/run_ablation.sh $cfg $rep
  done
done
echo SEGMENTS_MATRIX_DONE
