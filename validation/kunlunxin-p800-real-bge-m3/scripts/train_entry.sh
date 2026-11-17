#!/usr/bin/env bash
# One run of the REAL FlagScale trainer (flagscale.train.train_retrieval) on the
# real bge_m3.yaml-derived config, single P800 card, log_every=1 with externally
# timestamped stdout; xpu-smi memory sampled in the background.
# Usage: train_entry.sh <name> <rep> <out_root>
set -uo pipefail
EV=/workspace/evidence
FS=${FS_TREE:-/workspace/FlagScale}          # flagscale tree (patched or stock)
FG=${FG_TREE:-}                              # patched FlagGems tree or empty
NAME=$1; REP=$2; OUTROOT=${3:-$EV/results/trainer}
mkdir -p "$OUTROOT/$NAME" $EV/logs/trainer
CFG=$OUTROOT/$NAME
LOG=$EV/logs/trainer/${NAME}_r${REP}.log
MEMLOG=$EV/logs/trainer/${NAME}_r${REP}.mem.log

export TRITON_CACHE_DIR=/workspace/evidence/.triton
export FLAGGEMS_CACHE_DIR=/workspace/evidence/.flaggems
export TORCH_EXTENSIONS_DIR=/workspace/evidence/.torch_ext
export KLX_USE_AUTOTUNE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export FLAGSCALE_DEVICE=cuda FS_PLATFORM=cuda FLAGSCALE_AUTOCAST_DEVICE=cuda
export GEMS_VENDOR=${GEMS_VENDOR:-}
export FLAGSCALE_ADAMW_FUSED=${FLAGSCALE_ADAMW_FUSED:-auto}
unset USE_FLAGGEMS
[ -n "$FG" ] && export PYTHONPATH=$FS:$FG || export PYTHONPATH=$FS

python3 $EV/compose_train_cfg.py "$CFG"
if [ -f "$CFG/config.yaml.lock" ]; then rm -f "$CFG/config.yaml.lock"; fi

# background memory sampler (logical card 0 of CUDA_VISIBLE_DEVICES)
( while sleep 2; do xpu-smi -i ${XPU_ID:-0} 2>/dev/null | grep -E "MiB" | sed "s/^/$(date +%s) /" >> $MEMLOG; done ) &
SAMPLER=$!

python3 -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m flagscale.train.train_retrieval --config-file "$CFG/config.yaml" \
  2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done > "$LOG"
RC=${PIPESTATUS[0]}
kill $SAMPLER 2>/dev/null
echo "rc=$RC"
grep -E "optimizer_step|running_loss|ERROR|Error|Traceback" "$LOG" | tail -8 | sed 's/^/    /'
exit $RC
