#!/usr/bin/env bash
# Generate .env.json provenance sidecars for the archived (pre-sidecar) clean
# batch, reconstructed from scripts/ablation_configs.sh — the same source of
# truth run_ablation.sh uses. Each sidecar carries "retroactive": true so it
# is never mistaken for a runner-produced original. Values for gelu/all include
# GEMS_VENDOR=kunlunxin; native/gemm/glue ran with GEMS_VENDOR unset (inert —
# flag_gems is never imported in those configs).
set -uo pipefail
EV=/workspace/evidence
source $EV/ablation_configs.sh
OUT=${1:-$EV/results-clean/segments}
GELU_INCLUDE="gelu,gelu_backward"

for cfg in native gemm gelu glue all; do
  IFS='|' read -r pp fs env <<< "$(config_$cfg)"
  include=""
  case $cfg in gelu|all) include=$GELU_INCLUDE; env="$env GEMS_VENDOR=kunlunxin";; esac
  python3 - "$OUT" "$cfg" "$pp" "$fs" "$env" "$include" <<'PY'
import json, os, sys
out, cfg, pp, fs, env, include = sys.argv[1:7]
for rep in (1, 2, 3):
    main = f"{out}/{cfg}_r{rep}.json"
    if not os.path.exists(main):
        continue
    meta = {"pythonpath": pp, "flagscale_root": fs,
            "env": dict(kv.split("=", 1) for kv in env.split() if "=" in kv),
            "flaggems_include": [s for s in include.split(",") if s],
            "retroactive": True}
    json.dump(meta, open(main + ".env.json", "w"), indent=1)
    print("sidecar:", main + ".env.json")
PY
done
