#!/bin/bash
set -u
REPO="FLAG-OP/flag-fixes"
R="/tmp/opencode/remote-files"
W="/tmp/opencode/zw810e-problem29"

upload_blob() {
  local SRC="$1"
  local B64FILE="/tmp/opencode/remote-files/_blob_body_tmp.json"
  base64 -w0 "$SRC" > /tmp/opencode/remote-files/_blob_b64_tmp.txt
  echo "{\"content\":\"$(cat /tmp/opencode/remote-files/_blob_b64_tmp.txt)\",\"encoding\":\"base64\"}" > "$B64FILE"
  for i in 1 2 3 4 5 6 7 8; do
    gh api -X POST "repos/$REPO/git/blobs" --input "$B64FILE" > /tmp/opencode/remote-files/_blob_result.json 2>/dev/null
    if [ -s /tmp/opencode/remote-files/_blob_result.json ]; then
      SHA=$(python3 -c "import json;print(json.load(open('/tmp/opencode/remote-files/_blob_result.json'))['sha'])" 2>/dev/null)
      if [ -n "$SHA" ]; then echo "$SHA"; return 0; fi
    fi
    sleep 20
  done
  return 1
}

: > "$R/all_entries.txt"
count=0

upload_and_record() {
  local SRC="$1" DST="$2"
  SHA=$(upload_blob "$SRC")
  if [ -n "$SHA" ]; then
    echo "{\"path\":\"$DST\",\"mode\":\"100644\",\"type\":\"blob\",\"sha\":\"$SHA\"}" >> "$R/all_entries.txt"
    count=$((count+1))
    echo "OK[$count]: $DST"
  else
    echo "FAIL: $DST"
    return 1
  fi
}

# AB script from FlagScale-ppu (correct version with seed 42 + init outside loop)
upload_and_record "$FSCALE/examples/retrieval/profile_step_segments_ab.py" "validation/thead-zw810e-problem29/scripts/profile_step_segments_ab.py"
# probe11 from FlagScale-ppu
upload_and_record "$FSCALE/examples/retrieval/probe11_mask_vs_equal_length.py" "validation/thead-zw810e-problem29/scripts/probe11_mask_vs_equal_length.py"
# probe10 from zw810e (has pre-conversion)
upload_and_record "$W/scripts/probe10_gemm_ablation.py" "validation/thead-zw810e-problem29/scripts/probe10_gemm_ablation.py"

# Other scripts
for f in adamw_stable_ref_v2.py adamw_stable_ref.py comparator.py test_comparator_negative.py accuracy_matrix.py probe9_gemm_sdpa.py probe9b_sdpa_masked.py probe11_sdpa_context.py run_orig_baseline.sh dl_all.sh; do
  upload_and_record "$W/scripts/$f" "validation/thead-zw810e-problem29/scripts/$f"
done

# Results
for root_dirs in "$W/results" "$W/results/orig_ab" "$W/results/orig_ab_reps" "$W/results/perf_batch1_superseded" "$W/results/small_model_v1_superseded" "$W/results/small_model_v2" "$W/results/small_model_e"; do
  for f in $(find "$root_dirs" -name "*.json" -type f 2>/dev/null | sort); do
    rel=$(echo "$f" | sed "s|$W/||")
    upload_and_record "$f" "validation/thead-zw810e-problem29/$rel"
  done
done

# Logs
for f in $(find "$W/logs" -name "*.log" -type f 2>/dev/null | sort); do
  bn=$(basename "$f")
  rel=$(echo "$f" | sed "s|$W/||")
  upload_and_record "$f" "validation/thead-zw810e-problem29/$rel"
done

echo "ALL_BLOBS_DONE count=$count"
