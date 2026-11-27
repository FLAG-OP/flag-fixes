#!/bin/bash
set -u
R=/tmp/opencode/remote-files
REPO="FLAG-OP/flag-fixes"
BR="issue7-zw810e-opt-round1"
PARENT_FILE="$R/parent_sha_r15.txt"

# get parent
timeout 60 gh api "repos/$REPO/git/refs/heads/$BR" --jq '.object.sha' > "$R/parent_r16.sha" 2>/dev/null
PARENT=$(cat "$R/parent_r16.sha")
BASETREE=$(timeout 60 gh api "repos/$REPO/git/commits/$PARENT" --jq '.tree.sha' 2>/dev/null)
echo "parent=$PARENT base_tree=$BASETREE"

# Upload blobs and build entries
W="/tmp/opencode/zw810e-problem29"
: > "$R/entries_r16.txt"

upload_and_record() {
  local SRC="$1" DST="$2"
  base64 -w0 "$SRC" > "$R/_blob_b64_tmp.txt"
  echo "{\"content\":\"$(cat $R/_blob_b64_tmp.txt)\",\"encoding\":\"base64\"}" > "$R/_blob_body.json"
  local i=0
  for i in $(seq 1 10); do
    gh api -X POST "repos/$REPO/git/blobs" --input "$R/_blob_body.json" > "$R/_blob_out.json" 2>/dev/null
    if [ -s "$R/_blob_out.json" ]; then
      SHA=$(python3 -c "import json;print(json.load(open('$R/_blob_out.json'))['sha'])" 2>/dev/null)
      if [ -n "$SHA" ]; then
        echo "{\"path\":\"$DST\",\"mode\":\"100644\",\"type\":\"blob\",\"sha\":\"$SHA\"}" >> "$R/entries_r16.txt"
        echo "OK: $DST"
        return 0
      fi
    fi
    sleep 20
  done
  echo "FAIL: $DST"
  return 1
}

# 1. Updated scripts (probe10 fp32 pre-conversion, AB seed+init, AdamW verdict selftest)
W_SCRIPTS="/tmp/opencode/zw810e-problem29/scripts"
FS_SCRIPTS="/tmp/opencode/FlagScale-ppu/examples/retrieval"
upload_and_record "$FS_SCRIPTS/profile_step_segments_ab.py" "validation/thead-zw810e-problem29/scripts/profile_step_segments_ab.py"
upload_and_record "$W_SCRIPTS/probe10_gemm_ablation.py" "validation/thead-zw810e-problem29/scripts/probe10_gemm_ablation.py"
upload_and_record "$W_SCRIPTS/comparator.py" "validation/thead-zw810e-problem29/scripts/comparator.py"
upload_and_record "$W_SCRIPTS/accuracy_matrix.py" "validation/thead-zw810e-problem29/scripts/accuracy_matrix.py"
upload_and_record "$W_SCRIPTS/test_comparator_negative.py" "validation/thead-zw810e-problem29/scripts/test_comparator_negative.py"
upload_and_record "$W_SCRIPTS/probe11_sdpa_context.py" "validation/thead-zw810e-problem29/scripts/probe11_sdpa_context.py"
upload_and_record "$W_SCRIPTS/probe11_mask_vs_equal_length.py" "validation/thead-zw810e-problem29/scripts/probe11_mask_vs_equal_length.py"

# 2. Updated results (A/B seeded runs, AdamW verdicts, GELU attribution)
for f in accuracy_matrix.json adamw_stable_ref_acceptance.json adamw_stable_ref_v4.json orig_ab_reps_summary.json probe11_masked_v2.json probe10_gemm_ablation.json probe11_sdpa_context.json; do
  upload_and_record "$W/results/$f" "validation/thead-zw810e-problem29/results/$f"
done
for f in $(find "$W/results/orig_ab_reps" -name "*.json" -type f 2>/dev/null | sort); do
  bn=$(basename "$f")
  upload_and_record "$f" "validation/thead-zw810e-problem29/results/orig_ab_reps/$bn"
done

# 3. Updated README
upload_and_record "$R/platform-readme-r1b.md" "validation/thead-zw810e-problem29/README.md"

# 4. Logs
for f in adamw_stable_ref_v2_round12.log adamw_stable_ref_v2_fixed.log adamw_stable_ref_v2_solo.log probe11_v2.log probe11_ctx.stderr.log; do
  fp="$W/logs/$f"
  [ -f "$fp" ] && upload_and_record "$fp" "validation/thead-zw810e-problem29/logs/$f"
done

echo "ALL BLOBS DONE"
EOF
echo "upload script written"
