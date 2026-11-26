#!/bin/bash
# records delivery: split the 7.25 GB records.pkl, verify the rejoin, upload to a DRAFT release (retries)
set -u
S=/tmp/opencode/records_split; SRC=/tmp/opencode/zw810e-problem29/results/acc_hit_step_r44.json.records.pkl; LOG=/tmp/opencode/zw810e-problem29/logs/r58_records_upload.log
TAG=problem29-dw-records-r44
cd $S
[ -f records.part.03 ] && [ "$(stat -c %s records.part.03)" = "1271077117" ] || split -b 1900M -d -a 2 $SRC records.part.; echo "split present $(date -u +%H:%M:%S)" >> $LOG
sha256sum records.part.* > SHA256SUMS.parts
echo "7d00baa38532d5ad2a02c6681283a6a039448777ff1bb003c1daf8e3af8a5ec3  acc_hit_step_r44.json.records.pkl" > SHA256SUMS
ls -la $S >> $LOG 2>&1
echo "rejoin check: $(cat records.part.00 records.part.01 records.part.02 records.part.03 | sha256sum)" >> $LOG
if ! gh release view $TAG --repo FLAG-OP/flag-fixes --json id --jq .id > $S/release_id.txt 2>>$LOG; then
  gh release create $TAG --repo FLAG-OP/flag-fixes --draft --title "problem29 dW accuracy records (r44, 7.25 GB, 4 parts)" --notes-file $S/RELEASE_NOTES.md --target issue7-zw810e-opt-round1 >> $LOG 2>&1 && echo "draft release created $(date -u +%H:%M:%S)" >> $LOG
fi
for p in RELEASE_NOTES.md SHA256SUMS SHA256SUMS.parts records.part.03 records.part.00 records.part.01 records.part.02; do
  ok=0
  for t in 1 2 3 4 5; do
    T0=$(date +%s)
    if gh release upload $TAG $S/$p --clobber --repo FLAG-OP/flag-fixes >> $LOG 2>&1; then echo "uploaded $p try $t in $(( $(date +%s)-T0 ))s $(date -u +%H:%M:%S)" >> $LOG; ok=1; break; fi
    echo "FAILED $p try $t after $(( $(date +%s)-T0 ))s" >> $LOG; sleep 20
  done
  [ $ok -eq 1 ] || echo "GIVE_UP $p" >> $LOG
done
echo "UPLOAD_DONE $(date -u +%H:%M:%S)" >> $LOG
