#!/bin/bash
# Upload records.part.00 (the first attempt died with HTTP 408 after the sequential runner was stopped).
S=/tmp/opencode/records_split; LOG=/tmp/opencode/zw810e-problem29/logs/r58_records_upload_part00.log; TAG=problem29-dw-records-r44
for t in 1 2 3 4 5 6; do
  T0=$(date +%s)
  if gh release upload $TAG $S/records.part.00 --clobber --repo FLAG-OP/flag-fixes >> $LOG.detail 2>&1; then echo "uploaded records.part.00 try $t in $(( $(date +%s)-T0 ))s $(date -u +%H:%M:%S)" >> $LOG; break; fi
  echo "FAILED records.part.00 try $t after $(( $(date +%s)-T0 ))s $(tail -c 200 $LOG.detail | tr '\n' ' ')" >> $LOG; sleep 20
done
echo "PART00_DONE $(date -u +%H:%M:%S)" >> $LOG
