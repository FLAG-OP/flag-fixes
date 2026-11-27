#!/bin/bash
# download the smallest uploaded part back and verify its SHA-256 (end-to-end integrity check of the release upload)
LOG=/tmp/opencode/zw810e-problem29/logs/r58_records_verify.log; V=/tmp/opencode/records_verify
cd $V && rm -f records.part.03
T0=$(date +%s)
for t in 1 2 3; do
  if gh release download problem29-dw-records-r44 --repo FLAG-OP/flag-fixes --pattern records.part.03 --dir $V >> $LOG.detail 2>&1; then echo "downloaded records.part.03 try $t in $(( $(date +%s)-T0 ))s $(date -u +%H:%M:%S)" >> $LOG; break; fi
  echo "download FAILED try $t after $(( $(date +%s)-T0 ))s" >> $LOG; sleep 15
done
echo "sha256: $(sha256sum $V/records.part.03 | cut -d' ' -f1)  expected: $(grep 'records.part.03' /tmp/opencode/records_split/SHA256SUMS.parts | cut -d' ' -f1)  size: $(stat -c %s $V/records.part.03) expected: 1271077117" >> $LOG
echo "VERIFY_DONE $(date -u +%H:%M:%S)" >> $LOG
