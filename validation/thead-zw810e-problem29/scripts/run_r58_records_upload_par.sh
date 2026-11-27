#!/bin/bash
# Parallel upload of records parts 01 and 02 (part.03 is done, part.00 is being uploaded by the
# process left over from the sequential script). The sequential upload ran at ~0.95 MB/s per stream.
S=/tmp/opencode/records_split; LOG=/tmp/opencode/zw810e-problem29/logs/r58_records_upload_par.log; TAG=problem29-dw-records-r44
one(){ p=$1; for t in 1 2 3 4 5; do T0=$(date +%s); if gh release upload $TAG $S/$p --clobber --repo FLAG-OP/flag-fixes >> $LOG.$p 2>&1; then echo "uploaded $p try $t in $(( $(date +%s)-T0 ))s $(date -u +%H:%M:%S)" >> $LOG; return; fi; echo "FAILED $p try $t after $(( $(date +%s)-T0 ))s" >> $LOG; sleep 20; done; echo "GIVE_UP $p" >> $LOG; }
one records.part.01 & one records.part.02 & wait
echo "PAR_UPLOAD_DONE $(date -u +%H:%M:%S)" >> $LOG
