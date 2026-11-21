#!/bin/bash
# round-8: background downloader for original-problem29 assets (bge-m3 weights + t2ranking triplet-15)
set -u
mkdir -p /tmp/t2ranking/t2ranking/triplet-15
cd /tmp/bge-m3-hf || exit 1
(
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  timeout 900 curl -sSL -C - -o pytorch_model.bin "https://hf-mirror.com/BAAI/bge-m3/resolve/main/pytorch_model.bin" 2>/dev/null
  SZ=$(stat -c%s pytorch_model.bin 2>/dev/null || echo 0)
  echo "model attempt $i size=$SZ" >> /tmp/dl_model.log
  [ "$SZ" -ge 2270000000 ] && break
  sleep 8
done
echo MODEL_DONE >> /tmp/dl_model.log
) &
cd /tmp/t2ranking/t2ranking/triplet-15 || exit 1
(
for i in 00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010; do
  F="train-$i-of-00011.parquet"
  for t in 1 2 3 4 5 6; do
    timeout 600 curl -sSL -C - -o "$F" "https://hf-mirror.com/datasets/sentence-transformers/t2ranking/resolve/main/triplet-15/$F" 2>/dev/null && break
    sleep 6
  done
  echo "$i: $(stat -c%s "$F" 2>/dev/null)" >> /tmp/dl_data.log
done
echo DATA_DONE >> /tmp/dl_data.log
) &
wait
echo ALL_DONE >> /tmp/dl_data.log
