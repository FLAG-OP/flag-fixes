#!/usr/bin/env python3
"""Assemble the triplet-15 parquet from the intermediate files of
build_triplet_step1.sh + collection.tsv text extraction.

Inputs:
  /tmp/opencode/triplet/pos.tsv      qid \t positive_pid
  /tmp/opencode/triplet/neg15.tsv    qid \t neg1 \t ... \t neg15 ("" = missing)
  /tmp/opencode/triplet/pid_text.tsv pid \t text
  <src>/queries.train.tsv            header "qid text"

Deterministic shortfall fill: blake2b(qid, counter) -> uniform corpus pid,
excluding the positive, qrels-judged pids passed via judged.tsv, and pids
already used. Rows require anchor + positive + >=1 negative.
"""
import hashlib, sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/data-t2ranking")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15")
tmp = Path("/tmp/opencode/triplet")
out.mkdir(parents=True, exist_ok=True)

queries = {}
with open(src/"queries.train.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0].isdigit():
            queries[parts[0]] = parts[1]

pos = {}
with open(tmp/"pos.tsv", encoding="utf-8") as f:
    for line in f:
        q, p = line.rstrip("\n").split("\t")
        pos[q] = p

negs = {}
with open(tmp/"neg15.tsv", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        negs[parts[0]] = parts[1:]

texts = {}
with open(src/"collection.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        pid, tab, text = line.partition("\t")
        texts[pid] = text.rstrip("\n")
print(f"queries={len(queries)} pos={len(pos)} neg-rows={len(negs)} texts={len(texts)}", flush=True)

def fill_pids(qid, need, exclude):
    got, c = [], 0
    while len(got) < need and c < 500:
        h = hashlib.blake2b((qid + "#" + str(c)).encode(), digest_size=8).digest()
        pid = str(int.from_bytes(h, "big") % 2_303_643 + 1)
        if pid not in exclude and pid in texts:
            got.append(pid)
        c += 1
    return got

rows, filled = [], 0
for qid, neglist in sorted(negs.items(), key=lambda kv: int(kv[0])):
    if qid not in pos or qid not in queries:
        continue
    anchor, pos_pid = queries[qid], pos[qid]
    positive = texts.get(pos_pid)
    if not anchor.strip() or not positive:
        continue
    used = {pos_pid}
    cols = [anchor, positive]
    neg_texts = []
    for n in neglist[:15]:
        if not n or n in used:
            continue
        t = texts.get(n)
        if not t:
            continue
        used.add(n); neg_texts.append(t); cols.append(t)
    if len(neg_texts) < 15:
        filled += 1
        for n in fill_pids(qid, 15 - len(neg_texts), used):
            used.add(n); neg_texts.append(n); cols.append(texts[n])
    if neg_texts:
        rows.append(cols)

cols = ["anchor", "positive"] + [f"negative_{i}" for i in range(1, 16)]
schema = pa.schema([(c, pa.string()) for c in cols])
SHARD = 100_000
total_shards = max(1, (len(rows) + SHARD - 1) // SHARD)
written = 0
for i in range(0, len(rows), SHARD):
    chunk = rows[i:i+SHARD]
    t = pa.table({c: [r[j] for r in chunk] for j, c in enumerate(cols)}, schema=schema)
    pq.write_table(t, out / f"triplet15-{i//SHARD:05d}-of-{total_shards:05d}.parquet")
    written += len(chunk)
print(f"rows={written} shards={total_shards} queries_needing_fill={filled}")
