#!/usr/bin/env bash
# Step 1+2: per-query BM25 top-15 negatives (false-negative guarded) and
# highest-relevance positive, from the official THUIR/T2Ranking release.
# Deterministic rule documented in the FlagScale retrieval triplet-15 rebuild.
#   qrels.train.tsv: header "qid - pid rel", tab-separated
#   train.bm25.tsv : "qid pid index" (BM25 rank), tab-separated
set -euo pipefail
SRC=${1:-/workspace/data-t2ranking}
TMP=/tmp/opencode/triplet
mkdir -p "$TMP"

echo "[1] positives (highest rel, ties -> smallest pid, rel>=1)"
awk -F'\t' 'NR>1 { rel=$4+0; pid=$3+0
    if (!($1 in best) || rel>bestr[$1] || (rel==bestr[$1] && pid<best[$1])) { best[$1]=pid; bestr[$1]=rel } }
  END { for (q in best) if (bestr[q]>=1) print q"\t"best[q] }' "$SRC/qrels.train.tsv" > "$TMP/pos.tsv"
wc -l "$TMP/pos.tsv"

echo "[2] BM25 top-15 negatives excluding all qrels-judged pids for the query"
awk -F'\t' '
  FILENAME == ARGV[1] { if (NR>1) { qrel[$1,$3]=1; if (!($1 in qseen)) { qseen[$1]=1; qlist[++nq]=$1 } } next }
  FILENAME == ARGV[2] {
    if (FNR == 1) next
    if (!($1 in qseen)) next
    if (($3+0) > 40) next
    if (($1,$2) in qrel) next
    ncnt[$1] += 1
    if (ncnt[$1] <= 15) neg[$1,ncnt[$1]] = $2
    next
  }
  END {
    for (i=1;i<=nq;i++) { q=qlist[i]; line=q; ok=0
      for (j=1;j<=15;j++) { v = ((q,j) in neg) ? neg[q,j] : ""; line = line "\t" v; if (v != "") ok++ }
      if (ok>=1) print line } }
' "$SRC/qrels.train.tsv" "$SRC/train.bm25.tsv" > "$TMP/neg15.tsv"
wc -l "$TMP/neg15.tsv"
