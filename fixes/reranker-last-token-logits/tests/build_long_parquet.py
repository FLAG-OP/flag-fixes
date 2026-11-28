#!/usr/bin/env python3
"""Build the long-document T2Ranking subset used by the natural OOM reproduction.

Keeps the rows whose positive AND negative prompts both exceed the reranker's
``max_length`` budget, so every pair is truncated to exactly ``max_length``
(640) and every batch has the padded shape of the reported failure.

    python3 tests/build_long_parquet.py --tokenizer-model /path/Qwen3-Reranker-8B \\
        --source .../triplet-15/train-00000-of-00011.parquet --out .../long/train-long-00000.parquet

Needs the FlagScale tree on ``PYTHONPATH`` (the adapter's prompt framing) and
the real tokenizer.  On shard 0 (30,920 rows) this keeps 454 rows
(414 train / 13 validation / 27 test under the trainer's split rule).
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer-model", required=True, help="checkpoint dir with the Qwen2 tokenizer")
    parser.add_argument("--source", required=True, help="T2Ranking triplet parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-length", type=int, default=640)
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer  # noqa: F401

        from flagscale.models.retrieval.qwen3_reranker import DEFAULT_INSTRUCTION, RERANK_SYSTEM  # noqa: F401
        from flagscale.train.datasets.retrieval import split_accepts  # noqa: F401
    except Exception as error:
        print(f"SKIP: transformers / FlagScale not importable ({type(error).__name__}: {error}); exit 2")
        return 2
    from transformers import AutoTokenizer

    from flagscale.models.retrieval.qwen3_reranker import DEFAULT_INSTRUCTION, RERANK_SYSTEM
    from flagscale.train.datasets.retrieval import split_accepts

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    # the adapter's exact framing: prefix + body (truncated to the budget) + suffix
    prefix = tokenizer("<|im_start|>system\n" + RERANK_SYSTEM + "<|im_end|>\n<|im_start|>user\n",
                       add_special_tokens=False)["input_ids"]
    suffix = tokenizer("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                       add_special_tokens=False)["input_ids"]
    budget = max(16, args.max_length - len(prefix) - len(suffix))
    print(f"prefix/suffix/budget tokens: {len(prefix)}/{len(suffix)}/{budget}")

    table = pq.read_table(args.source, columns=["anchor", "positive", "negative_1"])
    anchors = table.column("anchor").to_pylist()
    positives = table.column("positive").to_pylist()
    negatives = table.column("negative_1").to_pylist()

    def body(query: str, document: str) -> str:
        return "<Instruct>: {i}\n<Query>: {q}\n<Document>: {d}".format(i=DEFAULT_INSTRUCTION, q=query, d=document)

    def lengths(rows: list[int], column: list[str]) -> dict[int, int]:
        texts = [body(anchors[i], column[i]) for i in rows]
        return dict(zip(rows, (len(x) for x in tokenizer(texts, add_special_tokens=False)["input_ids"])))

    candidates = [i for i in range(len(anchors)) if len(positives[i]) >= 500 and len(negatives[i]) >= 500]
    print(f"rows {len(anchors)}, char prefilter (>= 500 chars in both): {len(candidates)}")
    keep = []
    for start in range(0, len(candidates), 200):
        rows = candidates[start:start + 200]
        pos_len, neg_len = lengths(rows, positives), lengths(rows, negatives)
        keep += [i for i in rows if pos_len[i] >= budget and neg_len[i] >= budget]
    print(f"rows with both prompts >= budget (truncated to max_length): {len(keep)}")
    splits = collections.Counter(split for i in keep for split in ("train", "validation", "test")
                                 if split_accepts(i, split, 20, group_key=anchors[i]))
    print(f"split rule (modulo 20, by anchor): {dict(splits)}")
    if len(keep) < 64:
        print("too few long rows in this shard; exit 1")
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(pa.array(keep, type=pa.int64())), args.out)
    print(f"wrote {args.out} ({len(keep)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
