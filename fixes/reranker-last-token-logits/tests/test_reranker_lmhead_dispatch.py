#!/usr/bin/env python3
"""Dispatch: the reranker adapter projects only the scored position through the LM head.

Runs the real FlagScale adapter on a tiny random Qwen3 on CPU (no checkpoint,
no accelerator).  A forward pre-hook on ``backbone.lm_head`` records what the
adapter feeds it; the baseline formula (full ``[batch, seq, vocab]`` logits,
last non-pad row kept) is the numerical reference.

Exit 0  every case passes
Exit 1  a case fails (baseline: lm_head receives ``[rows, seq, hidden]``)
Exit 2  dependency missing (torch / transformers / FlagScale tree) -- SKIP
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reranker_common import (  # noqa: E402
    EXIT_FAIL, EXIT_PASS, Skip, StubTokenizer, build_adapter, build_tiny_qwen3, full_logit_reference,
    lm_head_probe, random_texts, require_module,
)

# Large enough that short bodies are padded, not truncated: budget 158 tokens > the
# ~115-token body of the shortest random pair.
MAX_LENGTH = 400


def make_batch(rows: int, seed: int) -> dict[str, list[str]]:
    return {
        "anchor": random_texts(rows, seed),
        "positive": random_texts(rows, seed + 1),
        "negative_1": ["负" + text for text in random_texts(rows, seed + 2)],
    }


def pairs_of(batch: dict[str, list[str]]) -> list[list[str]]:
    """Same order as reranker_loss: every positive, then every negative."""
    return ([[a, p] for a, p in zip(batch["anchor"], batch["positive"])]
            + [[a, n] for a, n in zip(batch["anchor"], batch["negative_1"])])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    try:
        from flagscale.train.losses.retrieval import reranker_batch, reranker_loss  # noqa: E402

        require_module("transformers", "pip install transformers")
    except Skip as skip:
        print(f"SKIP: {skip}; exit 2")
        return 2

    torch.manual_seed(0)
    backbone = build_tiny_qwen3()
    # the stub vocabulary covers the adapter's token ids (3 / 7)
    tokenizer = StubTokenizer(vocab=1000, pad_id=0)
    adapter = build_adapter(backbone, tokenizer, MAX_LENGTH).eval()
    results: list[dict] = []

    def record(name: str, ok: bool, **detail) -> None:
        results.append({"case": name, "pass": bool(ok), **detail})

    def scored_rows_of(batch):
        """Forward through the real loss path with the lm_head probe active."""
        with lm_head_probe(adapter) as calls:
            scores = reranker_batch(adapter, pairs_of(batch), torch.device("cpu"))
        return scores, list(calls)

    # ---- shapes and positions -------------------------------------------------
    cases = [
        ("mixed_lengths", 5, 1),
        ("equal_lengths_all_truncated", 4, 2),   # every body longer than the budget
        ("single_sequence", 1, 3),
        ("two_sequences", 2, 4),
        ("batch_of_32_rows", 32, 5),
    ]
    for name, rows, seed in cases:
        batch = make_batch(rows, seed)
        if name == "equal_lengths_all_truncated":
            batch = {"anchor": ["x" * 200] * rows, "positive": ["y" * 200] * rows, "negative_1": ["z" * 200] * rows}
        scores, calls = scored_rows_of(batch)
        features = adapter.prepare_reranker_inputs(pairs_of(batch))
        reference = full_logit_reference(adapter, features)
        lengths = features["attention_mask"].sum(dim=1)
        max_abs = float((scores - reference).abs().max())
        padded = int(features["input_ids"].shape[1])
        record(
            f"{name}:scores_shape", scores.shape == (2 * rows,), shape=list(scores.shape), rows=rows,
        )
        record(
            f"{name}:scores_match_full_logit_reference", max_abs <= 1e-4, max_abs_diff=max_abs,
            padded_length=padded, min_real_length=int(lengths.min()), max_real_length=int(lengths.max()),
        )
        record(
            f"{name}:lm_head_called_once", len(calls) == 1, calls=len(calls),
        )
        record(
            f"{name}:lm_head_projects_scored_rows_only",
            len(calls) == 1 and calls[0]["input_dim"] == 2
            and calls[0]["input_shape"] == (2 * rows, backbone.config.hidden_size)
            and calls[0]["output_shape"] == (2 * rows, backbone.config.vocab_size),
            lm_head_calls=calls,
            full_sequence_logit_bytes=2 * rows * padded * backbone.config.vocab_size * 2,
            note="baseline feeds [rows, seq, hidden] and materialises [rows, seq, vocab]; the fix feeds [rows, hidden]",
        )

    # ---- gradient flow through the two-call forward ----------------------------
    adapter.train()
    batch = make_batch(4, 11)
    adapter.zero_grad(set_to_none=True)
    loss = reranker_loss(adapter, batch, torch.device("cpu"))
    loss.backward()
    watched = {
        "backbone.lm_head.weight", "backbone.model.embed_tokens.weight", "backbone.model.norm.weight",
        "backbone.model.layers.0.self_attn.q_proj.weight", "backbone.model.layers.1.mlp.down_proj.weight",
    }
    grads = {}
    for name, parameter in adapter.named_parameters():
        if name in watched:
            grads[name] = None if parameter.grad is None else round(float(parameter.grad.float().norm()), 6)
    record(
        "gradients:reach_lm_head_embedding_norm_blocks", all(g is not None and g > 0 for g in grads.values())
        and len(grads) == len(watched), grad_norms=grads, loss=round(float(loss), 6),
    )

    # ---- gradient checkpointing + training mode keep the same scores ------------
    adapter.zero_grad(set_to_none=True)
    if not adapter.enable_gradient_checkpointing():
        record("gradient_checkpointing:enabled", False, note="adapter refused gradient checkpointing")
    else:
        batch = make_batch(4, 13)
        pairs = pairs_of(batch)
        features = adapter.prepare_reranker_inputs(pairs)
        reference = full_logit_reference(adapter, features)
        adapter.train()
        with lm_head_probe(adapter) as calls:
            scores = adapter(features)["scores"].float()
            scores.sum().backward()
        max_abs = float((scores.detach() - reference).abs().max())
        record(
            "gradient_checkpointing:train_mode_scores_match_reference", max_abs <= 1e-4, max_abs_diff=max_abs,
            use_cache=bool(adapter.backbone.config.use_cache),
        )
        record(
            "gradient_checkpointing:lm_head_projects_scored_rows_only",
            len(calls) == 1 and calls[0]["input_dim"] == 2, lm_head_calls=list(calls),
        )

    # ---- kwargs pass through to the decoder ------------------------------------
    adapter.eval()
    batch = make_batch(2, 17)
    try:
        features = adapter.prepare_reranker_inputs(pairs_of(batch))
        with torch.no_grad():
            scores = adapter(features, use_cache=False)["scores"]
        record("kwargs:use_cache_false_accepted", scores.shape == (4,), shape=list(scores.shape))
    except TypeError as error:
        record("kwargs:use_cache_false_accepted", False, error=str(error)[:200])

    # ---- the score-head variant is untouched ------------------------------------
    score_adapter = build_adapter(build_tiny_qwen3(seed=1), StubTokenizer(vocab=1000), MAX_LENGTH, use_score_head=True).eval()
    with torch.no_grad():
        head_scores = score_adapter(score_adapter.prepare_reranker_inputs([["a", "b"], ["c", "d"]]))["scores"]
    record("score_head_variant:unchanged_shape", head_scores.shape == (2,), shape=list(head_scores.shape))

    failed = [r for r in results if not r["pass"]]
    summary = {
        "passed": not failed, "cases": len(results), "failed": [r["case"] for r in failed], "results": results,
        "torch": torch.__version__,
    }
    print("RESULT_JSON " + json.dumps(summary, ensure_ascii=False, default=str))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for item in failed:
        print(f"FAIL {item['case']}: {json.dumps({k: v for k, v in item.items() if k not in ('case', 'pass')}, default=str)[:500]}")
    if failed:
        print(f"dispatch test FAILED: {len(failed)}/{len(results)} cases; exit 1")
        return EXIT_FAIL
    print(f"dispatch OK: {len(results)} cases; lm_head receives only the scored rows; exit 0")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
