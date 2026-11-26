#!/usr/bin/env python3
"""Regression: the two-call forward scores the same positions as the full-logit formula.

Real Qwen3-Reranker-8B shapes (hidden 4096, vocabulary 151,669), random bf16
weights, bf16 autocast and gradient checkpointing as in the trainer.  For each
batch the adapter's ``scores`` and its parameter gradients are compared with
the baseline formula (materialise ``[batch, seq, vocab]`` logits, keep the last
non-pad row) on the same weights and the same tokenised batch.

Tolerances are rounding units of the compared quantity: scores within two bf16
ulp of the largest reference logit under autocast (1e-4 in fp32), gradients
within 3 % of the largest reference gradient under autocast (0.1 % in fp32).
The baseline tree passes this trivially -- its forward *is* the reference --
so the test guards the fix.

Exit 0  every case within tolerance
Exit 1  a case is out of tolerance
Exit 2  precondition missing (accelerator / dependencies) -- SKIP, not a pass
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reranker_common import (  # noqa: E402
    EXIT_FAIL, EXIT_PASS, REAL_CONFIG, QWEN3_CONFIG_KWARGS, Skip, StubTokenizer, lm_head_probe, require_device,
)

TRUE_ID, FALSE_ID = 9693, 2152  # "yes" / "no"
GIB = 2**30


def build(layers: int, device: torch.device, seed: int = 0):
    import transformers

    from flagscale.models.retrieval.qwen3_reranker import DEFAULT_INSTRUCTION, Qwen3RerankerModel

    config = transformers.AutoConfig.for_model("qwen3", **QWEN3_CONFIG_KWARGS)
    config.num_hidden_layers = layers
    torch.manual_seed(seed)
    with torch.device(device):
        backbone = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model = Qwen3RerankerModel(backbone, StubTokenizer(vocab=config.vocab_size), 640, TRUE_ID, FALSE_ID,
                               False, DEFAULT_INSTRUCTION)
    model.enable_gradient_checkpointing()
    return model.to(device)


def make_batch(rows: int, seed: int, full_length: bool) -> dict[str, list[str]]:
    def body(index: int, salt: int) -> str:
        if full_length:  # longer than the 640-token budget -> truncated
            return " ".join(f"词{index}_{salt}_{j}" for j in range(400))
        return "".join(chr(0x4e00 + (index * 7919 + salt * 31 + j * 104729) % 20000)
                       for j in range(40 + (index * 53 + salt * 17) % 500))

    return {
        "anchor": [f"问题{index} " + body(index, 0) for index in range(rows)],
        "positive": [body(index, 1) for index in range(rows)],
        "negative_1": ["负" + body(index, 2) for index in range(rows)],
    }


def features_of(model, data: dict[str, list[str]], device) -> dict[str, torch.Tensor]:
    """Same pair order as reranker_loss: all positives, then all negatives."""
    pairs = ([[a, p] for a, p in zip(data["anchor"], data["positive"])]
             + [[a, n] for a, n in zip(data["anchor"], data["negative_1"])])
    return {key: value.to(device) for key, value in model.prepare_reranker_inputs(pairs).items()}


def baseline_forward(self, features, **kwargs):
    """The formula the baseline adapter used (full-sequence logits)."""
    logits = self.backbone(**features, return_dict=True, **kwargs).logits
    positions = features["attention_mask"].to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
    rows = torch.arange(logits.shape[0], device=logits.device)
    final_logits = logits[rows, positions]
    return {"scores": final_logits[:, self.true_token_id] - final_logits[:, self.false_token_id]}


@contextlib.contextmanager
def baseline_adapter(model):
    original = model.forward
    model.forward = types.MethodType(baseline_forward, model)
    try:
        yield model
    finally:
        model.forward = original


def watched_names(layers: int) -> list[str]:
    names = ["backbone.lm_head.weight", "backbone.model.embed_tokens.weight", "backbone.model.norm.weight"]
    for layer in sorted({0, layers - 1}):
        names += [
            f"backbone.model.layers.{layer}.input_layernorm.weight",
            f"backbone.model.layers.{layer}.self_attn.o_proj.weight",
            f"backbone.model.layers.{layer}.mlp.down_proj.weight",
        ]
    return names


def watched_grads(model, layers: int) -> dict[str, torch.Tensor]:
    wanted = set(watched_names(layers))
    return {name: parameter.grad.detach().float() for name, parameter in model.named_parameters()
            if name in wanted and parameter.grad is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    try:
        device = require_device()
        __import__("flagscale.models.retrieval", fromlist=["__doc__"])
        __import__("transformers", fromlist=["__doc__"])
    except Skip as skip:
        print(f"SKIP: {skip}; exit 2")
        return 2

    model = build(args.layers, device)
    results: list[dict] = []

    def record(name, ok, **detail):
        results.append({"case": name, "pass": bool(ok), **detail})

    def autocast(on: bool):
        return torch.autocast("cuda", dtype=torch.bfloat16) if on else torch.autocast("cuda", enabled=False)

    def compare_scores(name, rows, seed, use_autocast, full_length):
        features = features_of(model, make_batch(rows, seed, full_length), device)
        model.eval()
        with torch.no_grad(), autocast(use_autocast):
            with lm_head_probe(model) as calls:
                scores = model(features)["scores"].float()
            adapter_calls = list(calls)
            with baseline_adapter(model) as baseline:
                reference = baseline(features)["scores"].float()
            final = model.backbone(**features, return_dict=True).logits[
                torch.arange(scores.shape[0], device=device),
                features["attention_mask"].to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1,
            ]
            largest = float(final.abs().max())
        max_abs = float((scores - reference).abs().max())
        tolerance = 2.0 * 2.0 ** (math.floor(math.log2(largest)) - 7) if use_autocast else 1e-4
        record(name, scores.shape == reference.shape and max_abs <= tolerance, max_abs_diff=max_abs,
               tolerance=tolerance, largest_reference_logit=largest, lm_head_calls=adapter_calls,
               pairs=2 * rows, padded_length=int(features["input_ids"].shape[1]))

    def compare_grads(name, rows, seed, use_autocast):
        features = features_of(model, make_batch(rows, seed, True), device)
        labels = torch.cat([torch.ones(rows), torch.zeros(rows)]).to(device)
        model.train()
        with autocast(use_autocast):
            model.zero_grad(set_to_none=True)
            torch.nn.functional.binary_cross_entropy_with_logits(model(features)["scores"].float(), labels).backward()
            adapter_grads = watched_grads(model, args.layers)
            model.zero_grad(set_to_none=True)
            with baseline_adapter(model) as baseline:
                torch.nn.functional.binary_cross_entropy_with_logits(
                    baseline(features)["scores"].float(), labels).backward()
            reference_grads = watched_grads(model, args.layers)
        worst_rel, worst_abs = 0.0, 0.0
        for key, reference in reference_grads.items():
            diff = (adapter_grads[key] - reference).abs()
            worst_abs = max(worst_abs, float(diff.max()))
            worst_rel = max(worst_rel, float(diff.max()) / max(float(reference.abs().max()), 1e-30))
        tolerance = 0.03 if use_autocast else 1e-3
        record(name, adapter_grads.keys() == reference_grads.keys() and worst_rel <= tolerance,
               worst_relative_diff=worst_rel, worst_abs_diff=worst_abs, tolerance=tolerance,
               compared=sorted(reference_grads), pairs=2 * rows)

    compare_scores("bf16_autocast:padded_mixed_length:16rows", 16, 1, True, False)
    compare_scores("bf16_autocast:full_length_640:32rows", 32, 2, True, True)
    compare_scores("fp32_no_autocast:padded_mixed_length:4rows", 4, 3, False, False)
    compare_grads("bf16_autocast:gradients:16rows", 16, 4, True)
    compare_grads("fp32_no_autocast:gradients:2rows", 2, 5, False)

    failed = [r for r in results if not r["pass"]]
    summary = {
        "passed": not failed, "layers": args.layers, "cases": len(results),
        "failed": [r["case"] for r in failed], "results": results, "torch": torch.__version__,
        "weights_gib": round(sum(p.numel() * p.element_size() for p in model.parameters()) / GIB, 3),
    }
    print("RESULT_JSON " + json.dumps(summary, ensure_ascii=False, default=str))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for item in failed:
        print(f"FAIL {item['case']}: "
              f"{json.dumps({k: v for k, v in item.items() if k not in ('case', 'pass')}, default=str)[:500]}")
    if failed:
        print(f"score equivalence FAILED: {len(failed)}/{len(results)} cases; exit 1")
        return EXIT_FAIL
    print(f"score equivalence OK: {len(results)} cases within tolerance; exit 0")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
