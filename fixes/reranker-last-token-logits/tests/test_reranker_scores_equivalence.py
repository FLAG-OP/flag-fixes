#!/usr/bin/env python3
"""Regression: the two-call forward scores the same positions as the full-logit formula.

Real Qwen3-Reranker-8B shapes (hidden 4096, vocabulary 151,669), random weights, gradient
checkpointing as in the trainer.  For each batch the adapter's ``scores`` and its parameter
gradients are compared with the baseline formula (materialise ``[batch, seq, vocab]`` logits,
keep the last non-pad row) on the same weights and the same tokenised batch.

Every case records the dtype that actually computed: the weight dtype, and whether autocast
was on.  Tolerances follow that dtype:

    bf16 weights, autocast on or off   scores within two bf16 ulp of the largest reference
                                       logit; gradients within 3 % of the largest reference
                                       gradient
    fp32 weights, no autocast           scores 1e-4; gradients 0.1 %

The baseline tree passes this trivially -- its forward *is* the reference -- so the test
guards the fix.

The gradient comparison asserts the structure before it measures an error: both sides must
hold exactly the expected tensors (nine for the default 36 layers: lm_head, embedding, the
final norm and, on the first and the last block, input norm / o_proj / down_proj), and every
reference tensor, candidate tensor and their difference must be finite.  Only then is the
error computed.  The ``comparator_*`` cases feed corrupted copies of a real result to the
comparator and require it to reject them: a NaN gradient, the same expected gradient missing
on both sides, a gradient missing on one side, a non-finite difference.

Exit 0  every case within tolerance, every corrupted pair rejected
Exit 1  otherwise
Exit 2  precondition missing (accelerator / dependencies) -- SKIP, not a pass
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reranker_common import (  # noqa: E402
    EXIT_FAIL, EXIT_PASS, QWEN3_CONFIG_KWARGS, Skip, StubTokenizer, lm_head_probe, require_device,
)

TRUE_ID, FALSE_ID = 9693, 2152  # "yes" / "no"
GIB = 2**30
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def build(layers: int, device: torch.device, weights_dtype, seed: int = 0):
    import transformers

    from flagscale.models.retrieval.qwen3_reranker import DEFAULT_INSTRUCTION, Qwen3RerankerModel

    config = transformers.AutoConfig.for_model("qwen3", **QWEN3_CONFIG_KWARGS)
    config.num_hidden_layers = layers
    torch.manual_seed(seed)
    with torch.device(device):
        backbone = transformers.AutoModelForCausalLM.from_config(config, dtype=weights_dtype)
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
    return {name: parameter.grad.detach().float().clone() for name, parameter in model.named_parameters()
            if name in wanted and parameter.grad is not None}


def compare_gradients(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor],
                      expected: list[str]) -> dict:
    """Compare two sets of gradients.  Structure first, finiteness next, error last.

    ``complete``: both sides hold exactly the expected tensors.  ``finite``: every reference
    tensor, candidate tensor and their difference is finite.  The error is only meaningful
    when both hold; tensors that fail the finiteness check are left out of it.
    """
    outcome = {
        "expected": len(expected),
        "missing_reference": sorted(set(expected) - set(reference)),
        "missing_candidate": sorted(set(expected) - set(candidate)),
        "unexpected_reference": sorted(set(reference) - set(expected)),
        "unexpected_candidate": sorted(set(candidate) - set(expected)),
        "nonfinite_reference": [], "nonfinite_candidate": [], "nonfinite_difference": [],
    }
    worst_rel, worst_abs, worst_name = 0.0, 0.0, None
    for name in expected:
        if name not in reference or name not in candidate:
            continue
        ref, cand = reference[name].float(), candidate[name].float()
        reference_finite = bool(torch.isfinite(ref).all())
        candidate_finite = bool(torch.isfinite(cand).all())
        if not reference_finite:
            outcome["nonfinite_reference"].append(name)
        if not candidate_finite:
            outcome["nonfinite_candidate"].append(name)
        difference = cand - ref
        if not (reference_finite and candidate_finite and bool(torch.isfinite(difference).all())):
            outcome["nonfinite_difference"].append(name)
            continue
        absolute = float(difference.abs().max())
        relative = absolute / max(float(ref.abs().max()), 1e-30)
        if relative > worst_rel:
            worst_rel, worst_name = relative, name
        worst_abs = max(worst_abs, absolute)
    outcome["complete"] = not any(outcome[key] for key in (
        "missing_reference", "missing_candidate", "unexpected_reference", "unexpected_candidate"))
    outcome["finite"] = not any(outcome[key] for key in (
        "nonfinite_reference", "nonfinite_candidate", "nonfinite_difference"))
    outcome.update(worst_relative_diff=worst_rel, worst_abs_diff=worst_abs, worst_tensor=worst_name,
                   compared=len(expected))
    return outcome


def gradients_match(outcome: dict, tolerance: float) -> bool:
    return outcome["complete"] and outcome["finite"] and outcome["worst_relative_diff"] <= tolerance


def score_tolerance(largest: float, fp32_compute: bool) -> float:
    if fp32_compute:
        return 1e-4
    # two bf16 ulp of the largest reference logit (bf16 keeps 7 explicit mantissa bits)
    return 2.0 * 2.0 ** (math.floor(math.log2(max(largest, 1e-30))) - 7)


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

    expected = watched_names(args.layers)
    results: list[dict] = []

    def record(name, ok, **detail):
        results.append({"case": name, "pass": bool(ok), **detail})

    def autocast(on: bool):
        return torch.autocast("cuda", dtype=torch.bfloat16) if on else torch.autocast("cuda", enabled=False)

    def compare_scores(label, model, rows, seed, use_autocast, full_length):
        # fp32 weights without autocast are the only configuration that computes in fp32
        fp32_compute = next(model.parameters()).dtype is torch.float32 and not use_autocast
        features = features_of(model, make_batch(rows, seed, full_length), device)
        model.eval()
        with torch.no_grad(), autocast(use_autocast):
            with lm_head_probe(model) as calls:
                output = model(features)["scores"]
            score_dtype = str(output.dtype)
            scores = output.float()
            adapter_calls = list(calls)
            with baseline_adapter(model) as baseline:
                reference = baseline(features)["scores"].float()
            final = model.backbone(**features, return_dict=True).logits[
                torch.arange(scores.shape[0], device=device),
                features["attention_mask"].to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1,
            ]
            largest = float(final.abs().max())
        max_abs = float((scores - reference).abs().max())
        tolerance = score_tolerance(largest, fp32_compute)
        record(label, scores.shape == reference.shape and torch.isfinite(scores).all()
               and torch.isfinite(reference).all() and max_abs <= tolerance,
               weights_dtype=str(next(model.parameters()).dtype), autocast=use_autocast,
               score_dtype=score_dtype, max_abs_diff=max_abs, tolerance=tolerance,
               largest_reference_logit=largest, lm_head_calls=adapter_calls,
               pairs=2 * rows, padded_length=int(features["input_ids"].shape[1]))
        return features

    def compare_grads(label, model, rows, seed, use_autocast):
        fp32_compute = next(model.parameters()).dtype is torch.float32 and not use_autocast
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
        model.zero_grad(set_to_none=True)
        outcome = compare_gradients(reference_grads, adapter_grads, expected)
        tolerance = 1e-3 if fp32_compute else 0.03
        record(label, gradients_match(outcome, tolerance), weights_dtype=str(next(model.parameters()).dtype),
               autocast=use_autocast, tolerance=tolerance, **outcome)
        return reference_grads, adapter_grads, tolerance

    def comparator_cases(reference, candidate, tolerance):
        """Corrupted copies of a real result must be rejected; the uncorrupted copy accepted."""
        target = expected[3]  # a block tensor, not one of the three global ones

        def run(name, should_pass, edit):
            ref = {key: value.clone() for key, value in reference.items()}
            cand = {key: value.clone() for key, value in candidate.items()}
            edit(ref, cand)
            outcome = compare_gradients(ref, cand, expected)
            accepted = gradients_match(outcome, tolerance)
            record(name, accepted == should_pass, comparator_accepted=accepted,
                   corrupted_tensor=None if should_pass else target,
                   missing_reference=outcome["missing_reference"], missing_candidate=outcome["missing_candidate"],
                   nonfinite_reference=outcome["nonfinite_reference"], nonfinite_candidate=outcome["nonfinite_candidate"],
                   nonfinite_difference=outcome["nonfinite_difference"])

        def nan_in_candidate(ref, cand):
            cand[target].flatten()[0].fill_(float("nan"))

        def nan_in_reference(ref, cand):
            ref[target].flatten()[0].fill_(float("nan"))

        def missing_on_both_sides(ref, cand):
            ref.pop(target)
            cand.pop(target)

        def missing_on_candidate_side(ref, cand):
            cand.pop(target)

        def difference_overflows(ref, cand):  # both sides finite, fp32 difference infinite
            ref[target].fill_(3.0e38)
            cand[target].fill_(-3.0e38)

        def both_infinite(ref, cand):  # both sides +inf, difference NaN
            ref[target].fill_(float("inf"))
            cand[target].fill_(float("inf"))

        run("comparator_accepts:uncorrupted_pair", True, lambda ref, cand: None)
        run("comparator_rejects:nan_gradient", False, nan_in_candidate)
        run("comparator_rejects:nan_in_reference", False, nan_in_reference)
        run("comparator_rejects:both_sides_missing_one_expected_gradient", False, missing_on_both_sides)
        run("comparator_rejects:missing_on_one_side", False, missing_on_candidate_side)
        run("comparator_rejects:difference_overflows", False, difference_overflows)
        run("comparator_rejects:nan_difference_from_infinite_sides", False, both_infinite)

    weights_gib = {}
    for name, weights_dtype in DTYPES.items():
        model = build(args.layers, device, weights_dtype)
        weights_gib[name] = round(sum(p.numel() * p.element_size() for p in model.parameters()) / GIB, 3)
        prefix = f"{name}_weights"
        if name == "bf16":
            compare_scores(f"{prefix}:bf16_autocast:scores:padded_mixed_length:16rows", model, 16, 1, True, False)
            compare_scores(f"{prefix}:bf16_autocast:scores:full_length_640:32rows", model, 32, 2, True, True)
            compare_scores(f"{prefix}:no_autocast:scores:padded_mixed_length:4rows", model, 4, 3, False, False)
            reference, candidate, tolerance = compare_grads(f"{prefix}:bf16_autocast:gradients:16rows", model, 16, 4, True)
            comparator_cases(reference, candidate, tolerance)
            del reference, candidate
            compare_grads(f"{prefix}:no_autocast:gradients:2rows", model, 2, 5, False)
        else:
            compare_scores(f"{prefix}:no_autocast:scores:padded_mixed_length:4rows", model, 4, 3, False, False)
            compare_grads(f"{prefix}:no_autocast:gradients:2rows", model, 2, 5, False)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    failed = [r for r in results if not r["pass"]]
    summary = {
        "passed": not failed, "layers": args.layers, "cases": len(results),
        "failed": [r["case"] for r in failed], "results": results, "torch": torch.__version__,
        "weights_gib": weights_gib, "expected_gradients": len(expected),
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
    print(f"score equivalence OK: {len(results)} cases (scores, gradients, comparator self-checks); exit 0")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
