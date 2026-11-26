"""FSDP2 worker for test_reranker_fsdp2_step.py (launched through torchrun).

Mirrors the trainer's FSDP2 wiring: fp32 model built on CPU, gradient
checkpointing, ``parallel_utils.apply_fsdp2`` with the same arguments as
``train_retrieval.train`` (bf16 compute/reduce on the device), the local shard
moved in the master dtype, AdamW, bf16 autocast, and the real ``reranker_loss``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reranker_common import REAL_CONFIG, StubTokenizer, lm_head_probe, random_texts  # noqa: E402

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
GIB = 2**30
TRUE_ID, FALSE_ID = 9693, 2152  # "yes" / "no" in the Qwen3 vocabulary


def log(message: str) -> None:
    if RANK == 0:
        print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    on_cuda = args.device == "cuda"
    device = torch.device("cuda", LOCAL_RANK) if on_cuda else torch.device("cpu")
    backend = args.backend or ("nccl" if on_cuda else "gloo")
    if on_cuda:
        torch.cuda.set_device(LOCAL_RANK)
    # Same initialisation order as train_retrieval.init_distributed.
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    from transformers import AutoConfig, AutoModelForCausalLM

    from flagscale.models.retrieval.qwen3_reranker import DEFAULT_INSTRUCTION, Qwen3RerankerModel
    from flagscale.train.losses.retrieval import reranker_loss
    from flagscale.train.utils import parallel as parallel_utils
    from torch.distributed.tensor import DTensor, Shard

    hidden = args.hidden
    config = AutoConfig.for_model(
        "qwen3",
        hidden_size=hidden,
        intermediate_size=3 * hidden,
        num_attention_heads=hidden // 128,
        num_key_value_heads=max(1, hidden // 512),
        head_dim=128,
        # the real vocabulary keeps embed_tokens / lm_head above the 1 GiB
        # "oversized leaf" threshold, so lm_head is its own FSDP module as in
        # the 8B model
        vocab_size=REAL_CONFIG["vocab_size"],
        max_position_embeddings=REAL_CONFIG["max_position_embeddings"],
        rope_theta=REAL_CONFIG["rope_theta"],
        rms_norm_eps=REAL_CONFIG["rms_norm_eps"],
        tie_word_embeddings=False,
        attention_bias=False,
        hidden_act=REAL_CONFIG["hidden_act"],
    )
    config.num_hidden_layers = args.layers
    torch.manual_seed(0)
    backbone = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model = Qwen3RerankerModel(backbone, StubTokenizer(vocab=config.vocab_size), args.max_length,
                               TRUE_ID, FALSE_ID, False, DEFAULT_INSTRUCTION)
    model.enable_gradient_checkpointing()

    compute_dtype = torch.bfloat16 if on_cuda else torch.float32
    model = parallel_utils.apply_fsdp2(
        model, device_type=args.device, world_size=max(WORLD_SIZE, 1), compute_dtype=compute_dtype,
        reduce_dtype=compute_dtype, log=log,
    )
    model = model.to(device=device, dtype=torch.float32)
    log(f"FSDP2 applied over {WORLD_SIZE} rank(s); lm_head weight is a {type(model.backbone.lm_head.weight).__name__}")

    checks: dict[str, object] = {}

    def check(name: str, ok: bool, detail) -> None:
        checks[name] = {"pass": bool(ok), "detail": detail}

    def is_shard_dtensor(tensor) -> bool:
        return isinstance(tensor, DTensor) and isinstance(tensor.placements[0], Shard)

    lm_head = model.backbone.lm_head
    check("lm_head_sharded", is_shard_dtensor(lm_head.weight), f"placements={getattr(lm_head.weight, 'placements', None)}")
    check("block_sharded", is_shard_dtensor(model.backbone.model.layers[0].self_attn.q_proj.weight),
          "first block q_proj is a Shard DTensor")
    # lm_head is a separately managed FSDP module, so its own pre/post-forward
    # hooks carry the all-gather/reshard. That is why the adapter has to call
    # the module rather than F.linear on its weight.
    check("lm_head_has_fsdp_hooks",
          len(lm_head._forward_pre_hooks_with_kwargs) >= 1 and len(lm_head._forward_hooks) >= 1,
          {"pre_hooks_with_kwargs": len(lm_head._forward_pre_hooks_with_kwargs),
           "post_hooks": len(lm_head._forward_hooks)})

    def batch_for(step: int, micro: int) -> dict[str, list[str]]:
        anchors = random_texts(args.rows, seed=100 + 10 * step + micro)
        positives = random_texts(args.rows, seed=200 + 10 * step + micro)
        negatives = ["负" + text for text in random_texts(args.rows, seed=300 + 10 * step + micro)]
        return {"anchor": anchors, "positive": positives, "negative_1": negatives}

    def loss_for(batch) -> torch.Tensor:
        if on_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return reranker_loss(model, batch, device)
        return reranker_loss(model, batch, device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.01)
    losses = []
    peak_allocated = 0
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    with lm_head_probe(model) as calls:
        for step in range(args.steps):
            # Same shape as train_epoch with gradient_accumulation_steps=2: the
            # first micro-batch skips the gradient sync, the window-end one syncs.
            optimizer.zero_grad(set_to_none=True)
            model.train()
            with parallel_utils.gradient_sync_disabled(model):
                loss_a = loss_for(batch_for(step, 0)) / 2
                loss_a.backward()
            loss_b = loss_for(batch_for(step, 1)) / 2
            loss_b.backward()
            if on_cuda:
                torch.cuda.synchronize()
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            losses.append(round((loss_a + loss_b).item(), 6))

            missing, nonfinite, norms = [], [], {}
            for name, parameter in model.named_parameters():
                grad = parameter.grad
                if grad is None:
                    missing.append(name)
                    continue
                local = grad.to_local() if hasattr(grad, "to_local") else grad
                if not bool(torch.isfinite(local).all()):
                    nonfinite.append(name)
                elif name in ("backbone.lm_head.weight", "backbone.model.embed_tokens.weight",
                              "backbone.model.norm.weight", "backbone.model.layers.0.self_attn.q_proj.weight"):
                    norms[name] = round(float(local.float().norm()), 6)
            # lm_head is sharded along the vocabulary. Only the two scored rows
            # ("yes"/"no") ever receive gradient, so every shard that does not
            # own them is exactly zero -- with the baseline as well.
            rows_per_rank = lm_head.weight.to_local().shape[0]
            owns_scored_rows = all(
                RANK * rows_per_rank <= token < (RANK + 1) * rows_per_rank for token in (TRUE_ID, FALSE_ID)
            )
            expected_nonzero = {
                "backbone.model.embed_tokens.weight", "backbone.model.norm.weight",
                "backbone.model.layers.0.self_attn.q_proj.weight",
            } | ({"backbone.lm_head.weight"} if owns_scored_rows else set())
            check(f"grads_present_finite_step{step}",
                  not missing and not nonfinite and len(norms) == 4
                  and all((norms[n] > 0) == (n in expected_nonzero) for n in norms),
                  {"missing": missing[:4], "nonfinite": nonfinite[:4], "grad_norms": norms,
                   "rank_owns_yes_no_rows": owns_scored_rows})

            lm_head_before = lm_head.weight.to_local().detach().clone()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            check(f"optimizer_updates_lm_head_step{step}",
                  not torch.equal(lm_head_before, lm_head.weight.to_local()), "lm_head local shard changed")
            log(f"step {step}: train_loss={losses[-1]}")

        # Evaluation forward through the same sharded model, then the baseline
        # formula (full logits) on the same weights as the reference.
        model.eval()
        pairs = list(zip(random_texts(4, 7), random_texts(4, 9)))
        features = {k: v.to(device) for k, v in model.prepare_reranker_inputs(pairs).items()}
        with torch.no_grad():
            if on_cuda:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    scores = model(features)["scores"].float()
            else:
                scores = model(features)["scores"].float()
        adapter_calls = list(calls)

        # The baseline formula, bound to the same module so the FSDP root's
        # pre-forward hook still runs (calling model.backbone() directly would
        # leave the root-managed norm weight sharded).
        def baseline_forward(self, features, **kwargs):
            logits = self.backbone(**features, return_dict=True, **kwargs).logits
            positions = features["attention_mask"].to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
            rows_index = torch.arange(logits.shape[0], device=logits.device)
            final_logits = logits[rows_index, positions]
            return {"scores": final_logits[:, self.true_token_id] - final_logits[:, self.false_token_id]}

        import types

        original_forward = model.forward
        model.forward = types.MethodType(baseline_forward, model)
        try:
            with torch.no_grad():
                if on_cuda:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        reference = model(features)["scores"].float()
                else:
                    reference = model(features)["scores"].float()
        finally:
            model.forward = original_forward

    max_abs = float((scores - reference).abs().max()) if scores.numel() else 0.0
    tolerance = 0.05 if on_cuda else 1e-3
    check("scores_match_full_logit_reference", scores.shape == reference.shape and max_abs <= tolerance,
          {"max_abs_diff": max_abs, "tolerance": tolerance, "shape": list(scores.shape),
           "note": "baseline passes trivially (it is the reference); this guards the fix"})

    expected_calls = 2 * args.steps + 1  # two micro-batches per step + the evaluation forward
    scored_only = [c for c in adapter_calls if c["input_dim"] == 2 and c["output_shape"][0] == c["input_shape"][0]]
    full_sequence = [c for c in adapter_calls if c["input_dim"] == 3]
    check("lm_head_called_once_per_forward", len(adapter_calls) == expected_calls,
          {"calls": len(adapter_calls), "expected": expected_calls})
    check("lm_head_projects_scored_rows_only", len(scored_only) == expected_calls and not full_sequence,
          {"calls": adapter_calls,
           "note": "baseline feeds [rows, seq, hidden] (full-sequence logits); the fix feeds [rows, hidden]"})
    check("gradient_sync_toggle_available", hasattr(model, "set_requires_gradient_sync"),
          "FSDP2 set_requires_gradient_sync present")

    record = {
        "rank": RANK, "world_size": WORLD_SIZE, "device": args.device, "backend": backend,
        "layers": args.layers, "hidden": hidden, "rows": args.rows, "max_length": args.max_length,
        "steps": args.steps, "losses": losses, "lm_head_calls": adapter_calls,
        "peak_allocated_gib": round(peak_allocated / GIB, 3), "checks": checks,
        "passed": all(c["pass"] for c in checks.values()),
    }
    Path(f"{args.json}.rank{RANK}").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    passed = all(c["pass"] for c in checks.values())
    if WORLD_SIZE > 1:
        dist.barrier()
    dist.destroy_process_group()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
