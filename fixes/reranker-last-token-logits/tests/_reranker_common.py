"""Shared helpers for the reranker-last-token-logits verification scripts.

The scripts exercise the real FlagScale adapter
(``flagscale.models.retrieval.qwen3_reranker.Qwen3RerankerModel``) on random
weights: the allocation pattern under test is shape-determined, not
weight-determined, so no 16 GB checkpoint is needed.  Token ids likewise do not
matter for shapes, so ``StubTokenizer`` replaces the Qwen2 tokenizer (one id
per character, right padding, deterministic).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# config.json of Qwen/Qwen3-Reranker-8B (published checkpoint), verbatim.
REAL_CONFIG = json.loads((HERE / "fixtures" / "qwen3_reranker_8b_config.json").read_text())
# The architecture fields, usable as Qwen3Config / AutoConfig.for_model keyword arguments.
QWEN3_CONFIG_KWARGS = {
    key: value for key, value in REAL_CONFIG.items()
    if key not in {"architectures", "model_type", "transformers_version", "torch_dtype"}
}
EXIT_PASS, EXIT_FAIL, EXIT_SKIP = 0, 1, 2


class Skip(Exception):
    """Raised when a precondition is missing; maps to exit code 2 (not a pass)."""


def require_device(index: int | None = None) -> torch.device:
    if os.environ.get("RERANKER_TEST_DEVICE", "").lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise Skip("no accelerator device (torch.cuda.is_available() is False)")
    device = torch.device("cuda", int(os.environ.get("RERANKER_TEST_DEVICE_INDEX", index or 0)))
    torch.cuda.set_device(device)
    return device


def require_module(name: str, hint: str):
    try:
        return __import__(name, fromlist=["__doc__"])
    except Exception as error:  # pragma: no cover - environment dependent
        raise Skip(f"{name} is not importable ({type(error).__name__}: {error}); {hint}")


def gib(value: float) -> float:
    return round(value / 2**30, 3)


def full_logit_reference(adapter, features) -> torch.Tensor:
    """Score ``features`` the way the baseline adapter did.

    Materialises the causal LM's ``[batch, seq, vocab]`` logits and keeps the
    last non-pad row of each sequence.  Independent of the adapter's forward.
    """
    backbone = adapter.backbone
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        logits = backbone(**features, return_dict=True).logits
        positions = features["attention_mask"].to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
        rows = torch.arange(logits.shape[0], device=logits.device)
        final = logits[rows, positions]
        scores = final[:, adapter.true_token_id] - final[:, adapter.false_token_id]
    backbone.train(was_training)
    return scores.float()


class StubTokenizer:
    """Deterministic token ids: one per character, truncated to ``max_length``.

    The adapter tokenises the prompt body, prepends ``_prefix_ids`` and appends
    ``_suffix_ids``, then pads on the right -- exactly what it does with the
    real tokenizer.  Only the resulting shapes matter here.
    """

    def __init__(self, vocab: int = 1000, pad_id: int = 0) -> None:
        self.vocab = vocab
        self.pad_id = pad_id

    def __call__(self, texts, padding=False, truncation=None, return_attention_mask=False,
                 max_length=None, add_special_tokens=False, **_):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        rows = []
        for text in items:
            ids = [1 + (ord(ch) * 7919 + index * 104729) % (self.vocab - 1)
                   for index, ch in enumerate(text)]
            if max_length is not None:
                ids = ids[:max_length]
            rows.append(ids)
        return {"input_ids": rows[0] if single else rows}

    def pad(self, encoded, padding=True, return_tensors="pt", **_):
        rows = encoded["input_ids"]
        width = max(len(row) for row in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[index, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def build_tiny_qwen3(seed: int = 0):
    """A 2-layer Qwen3 with the real adapter layout (untied lm_head)."""
    transformers = require_module("transformers", "pip install transformers")
    config = transformers.Qwen3Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, vocab_size=1000, max_position_embeddings=4096,
        rope_theta=1000000.0, rms_norm_eps=1e-06, tie_word_embeddings=False,
        attention_bias=False, attention_dropout=0.0,
    )
    torch.manual_seed(seed)
    backbone = transformers.Qwen3ForCausalLM(config).to(torch.float32)
    return backbone


def build_adapter(backbone, tokenizer, max_length: int, use_score_head: bool = False):
    models = require_module("flagscale.models.retrieval", "run with PYTHONPATH pointing at the FlagScale tree")
    from flagscale.models.retrieval.qwen3_reranker import DEFAULT_INSTRUCTION, Qwen3RerankerModel

    return Qwen3RerankerModel(
        backbone, tokenizer, max_length, 3, 7, use_score_head, DEFAULT_INSTRUCTION
    )


def random_texts(count: int, seed: int = 1) -> list[str]:
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz 一二三四五六七八九十"
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40))) for _ in range(count)]


def lm_head_probe(adapter):
    """Record every call of ``backbone.lm_head``: input shape and output nbytes."""
    from contextlib import contextmanager

    calls: list[dict] = []

    def pre_hook(module, args, kwargs):
        tensor = args[0] if args else next(iter(kwargs.values()))
        calls.append({"input_shape": tuple(tensor.shape), "input_dim": tensor.dim()})

    def post_hook(module, args, output):
        calls[-1]["output_shape"] = tuple(output.shape)
        calls[-1]["output_bytes"] = output.numel() * output.element_size()

    @contextmanager
    def active():
        handle_pre = adapter.backbone.lm_head.register_forward_pre_hook(pre_hook, with_kwargs=True)
        handle_post = adapter.backbone.lm_head.register_forward_hook(post_hook)
        try:
            yield calls
        finally:
            handle_pre.remove()
            handle_post.remove()

    return active()


def print_json(record: dict) -> None:
    print("RESULT_JSON " + json.dumps(record, ensure_ascii=False, default=str))
