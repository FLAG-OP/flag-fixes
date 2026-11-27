#!/usr/bin/env python3
"""Regression tests for opt-in negative token-length regrouping."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import torch
from torch import nn


FIX_DIR = Path(__file__).resolve().parents[1]
SOURCE = FIX_DIR / "src" / "flagscale" / "train" / "losses" / "retrieval.py"


def load_loss_module():
    target = os.environ.get("FLAGSCALE_TREE")
    if target:
        sys.path.insert(0, target)
        return importlib.import_module("flagscale.train.losses.retrieval")

    fake_model_package = ModuleType("flagscale.models.retrieval")
    fake_model_package.unwrap_model = lambda model: model
    sys.modules["flagscale"] = ModuleType("flagscale")
    sys.modules["flagscale.models"] = ModuleType("flagscale.models")
    sys.modules["flagscale.models.retrieval"] = fake_model_package
    spec = importlib.util.spec_from_file_location(
        "flagscale.train.losses.retrieval", SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LengthEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prepare_calls = 0

    def prepare_embedding_inputs(self, inputs: list[str]) -> dict[str, torch.Tensor]:
        self.prepare_calls += 1
        lengths = torch.tensor([len(text.split()) for text in inputs], dtype=torch.long)
        width = int(lengths.max().item())
        input_ids = torch.arange(1, width + 1).unsqueeze(0).repeat(len(inputs), 1)
        attention_mask = (
            torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
        ).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        lengths = features["attention_mask"].sum(dim=-1)
        return {"sentence_embedding": lengths.unsqueeze(-1).float()}


def make_batch() -> dict[str, list[str]]:
    return {
        "anchor": ["a " * 4, "a " * 9, "a " * 2],
        "positive": ["p " * 5, "p " * 3, "p " * 8],
        "negative_1": ["n1 " * 12, "n1 " * 1, "n1 " * 7],
        "negative_2": ["n2 " * 2, "n2 " * 11, "n2 " * 4],
    }


def run_once(loss_module, env_value: str | None) -> tuple[float, int]:
    os.environ.pop("FLAGSCALE_NEGATIVE_LENGTH_REGROUP", None)
    if env_value is not None:
        os.environ["FLAGSCALE_NEGATIVE_LENGTH_REGROUP"] = env_value
    model = LengthEncoder()
    batch = make_batch()
    loss = loss_module.embedding_loss(model, batch, torch.device("cpu"), 0.05)
    return float(loss.detach()), model.prepare_calls


def main() -> int:
    if not SOURCE.exists():
        print(f"FAIL: source snapshot missing: {SOURCE}")
        return 1

    loss_module = load_loss_module()
    legacy_loss, legacy_calls = run_once(loss_module, None)
    regrouped_loss, regrouped_calls = run_once(loss_module, "1")

    if not torch.isfinite(torch.tensor(legacy_loss)):
        print("FAIL: legacy loss is not finite")
        return 1
    if not torch.allclose(
        torch.tensor(legacy_loss), torch.tensor(regrouped_loss), rtol=0, atol=1e-7
    ):
        print(f"FAIL: loss changed: {legacy_loss} -> {regrouped_loss}")
        return 1
    if regrouped_calls >= legacy_calls:
        print(
            "FAIL: regrouping did not flatten negative tokenization: "
            f"calls {legacy_calls} -> {regrouped_calls}"
        )
        return 1

    print(
        "PASS: default path unchanged; opt-in regrouping preserves loss and "
        f"reduces tokenization calls ({legacy_calls} -> {regrouped_calls})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
