#!/usr/bin/env python3
"""Run a real-model legacy/regroup A/B on one PPU device."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType

import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer


FIX_DIR = Path(__file__).resolve().parents[1]
SOURCE = FIX_DIR / "src" / "flagscale" / "train" / "losses" / "retrieval.py"


def load_loss_module():
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


class CLSEncoder(torch.nn.Module):
    def __init__(self, model, tokenizer, max_length: int) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def prepare_embedding_inputs(self, inputs: list[str]):
        return self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, features):
        output = self.model(**features, return_dict=True)
        return {"sentence_embedding": output.last_hidden_state[:, 0]}


def timed_round(loss_module, model, batch, device, temperature: float):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss = loss_module.embedding_loss(model, batch, device, temperature)
    backward_started = time.perf_counter()
    loss.backward()
    total_norm = torch.norm(
        torch.stack(
            [
                parameter.grad.norm()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
        )
    )
    torch.cuda.synchronize(device)
    finished = time.perf_counter()
    return {
        "loss": float(loss.detach()),
        "grad_norm": float(total_norm.detach()),
        "total_ms": (finished - started) * 1000,
        "forward_ms": (backward_started - started) * 1000,
        "backward_ms": (finished - backward_started) * 1000,
    }


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--mode", choices=["legacy", "regroup"], required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.rows <= 0 or args.warmup < 0 or args.rounds <= 0:
        raise SystemExit("rows/rounds must be positive; warmup must be non-negative")

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)

    table = pq.read_table(args.data)
    needed = ["anchor", "positive"] + [
        f"negative_{index}" for index in range(1, args.negatives + 1)
    ]
    indices = random.Random(args.seed).sample(range(table.num_rows), args.rows)
    rows = table.take(indices).select(needed).to_pylist()
    batch = {
        "anchor": [row["anchor"] for row in rows],
        "positive": [row["positive"] for row in rows],
        **{
            f"negative_{index}": [row[f"negative_{index}"] for row in rows]
            for index in range(1, args.negatives + 1)
        },
    }

    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoder = CLSEncoder(model, tokenizer, args.max_length).to(device).eval()
    loss_module = load_loss_module()

    os.environ.pop("FLAGSCALE_NEGATIVE_LENGTH_REGROUP", None)
    if args.mode == "regroup":
        os.environ["FLAGSCALE_NEGATIVE_LENGTH_REGROUP"] = "1"

    for _ in range(args.warmup):
        timed_round(loss_module, encoder, batch, device, args.temperature)

    rounds = []
    for _ in range(args.rounds):
        rounds.append(
            timed_round(loss_module, encoder, batch, device, args.temperature)
        )

    peak_memory = torch.cuda.max_memory_allocated(device)
    result = {
        "mode": args.mode,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "model": str(Path(args.model).resolve()),
        "data": str(Path(args.data).resolve()),
        "rows": args.rows,
        "negatives": args.negatives,
        "max_length": args.max_length,
        "dtype": "bfloat16",
        "warmup_rounds": args.warmup,
        "timed_rounds": args.rounds,
        "seed": args.seed,
        "row_indices": indices,
        "peak_memory_bytes": int(peak_memory),
        "losses": [round_item["loss"] for round_item in rounds],
        "grad_norms": [round_item["grad_norm"] for round_item in rounds],
        "total_ms": summarize([round_item["total_ms"] for round_item in rounds]),
        "forward_ms": summarize([round_item["forward_ms"] for round_item in rounds]),
        "backward_ms": summarize([round_item["backward_ms"] for round_item in rounds]),
        "raw_rounds": rounds,
    }

    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
