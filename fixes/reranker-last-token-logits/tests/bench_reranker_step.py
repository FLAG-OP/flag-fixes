#!/usr/bin/env python3
"""Benchmark: forward/backward time and peak memory of one reranker step.

Real Qwen3-Reranker-8B architecture (random bf16 weights, 36 layers), gradient
checkpointing, bf16 autocast, every pair truncated to ``max_length`` 640 -- the
padded shape of the reported failure at ``batch_size=32`` (64 pairs).  No
optimizer state, single card, so the numbers isolate the forward/backward of
the adapter.

Same process, same card, same inputs for every row of the table; the tree under
test is whatever ``PYTHONPATH`` points at.  Wall time is the median of
``--iters`` after ``--warmup`` (device synchronised around each phase); event
time is the CUDA-event elapsed time of the same phases.

Exit 0  benchmark completed (records; compare the JSON of the two trees)
Exit 2  precondition missing (accelerator / dependencies) -- SKIP, not a pass
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reranker_common import (  # noqa: E402
    REAL_CONFIG, QWEN3_CONFIG_KWARGS, Skip, StubTokenizer, lm_head_probe, require_device,
)

GIB = 2**30
TRUE_ID, FALSE_ID = 9693, 2152


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


def features_for(model, rows: int, device) -> dict[str, torch.Tensor]:
    body = " ".join(f"词{index}_{j}" for index in range(3) for j in range(300))  # longer than the budget
    pairs = [[f"问题{i} {body}", f"{body} 文{i}"] for i in range(rows * 2)]
    return {key: value.to(device) for key, value in model.prepare_reranker_inputs(pairs).items()}


def measure(model, features, device, warmup: int, iters: int) -> dict:
    labels = torch.cat([torch.ones(features["input_ids"].shape[0] // 2),
                        torch.zeros(features["input_ids"].shape[0] // 2)]).to(device)
    weights = torch.cuda.memory_allocated()
    fwd_wall, bwd_wall, fwd_event, bwd_event = [], [], [], []
    peaks = {}
    max_output_bytes = 0
    with lm_head_probe(model) as calls:
        for iteration in range(warmup + iters):
            model.train()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            calls.clear()
            start_event, mid_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
            wall_start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = model(features)["scores"].float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
            torch.cuda.synchronize()
            mid_wall = time.perf_counter()
            mid_event.record()
            after_forward = {
                "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 3),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 3),
            }
            loss.backward()
            torch.cuda.synchronize()
            end_wall = time.perf_counter()
            end_event.record()
            torch.cuda.synchronize()
            after_backward = {
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 3),
            }
            if iteration >= warmup:
                fwd_wall.append(mid_wall - wall_start)
                bwd_wall.append(end_wall - mid_wall)
                fwd_event.append(start_event.elapsed_time(mid_event) / 1000.0)
                bwd_event.append(mid_event.elapsed_time(end_event) / 1000.0)
                peaks = {"after_forward": after_forward, "after_backward": after_backward}
            max_output_bytes = max(max_output_bytes, max(c["output_bytes"] for c in calls))
    return {
        "forward_s_wall_median": round(statistics.median(fwd_wall), 3),
        "backward_s_wall_median": round(statistics.median(bwd_wall), 3),
        "forward_s_event_median": round(statistics.median(fwd_event), 3),
        "backward_s_event_median": round(statistics.median(bwd_event), 3),
        "forward_s_wall_all": [round(v, 3) for v in fwd_wall],
        "backward_s_wall_all": [round(v, 3) for v in bwd_wall],
        "weights_gib": round(weights / GIB, 3),
        "lm_head_max_output_bytes": max_output_bytes,
        "lm_head_max_output_gib": round(max_output_bytes / GIB, 4),
        **peaks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
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
    table = []
    for rows in args.rows:
        features = features_for(model, rows, device)
        torch.cuda.empty_cache()
        row = measure(model, features, device, args.warmup, args.iters)
        row.update({
            "rows": rows, "pairs": int(features["input_ids"].shape[0]),
            "padded_length": int(features["input_ids"].shape[1]),
            "tokens": int(features["attention_mask"].sum()),
        })
        table.append(row)
        print(f"rows={rows} pairs={row['pairs']} L={row['padded_length']}: "
              f"fwd {row['forward_s_wall_median']:.3f}s bwd {row['backward_s_wall_median']:.3f}s (wall median) | "
              f"fwd_peak {row['after_forward']['peak_allocated_gib']:.2f} GiB "
              f"bwd_peak {row['after_backward']['peak_allocated_gib']:.2f} GiB | "
              f"lm_head output {row['lm_head_max_output_gib']:.4f} GiB", flush=True)
        del features
    summary = {
        "device": torch.cuda.get_device_name(device), "torch": torch.__version__, "layers": args.layers,
        "iters": args.iters, "warmup": args.warmup, "table": table,
        "total_memory_gib": round(torch.cuda.get_device_properties(device).total_memory / GIB, 2),
    }
    print("RESULT_JSON " + json.dumps(summary, ensure_ascii=False, default=str))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
