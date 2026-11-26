#!/usr/bin/env python3
"""Repro: one reranker step asks the allocator for a single 11.57 GiB block.

The reported failure (``conf/train/qwen3_reranker_p800.yaml``):

    Tried to allocate 11.57 GiB. 96.00 GiB total, 20.90 GiB free,
    71.68 GiB already allocated by PyTorch

at ``batch_size=32``.  64 pairs x 640 tokens x 151,669 vocabulary x 2 bytes
(bf16) = 11.57 GiB is the adapter's full-sequence ``logits`` -- the causal LM
projects every position and the adapter keeps one row per sequence.  The
``lm_head`` all-gather the config comment blames is 1.16 GiB.

Part 1 (clean card): hooks on ``backbone.lm_head`` record what one forward
produces.

Part 2 (heap shaped like step >= 2 of the real trainer): bf16 weights plus
three stand-ins per parameter (gradient, exp_avg, exp_avg_sq) reproduce the
61.0 GiB that the real trainer has resident after its first optimizer step
(61.03 GiB measured, results/e2e-*).  The remaining free memory is cut into
uneven cached blocks, each smaller than 11.57 GiB.  Then one training step
through the real adapter and the real ``reranker_loss``:

  baseline  the 11.57 GiB request cannot be served -> OutOfMemoryError
  fix       the largest request of the step is ~1.2 GiB -> completes

Exit 0  the step completed on the fragmented heap and the LM head output is
        ``[rows, vocab]`` only
Exit 1  OutOfMemoryError on the step, or the LM head output is full-sequence
Exit 2  precondition missing (accelerator / dependencies / < 90 GiB card)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reranker_common import (  # noqa: E402
    EXIT_FAIL, EXIT_PASS, REAL_CONFIG, QWEN3_CONFIG_KWARGS, Skip, StubTokenizer, lm_head_probe, require_device,
)

GIB = 2**30
TRUE_ID, FALSE_ID = 9693, 2152
# uneven cached blocks, every one smaller than the 11.57 GiB request and
# larger than the ~1.2 GiB the fixed step needs at most
FRAGMENT_PATTERN_GIB = (5.0, 2.5, 3.75, 2.0, 4.5, 3.0, 6.0, 2.25)


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


def make_batch(rows: int, seed: int = 0) -> dict[str, list[str]]:
    body = " ".join(f"词{index}_{j}" for index in range(3) for j in range(300))  # longer than the budget
    return {
        "anchor": [f"问题{index} " + body[: 200 + index] for index in range(rows)],
        "positive": [body + f" 文{index}" for index in range(rows)],
        "negative_1": ["负" + body + f" 负{index}" for index in range(rows)],
    }


def features_of(model, data, device):
    pairs = ([[a, p] for a, p in zip(data["anchor"], data["positive"])]
             + [[a, n] for a, n in zip(data["anchor"], data["negative_1"])])
    return {key: value.to(device) for key, value in model.prepare_reranker_inputs(pairs).items()}


def largest_cached_free_block(device) -> float | None:
    """Largest free block inside the PyTorch cache, in GiB (None if unsupported)."""
    try:
        snapshot = torch.cuda.memory_snapshot()
    except Exception:
        return None
    free = [block["size"] for segment in snapshot for block in segment.get("blocks", [])
            if block.get("state") != "active"]  # "inactive" blocks are cached and free
    return round(max(free) / GIB, 3) if free else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    try:
        device = require_device()
        __import__("flagscale.models.retrieval", fromlist=["__doc__"])
        __import__("transformers", fromlist=["__doc__"])
    except Skip as skip:
        print(f"SKIP: {skip}; exit 2")
        return 2
    total_gib = torch.cuda.get_device_properties(device).total_memory / GIB
    if total_gib < 90:
        print(f"SKIP: needs a >= 90 GiB card for the 61 GiB resident state, found {total_gib:.1f} GiB; exit 2")
        return 2
    from flagscale.train.losses.retrieval import reranker_loss

    record: dict = {"device_total_gib": round(total_gib, 2), "torch": torch.__version__, "rows": args.rows}
    model = build(REAL_CONFIG["num_hidden_layers"], device)
    data = make_batch(args.rows)
    features = features_of(model, data, device)
    pairs, padded = int(features["input_ids"].shape[0]), int(features["input_ids"].shape[1])
    record.update(pairs=pairs, padded_length=padded, vocab=REAL_CONFIG["vocab_size"],
                  full_sequence_logits_gib=round(pairs * padded * REAL_CONFIG["vocab_size"] * 2 / GIB, 4))
    weights = torch.cuda.memory_allocated()
    record["weights_gib"] = round(weights / GIB, 3)

    # ---- Part 1: what does one forward make the LM head produce? ----------------
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with lm_head_probe(model) as calls:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(features)["scores"]
    torch.cuda.synchronize()
    lm_head = max(calls, key=lambda call: call["output_bytes"])
    record["part1"] = {
        "lm_head_input_shape": list(lm_head["input_shape"]),
        "lm_head_output_shape": list(lm_head["output_shape"]),
        "lm_head_output_gib": round(lm_head["output_bytes"] / GIB, 4),
        "forward_peak_allocated_gib": round((torch.cuda.max_memory_allocated() - weights) / GIB, 3),
    }
    full_sequence = lm_head["input_dim"] == 3  # fed [rows, seq, hidden]
    del scores
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ---- Part 2: resident state + fragmented remainder, then one step ------------
    stand_ins = [torch.empty_like(parameter) for parameter in model.parameters() for _ in range(3)]
    resident = torch.cuda.memory_allocated()
    record["part2"] = {
        "resident_gib": round(resident / GIB, 3),
        "resident_note": "bf16 weights + 3 stand-ins per parameter = the 61.03 GiB the real trainer "
                         "has allocated after its first optimizer step (fp32 master/grad/AdamW on 2 cards "
                         "and bf16 on 1 card have the same footprint)",
    }
    fillers: list[torch.Tensor] = []
    sizes: list[float] = []
    index = 0
    while True:
        size_gib = FRAGMENT_PATTERN_GIB[index % len(FRAGMENT_PATTERN_GIB)]
        index += 1
        try:
            fillers.append(torch.empty(int(size_gib * GIB), dtype=torch.uint8, device=device))
            sizes.append(size_gib)
        except torch.OutOfMemoryError:
            break
    device_free_after_fill = torch.cuda.mem_get_info()[0] / GIB
    del fillers  # stay in the allocator cache as uneven free blocks
    torch.cuda.synchronize()
    free, _ = torch.cuda.mem_get_info()
    record["part2"].update({
        "fragment_sizes_gib": sizes,
        "fragments_total_gib": round(sum(sizes), 3),
        "largest_cached_free_block_gib": largest_cached_free_block(device),
        "device_free_gib_before_step": round(free / GIB, 3),
        "cached_free_gib_before_step": round((torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / GIB, 3),
    })

    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    step: dict = {"completed": False}
    try:
        with lm_head_probe(model) as calls:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = reranker_loss(model, data, device)
            loss.backward()
        torch.cuda.synchronize()
        step.update(
            completed=True, loss=round(float(loss), 6),
            peak_allocated_gib=round((torch.cuda.max_memory_allocated() - resident) / GIB, 3),
            peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GIB, 3),
            lm_head_output_shapes=[list(call["output_shape"]) for call in calls],
        )
    except torch.OutOfMemoryError as error:
        torch.cuda.synchronize()
        step.update(completed=False, oom_message=str(error)[:900])
    model.zero_grad(set_to_none=True)
    free_after, _ = torch.cuda.mem_get_info()
    step["device_free_gib_after_step"] = round(free_after / GIB, 3)
    record["part2"]["step"] = step

    passed = step["completed"] and not full_sequence
    record["passed"] = passed
    if passed:
        record["verdict"] = "the step runs on the fragmented heap; the LM head projects only the scored rows"
    elif not step["completed"] and full_sequence:
        record["verdict"] = "OutOfMemoryError on the fragmented heap at the full-sequence logits request"
    elif not step["completed"]:
        record["verdict"] = ("OutOfMemoryError on the fragmented heap although the LM head projects only the "
                             "scored rows -- the heap, not the logits; see oom_message")
    else:
        record["verdict"] = "the step ran, but the LM head still projects every position"
    print("RESULT_JSON " + json.dumps(record, ensure_ascii=False, default=str))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"part1: lm_head output {record['part1']['lm_head_output_shape']} = "
          f"{record['part1']['lm_head_output_gib']:.4f} GiB (full-sequence logits would be "
          f"{record['full_sequence_logits_gib']:.4f} GiB)")
    print(f"part2: resident {record['part2']['resident_gib']:.2f} GiB, {len(sizes)} cached fragments "
          f"total {record['part2']['fragments_total_gib']:.2f} GiB, largest {record['part2']['largest_cached_free_block_gib']} GiB; "
          f"step {'completed' if step['completed'] else 'OutOfMemoryError'}"
          + (f", peak +{step['peak_allocated_gib']:.2f} GiB" if step["completed"] else ""))
    if not passed:
        if not step["completed"]:
            print("OOM message: " + step["oom_message"])
        print("REPRO: the reranker step asks for an 11.57 GiB contiguous block; exit 1")
        return EXIT_FAIL
    print("REPRO resolved: no full-sequence logits request; exit 0")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
