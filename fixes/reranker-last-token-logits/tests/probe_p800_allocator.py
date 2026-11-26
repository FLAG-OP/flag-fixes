#!/usr/bin/env python3
"""Platform probe: no expandable segments on P800, and a fragmented heap refuses a large request.

Diagnostic only -- not a verification command; it records allocator behaviour,
it does not judge the fix.  Needs one idle card (uses nearly all of it).

  A  fresh 11.57 GiB request on an empty card (control: must succeed)
  B  82 x 1 GiB, free every other block, release them to the device -> 11.57 GiB
  C  same, but the holes stay in the PyTorch cache -> 11.57 GiB
  D  uneven blocks (64 MiB - 1 GiB), release 2/3 to the device -> 11.57 GiB,
     then a 1.25 GiB request (the size class the fixed step needs at most)
  E  pattern D again in child processes with expandable_segments requested through
     both spellings of the allocator env variable.  Whether the setting is
     honoured is judged by the outcome of the request, not by the absence of a
     warning.

Exit 0  probe completed (read the JSON)
Exit 2  no accelerator / less than 90 GiB free on the card
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

GIB, MIB = 1 << 30, 1 << 20
REQUEST = 11 * GIB + 614 * MIB  # 11.57 GiB: 64 pairs x 640 tokens x 151,669 vocab x bf16
UNEVEN = []
for index in range(200):
    UNEVEN += [(64 + (index % 5) * 128) * MIB, 256 * MIB, (512 + (index % 3) * 256) * MIB]


def state(device) -> dict:
    free, total = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    return {
        "device_free_gib": round(free / GIB, 2),
        "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 2),
        "reserved_gib": round(torch.cuda.memory_reserved() / GIB, 2),
        "segments": stats.get("segment.all.current"),
        "alloc_retries": stats.get("num_alloc_retries"),
    }


def attempt(nbytes: int, device) -> dict:
    try:
        block = torch.empty(nbytes, dtype=torch.uint8, device=device)
        del block
        return {"request_gib": round(nbytes / GIB, 2), "ok": True}
    except torch.OutOfMemoryError as error:
        return {"request_gib": round(nbytes / GIB, 2), "ok": False, "message": str(error)[:700]}


def pattern_uneven(device) -> dict:
    """Pattern D; also run standalone in child processes for pattern E."""
    blocks = []
    for size in UNEVEN:
        try:
            blocks.append(torch.empty(size, dtype=torch.uint8, device=device))
        except torch.OutOfMemoryError:
            break
    kept = len(blocks)
    for index, block in enumerate(blocks):
        if index % 3 != 0:
            blocks[index] = None
    torch.cuda.empty_cache()
    result = {
        "layout": f"{kept} blocks of 64 MiB - 1 GiB, 2/3 released", "before": state(device),
        "result_11_57_gib": attempt(REQUEST, device), "result_1_25_gib": attempt(int(1.25 * GIB), device),
    }
    del blocks
    torch.cuda.empty_cache()
    return result


def free_device() -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda", int(os.environ.get("RERANKER_TEST_DEVICE_INDEX", "0")))
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    if total < 90 * GIB or torch.cuda.mem_get_info()[0] < 90 * GIB:
        return None
    return device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=None)
    parser.add_argument("--child-d", action="store_true", help="internal: run pattern D and print one JSON line")
    args = parser.parse_args()
    device = free_device()
    if device is None:
        print("SKIP: needs an idle accelerator with >= 90 GiB free; exit 2")
        return 2

    if args.child_d:
        print("PROBE_JSON " + json.dumps(pattern_uneven(device)))
        return 0

    record: dict = {"torch": torch.__version__, "device": torch.cuda.get_device_name(device), "patterns": {}}
    record["patterns"]["A_fresh"] = {"before": state(device), "result": attempt(REQUEST, device)}
    torch.cuda.empty_cache()

    blocks = []
    for _ in range(82):
        try:
            blocks.append(torch.empty(GIB, dtype=torch.uint8, device=device))
        except torch.OutOfMemoryError:
            break
    for index in range(0, len(blocks), 2):
        blocks[index] = None
    torch.cuda.empty_cache()
    record["patterns"]["B_uniform_holes_released"] = {
        "layout": f"{len(blocks)} x 1 GiB, every other released", "before": state(device),
        "result": attempt(REQUEST, device)}
    del blocks
    torch.cuda.empty_cache()

    blocks = []
    for _ in range(82):
        try:
            blocks.append(torch.empty(GIB, dtype=torch.uint8, device=device))
        except torch.OutOfMemoryError:
            break
    for index in range(0, len(blocks), 2):
        blocks[index] = None
    record["patterns"]["C_uniform_holes_cached"] = {
        "layout": f"{len(blocks)} x 1 GiB, every other freed into the cache", "before": state(device),
        "result": attempt(REQUEST, device)}
    del blocks
    torch.cuda.empty_cache()

    record["patterns"]["D_uneven_holes_released"] = pattern_uneven(device)

    # E: is expandable_segments honoured under either spelling of the env variable?
    expandable = {}
    for variable in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child-d"],
                               env={**os.environ, variable: "expandable_segments:True"},
                               capture_output=True, text=True, timeout=1200)
        line = next((item for item in child.stdout.splitlines() if item.startswith("PROBE_JSON ")), None)
        outcome = json.loads(line[len("PROBE_JSON "):]) if line else None
        warnings = sorted({item.strip()[:300] for item in child.stderr.splitlines()
                           if "expandable_segments" in item})
        expandable[variable] = {
            "returncode": child.returncode,
            "device_free_gib": outcome["before"]["device_free_gib"] if outcome else None,
            "request_11_57_gib_ok": outcome["result_11_57_gib"]["ok"] if outcome else None,
            "honoured": bool(outcome and outcome["result_11_57_gib"]["ok"]),
            "warnings": warnings,
            "note": "same default-allocator OOM as pattern D" if outcome and not outcome["result_11_57_gib"]["ok"]
            else ("" if outcome else "child produced no record"),
        }
    record["patterns"]["E_expandable_segments_requested"] = expandable
    print("RESULT_JSON " + json.dumps(record, ensure_ascii=False))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
