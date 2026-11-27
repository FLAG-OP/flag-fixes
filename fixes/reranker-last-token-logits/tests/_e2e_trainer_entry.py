"""E2E entry: run the real FlagScale retrieval trainer and record device memory.

Launched by ``torchrun`` from ``e2e_reranker_train.py``.  The trainer itself is
not modified: this wrapper only starts a sampling thread, prints torch's own
peak counters after every optimizer step, and writes one JSON record per rank.

``RERANKER_E2E_NO_SAVE=1`` replaces ``save_model`` with a no-op so a run that
only needs the training loop does not spend minutes writing a 32 GB checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torchrun_job import announce_worker  # noqa: E402

RANK = int(os.environ.get("RANK", "0"))


def gib(value: int) -> float:
    return round(value / 2**30, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--mem-out", required=True)
    args, _ = parser.parse_known_args()
    announce_worker("e2e-trainer")  # names this worker's pid and job in the launcher's log
    sys.argv = [sys.argv[0], "--config-file", args.config_file]

    peak = {"allocated": 0, "reserved": 0}
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            try:
                peak["allocated"] = max(peak["allocated"], torch.cuda.memory_allocated())
                peak["reserved"] = max(peak["reserved"], torch.cuda.memory_reserved())
            except Exception:  # pragma: no cover - device lost
                pass
            stop.wait(0.05)

    threading.Thread(target=sample, daemon=True).start()

    from flagscale.train import train_retrieval

    if os.environ.get("RERANKER_E2E_NO_SAVE") == "1":
        train_retrieval.save_model = lambda *a, **k: None

    steps: list[dict] = []
    original_step = torch.optim.AdamW.step

    def step_with_memory(self, *a, **k):
        out = original_step(self, *a, **k)
        torch.cuda.synchronize()
        steps.append(
            {
                "step": len(steps) + 1,
                "allocated_gib": gib(torch.cuda.memory_allocated()),
                "reserved_gib": gib(torch.cuda.memory_reserved()),
                "step_peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
                "step_peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
            }
        )
        if RANK == 0:
            last = steps[-1]
            print(
                f"[e2e-mem] after optimizer step {last['step']}: "
                f"allocated={last['allocated_gib']:.2f} GiB reserved={last['reserved_gib']:.2f} GiB "
                f"step_peak_allocated={last['step_peak_allocated_gib']:.2f} GiB "
                f"step_peak_reserved={last['step_peak_reserved_gib']:.2f} GiB",
                flush=True,
            )
        torch.cuda.reset_peak_memory_stats()
        return out

    torch.optim.AdamW.step = step_with_memory

    record: dict = {"rank": RANK, "error": None, "oom": False}
    code = 0
    try:
        train_retrieval.main()
    except BaseException as error:  # record, then surface through the exit code
        code = 1
        record["error"] = f"{type(error).__name__}: {str(error)[:1500]}"
        record["oom"] = isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
        traceback.print_exc()
    finally:
        stop.set()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        record.update(
            steps=steps,
            peak_allocated_gib=gib(peak["allocated"]),
            peak_reserved_gib=gib(peak["reserved"]),
            max_memory_allocated_gib=gib(torch.cuda.max_memory_allocated()),
            max_memory_reserved_gib=gib(torch.cuda.max_memory_reserved()),
        )
        with open(f"{args.mem_out}.rank{RANK}", "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
