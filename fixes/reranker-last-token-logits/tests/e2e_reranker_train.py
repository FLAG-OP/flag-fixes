#!/usr/bin/env python3
"""E2E: the real FlagScale retrieval trainer on Qwen3-Reranker-8B, one run.

This is the integration path the problem was reported from: ``train.parallel``
``fsdp2`` with fp32 master weights, bf16 autocast, gradient checkpointing,
``max_length`` 640 and ``num_negatives`` 1 -- the recipe of
``examples/retrieval/run-p800-fsdp2-2card.sh`` and ``conf/train/qwen3_reranker*.yaml``.

The model and the T2Ranking ``triplet-15`` parquet shards are not bundled:

    --model / $RERANKER_MODEL   checkpoint directory (config + weights + tokenizer)
    --data  / $RERANKER_DATA    directory or single ``*.parquet`` with
                                anchor / positive / negative_1 columns

Exit codes
    0  trainer completed at least ``--min-steps`` optimizer steps
    1  trainer failed (non-zero return code, OOM in the log, too few steps)
    2  precondition missing (model / data / device) -- SKIP, not a pass
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _torchrun_job import limit_visible_devices, run_job  # noqa: E402

GIB = 2**30
OOM_PATTERN = re.compile(
    r"out of memory|Tried to allocate|torch\.OutOfMemoryError|OutOfMemoryError", re.I
)
ALLOCATE_PATTERN = re.compile(r"Tried to allocate ([0-9.]+) (\w+)")

TRAIN_DEFAULTS = {
    # conf/train/qwen3_reranker.yaml
    "task": "reranker",
    "seed": 42,
    "epochs": 1,
    "log_every": 1,
    "save_every_epoch": False,
    "save_steps": 0,
    "gradient_accumulation_steps": 1,
}


def gib(text: str) -> float:
    match = ALLOCATE_PATTERN.search(text)
    if not match:
        return float("nan")
    value, unit = float(match.group(1)), match.group(2).lower()
    return value * {"gib": 1.0, "mib": 1 / 1024, "kib": 1 / 1024**2, "bib": 1 / 1024**3}[unit]


def build_config(args) -> dict:
    data = {
        "path": args.data,
        "validation_path": args.data,
        "test_path": args.data,
        "train_split": "train",
        "validation_split": "validation",
        "test_split": "test",
        # one evaluation batch per split keeps the run short; the loaders share
        # the training batch size, so its shape is the training shape
        "eval_max_samples": args.batch,
        "split_modulo": 20,
        "shuffle_buffer": 20000,
        "length_sort_window": 256,
        "num_negatives": 1,
        "max_samples": args.batch * args.min_steps,
    }
    train = dict(TRAIN_DEFAULTS)
    train.update(
        {
            "name": "qwen3-reranker-8b",
            "batch_size": args.batch,
            "output_dir": str(args.out),
            "parallel": args.parallel,
            "master_dtype": args.master_dtype,
            "model": {
                "model_type": "qwen3_reranker",
                "model_path": args.model,
                "max_length": 640,
                "torch_dtype": "bf16",
                "freeze_backbone": False,
                "use_score_head": False,
                "gradient_checkpointing": True,
            },
            "data": data,
            "optimizer": {
                "name": "adamw",
                "lr": 1.0e-5,
                "lr_decay_style": "cosine",
                "min_lr": 1.0e-6,
                "warmup_steps": 200,
                "weight_decay": 0.01,
                "max_grad_norm": args.max_grad_norm,
            },
            "hardware": {
                "device_type": "cuda",
                "distributed_backend": "auto",
                "autocast_device": "cuda",
                "use_flaggems": False,
            },
            "deterministic": False,
        }
    )
    return {"train": train}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("RERANKER_MODEL"))
    parser.add_argument("--data", default=os.environ.get("RERANKER_DATA"))
    parser.add_argument("--nproc", type=int, default=int(os.environ.get("RERANKER_E2E_NPROC", "2")))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--parallel", choices=["fsdp2", "ddp"], default="fsdp2")
    parser.add_argument("--master-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--no-save", action="store_true", help="skip the final checkpoint write")
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="optimizer.max_grad_norm; 0 disables clipping (the vendor clip_grad_norm_ "
        "materialises ~35 GiB of fp32 temporaries for bf16 gradients)",
    )
    args = parser.parse_args()

    if not args.model or not Path(args.model, "config.json").is_file():
        print("SKIP: no checkpoint directory (--model / $RERANKER_MODEL); exit 2")
        return 2
    if not args.data or not (
        Path(args.data).is_file() or list(Path(args.data).glob("*.parquet"))
    ):
        print("SKIP: no triplet parquet (--data / $RERANKER_DATA); exit 2")
        return 2
    try:
        import torch  # noqa: F401
    except Exception as error:
        print(f"SKIP: torch not importable ({type(error).__name__}); exit 2")
        return 2
    probe = subprocess.run(
        [sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"],
        capture_output=True, text=True, timeout=600,
    )
    if probe.returncode != 0:
        print("SKIP: no CUDA device for the trainer; exit 2")
        return 2

    tag = f"{args.parallel}-{args.master_dtype}-b{args.batch}-n{args.nproc}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(args.out or Path(os.environ.get("RERANKER_E2E_OUT", "/tmp/opencode/reranker-e2e")) / tag)
    out.mkdir(parents=True, exist_ok=True)
    args.out = out
    config = build_config(args)
    config_path = out / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    env = dict(os.environ)
    # an explicit CUDA_VISIBLE_DEVICES is cut to the rank count and mirrored to XPU; with the
    # default visibility or an XPU-only list nothing is touched (see limit_visible_devices)
    visible = limit_visible_devices(env, args.nproc)
    env.update(
        {
            "FLAGSCALE_DEVICE": "cuda",
            "FS_PLATFORM": "cuda",
            "FLAGSCALE_AUTOCAST_DEVICE": "cuda",
            "FLAGOS_USE_FLAGGEMS": "false",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            # The P800 launch scripts set this: XBLAS otherwise picks a bmm
            # kernel that fails to JIT for the reranker's sequence lengths.
            "XMLIR_BMM_DISPATCH_VALUE": "0",
            "RERANKER_E2E_NO_SAVE": "1" if args.no_save else "0",
        }
    )
    mem_out = out / "memory.json"
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={args.nproc}", str(HERE / "_e2e_trainer_entry.py"),
        "--config-file", str(config_path), "--mem-out", str(mem_out),
    ]
    # own session per job; a timeout ends the agent and the processes carrying this job's id
    job = run_job(command, env, out / "train.log", args.timeout, cwd=str(HERE))
    returncode, timed_out, elapsed = job["returncode"], job["timed_out"], job["elapsed_s"]
    if timed_out:
        cleanup = job["cleanup"]
        ended = [item["pid"] for item in cleanup["signalled"]]
        print(f"launcher timeout after {args.timeout} s: torchrun agent ended, job {job['job_id']} "
              f"processes signalled {ended}, survived {cleanup['survived']}")

    text = (out / "train.log").read_text(errors="replace")
    oom_lines = [line.strip() for line in text.splitlines() if OOM_PATTERN.search(line)]
    requests = sorted({round(gib(line), 3) for line in oom_lines if gib(line) == gib(line)})
    steps_done = max(
        [int(m) for m in re.findall(r"optimizer_step=(\d+)", text)], default=0
    )
    # the trainer's own loss lines, kept with the memory record
    loss_log = [line.strip() for line in text.splitlines()
                if re.search(r"optimizer_step=\d+ running_loss=|train_loss=|test_loss=", line)]
    memories = []
    for path in sorted(out.glob("memory.json.rank*")):
        memories.append(json.loads(path.read_text()))
    peaks = [m.get("max_memory_allocated_gib") for m in memories]
    passed = returncode == 0 and steps_done >= args.min_steps and not oom_lines
    summary = {
        "passed": passed,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_s": round(elapsed, 1),
        "batch": args.batch,
        "parallel": args.parallel,
        "master_dtype": args.master_dtype,
        "nproc": args.nproc,
        "min_steps": args.min_steps,
        "steps_done": steps_done,
        # the training loop reached --min-steps without OOM; distinct from
        # `passed`, which also requires the trainer process to exit cleanly
        "training_steps_complete": steps_done >= args.min_steps and not oom_lines,
        "loss_log": loss_log,
        "oom": bool(oom_lines),
        "oom_allocate_requests_gib": requests,
        "first_oom_line": oom_lines[0][:400] if oom_lines else None,
        "max_memory_allocated_gib": max((p for p in peaks if p is not None), default=None),
        "per_rank": memories,
        "log": str(out / "train.log"),
        "command": " ".join(command),
        # which processes belonged to this run: the id every process of the job carried, what
        # the timeout cleanup signalled (None unless the timeout fired), and how the
        # visible-device variables were handled
        "job_id": job["job_id"],
        "timeout_cleanup": job["cleanup"],
        "visible_devices": visible,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not passed:
        print(f"E2E FAIL: steps {steps_done}/{args.min_steps}, returncode {returncode}, oom={bool(oom_lines)}")
        return 1
    print(f"E2E OK: {steps_done} optimizer steps at batch {args.batch} ({args.parallel}, {args.master_dtype}); "
          f"peak allocated {summary['max_memory_allocated_gib']} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
