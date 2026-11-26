#!/usr/bin/env python3
"""FSDP2 dispatch: the adapter's forward under FlagScale's real FSDP2 wiring.

``train.parallel=fsdp2`` shards ``backbone.lm_head`` as its own FSDP module
("oversized leaf", see ``parallel.iter_large_leaf_modules``).  The fix calls
``self.backbone.lm_head`` directly instead of letting ``Qwen3ForCausalLM``
call it, so this test checks on the real ``apply_fsdp2`` path that

* lm_head keeps its FSDP pre/post-forward hooks and Shard(0) placement,
* every forward routes exactly one ``[rows, hidden]`` tensor through it
  (baseline: ``[rows, seq, hidden]`` -- the full-sequence logits),
* gradients reach lm_head, the embedding, the norm and the blocks and the
  optimizer updates the lm_head shard,
* scores equal the full-logit reference on the same sharded weights.

Default: ``--device cpu`` (gloo, single rank, small hidden size) so the check
runs wherever torch has composable FSDP.  ``--device cuda --nproc 2`` repeats
it on the accelerator with the trainer's bf16 autocast.

Exit 0  all checks pass on every rank
Exit 1  a check failed
Exit 2  precondition missing, or the platform blocked the run (timeout /
        collective watchdog) -- SKIP, not a pass
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLOCKER = re.compile(r"Watchdog caught collective operation timeout|kl3ChannelCheckErrors|kernel exception", re.I)


def write_summary(path, summary: dict) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--nproc", type=int, default=None, help="default 1 for cpu, 2 for cuda")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--layers", type=int, default=None, help="default: 2 on cpu, 4 on cuda")
    parser.add_argument("--hidden", type=int, default=None, help="default: 2048 on cpu, 4096 on cuda")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--json", default=None)
    args, extra = parser.parse_known_args()

    try:
        import torch  # noqa: F401
        from torch.distributed._composable.fsdp import fully_shard  # noqa: F401
    except Exception as error:
        print(f"SKIP: torch / composable FSDP (fully_shard) unavailable ({type(error).__name__}: {error}); exit 2")
        return 2

    nproc = args.nproc or (1 if args.device == "cpu" else 2)
    layers = args.layers or (2 if args.device == "cpu" else 4)
    hidden = args.hidden or (2048 if args.device == "cpu" else 4096)
    if args.device == "cuda":
        probe = subprocess.run(
            [sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"],
            capture_output=True, text=True, timeout=600,
        )
        if probe.returncode != 0:
            print("SKIP: no CUDA device; exit 2")
            return 2

    given = Path(args.json or Path(os.environ.get("RERANKER_TEST_OUT", "/tmp/opencode/reranker-tests")) / "fsdp2-step.json")
    # the worker runs with cwd=tests/, so hand it an absolute record path
    out = given.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    def display(path) -> str:
        relative = os.path.relpath(path)
        return relative if not relative.startswith("..") else str(path)
    for stale in out.parent.glob(out.name + ".rank*"):
        stale.unlink()
    log_path = out.with_suffix(".log")

    env = dict(os.environ)
    if args.device == "cuda":
        visible = env.get("CUDA_VISIBLE_DEVICES", "")
        if "," in visible and nproc > 1:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(visible.split(",")[:nproc])
        env.setdefault("XPU_VISIBLE_DEVICES", env["CUDA_VISIBLE_DEVICES"])
        env.setdefault("XMLIR_BMM_DISPATCH_VALUE", "0")
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
        str(HERE / "_fsdp2_worker.py"), "--device", args.device, "--steps", str(args.steps),
        "--rows", str(args.rows), "--json", str(out),
    ]
    if args.backend:
        command += ["--backend", args.backend]
    command += ["--layers", str(layers), "--hidden", str(hidden)]
    command += extra

    started = time.time()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(HERE),
                                   start_new_session=True)
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
            # torchrun starts each worker in its own session: a stuck worker
            # survives the group kill, so remove the workers by their entry point
            subprocess.run(["pkill", "-KILL", "-f", str(HERE / "_fsdp2_worker.py")], capture_output=True)
    elapsed = time.time() - started

    text = log_path.read_text(errors="replace")
    records = [json.loads(p.read_text()) for p in sorted(out.parent.glob(out.name + ".rank*"))]
    summary = {
        "device": args.device, "nproc": nproc, "layers": layers, "hidden": hidden,
        "returncode": returncode, "timed_out": timed_out,
        "elapsed_s": round(elapsed, 1),
        "ranks": records,
    }
    summary["log"] = display(log_path)
    summary["command"] = " ".join(display(part) for part in command)
    blocked = timed_out or bool(BLOCKER.search(text))
    if blocked:
        summary.update(passed=False, blocked=True)
        write_summary(args.json, summary)
        print("RESULT_JSON " + json.dumps(summary, ensure_ascii=False, default=str))
        print("BLOCKED: the platform did not complete the FSDP2 run (collective timeout / device error); "
              "exit 2 -- not a pass. See the log for the watchdog trace.")
        return 2
    if not records:
        summary["passed"] = False
        write_summary(args.json, summary)
        print("RESULT_JSON " + json.dumps(summary, ensure_ascii=False, default=str))
        print("FSDP2 test produced no per-rank record (worker crashed); see the log. exit 1")
        return 1
    summary["passed"] = returncode == 0 and len(records) == nproc and all(r["passed"] for r in records)
    write_summary(args.json, summary)
    print("RESULT_JSON " + json.dumps(summary, ensure_ascii=False, default=str))
    for record in records:
        for name, check in record["checks"].items():
            if not check["pass"]:
                print(f"FAIL rank{record['rank']} {name}: {json.dumps(check['detail'], default=str)[:600]}")
    if not summary["passed"]:
        print("FSDP2 dispatch test FAILED; exit 1")
        return 1
    print(f"FSDP2 dispatch OK on {nproc} rank(s) ({args.device}): lm_head is a sharded FSDP module and receives "
          f"only the scored rows; exit 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
