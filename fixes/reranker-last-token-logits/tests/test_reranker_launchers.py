#!/usr/bin/env python3
"""Launcher self-test: a timed-out run removes only its own workers; visible-device mapping.

The two launchers (``e2e_reranker_train.py``, ``test_reranker_fsdp2_step.py``) share
``_torchrun_job.py``.  This script exercises the shared pieces on their own, on the CPU.

1. Timeout cleanup scope.  Two torchrun jobs started from one worker script path -- the
   situation of a baseline and a patched run going in parallel from one checkout:

       hang    ``--nproc`` ranks that record their pid, start one child each and never
               return; the job hits the launcher timeout (``--hang-timeout``)
       normal  ``--nproc`` ranks that record their pid, sleep ``--hang-timeout`` +
               ``--margin`` seconds and exit 0

   Passing means: the hang ranks and their children are all gone when the hang launcher
   returns, no normal rank was signalled, and the normal job exits 0 with every rank having
   started before and finished after the hang job's timeout (so the surviving ranks were
   really alive when the other job's cleanup ran).

   ``--cleanup legacy`` starts the same two jobs with the cleanup the launchers had before
   (``pkill -KILL -f <worker script>``).  It is expected to FAIL: the normal job's ranks are
   killed by the hang job's cleanup.  It exists so the record shows that this self-test tells
   the two behaviours apart.  It signals by script path, so run it only when nothing else
   from this checkout is running.  Both modes end with a sweep limited to the two job ids.

2. Visible devices.  ``limit_visible_devices`` with the default visibility, an explicit CUDA
   list (kept, cut to the rank count, XPU already set) and an XPU-only list.

Exit 0  every check passes
Exit 1  a check failed (with ``--cleanup legacy`` this is the expected outcome)
Exit 2  precondition missing (torch.distributed.run not importable) -- SKIP, not a pass
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _torchrun_job import (  # noqa: E402
    JOB_ID_ENV, job_processes, limit_visible_devices, new_job_id, run_job, terminate_job,
)

EXIT_PASS, EXIT_FAIL, EXIT_SKIP = 0, 1, 2
WORKER = HERE / "_launcher_worker.py"
VISIBLE_CASES = [
    # name, environment given to the launcher, ranks, expected environment afterwards
    ("default_visibility_two_ranks", {}, 2, {}),
    ("default_visibility_single_rank", {}, 1, {}),
    ("explicit_two_cards_two_ranks", {"CUDA_VISIBLE_DEVICES": "1,2"}, 2,
     {"CUDA_VISIBLE_DEVICES": "1,2", "XPU_VISIBLE_DEVICES": "1,2"}),
    ("explicit_three_cards_two_ranks_cut_to_two", {"CUDA_VISIBLE_DEVICES": "1,2,3"}, 2,
     {"CUDA_VISIBLE_DEVICES": "1,2", "XPU_VISIBLE_DEVICES": "1,2"}),
    ("explicit_two_cards_single_rank_kept", {"CUDA_VISIBLE_DEVICES": "1,2"}, 1,
     {"CUDA_VISIBLE_DEVICES": "1,2", "XPU_VISIBLE_DEVICES": "1,2"}),
    ("explicit_list_with_xpu_already_set", {"CUDA_VISIBLE_DEVICES": "1,2", "XPU_VISIBLE_DEVICES": "5"}, 2,
     {"CUDA_VISIBLE_DEVICES": "1,2", "XPU_VISIBLE_DEVICES": "5"}),
    ("xpu_list_only", {"XPU_VISIBLE_DEVICES": "1,2"}, 2, {"XPU_VISIBLE_DEVICES": "1,2"}),
]


def read_stamps(directory: Path, kind: str, nproc: int) -> list[dict | None]:
    """Per-rank ``start`` / ``done`` records; ``None`` for a rank that did not write one."""
    rows = []
    for rank in range(nproc):
        path = directory / f"rank{rank}.{kind}"
        rows.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None)
    return rows


def wait_until_gone(job_id: str, pids: set[int], limit: float = 10.0) -> list[dict]:
    """Wait up to ``limit`` seconds for ``pids`` to leave the job, then return what is left."""
    deadline = time.monotonic() + limit
    while True:
        left = job_processes(job_id)
        if not (pids & {item["pid"] for item in left}) or time.monotonic() >= deadline:
            return left
        time.sleep(0.2)


def run_legacy_job(command: list[str], env: dict, log_path: Path, timeout: float, job_id: str) -> dict:
    """The launch path as it was before the fix, cleanup by entry-point name included.

    The two lines after the group kill are the cleanup the launchers had (E2E and FSDP2
    entry alike): remove the workers by the path of their entry script.
    """
    env = dict(env)
    env[JOB_ID_ENV] = job_id
    result = {"job_id": job_id, "started_at": time.time(), "timed_out": False, "cleanup": None}
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(HERE),
                                   start_new_session=True)
        try:
            result["returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            os.killpg(process.pid, signal.SIGKILL)
            result["returncode"] = process.wait()
            subprocess.run(["pkill", "-KILL", "-f", str(WORKER)], capture_output=True)
            result["cleanup"] = {"mode": "legacy", "command": f"pkill -KILL -f {WORKER}",
                                 "signalled": None}
    result["elapsed_s"] = round(time.time() - result["started_at"], 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cleanup", choices=["scoped", "legacy"], default="scoped")
    parser.add_argument("--nproc", type=int, default=2, help="ranks per job (default 2)")
    parser.add_argument("--hang-timeout", type=float, default=60.0,
                        help="launcher timeout of the hang job; must exceed the torchrun start-up")
    parser.add_argument("--margin", type=float, default=30.0,
                        help="the normal ranks sleep this long past the hang job's timeout")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if importlib.util.find_spec("torch.distributed.run") is None:
        print("SKIP: torch.distributed.run not importable; exit 2")
        return EXIT_SKIP
    if args.cleanup == "legacy" and shutil.which("pkill") is None:
        print("SKIP: --cleanup legacy needs pkill; exit 2")
        return EXIT_SKIP

    given = Path(args.json) if args.json else None
    if given is None:
        given = Path(os.environ.get("RERANKER_TEST_OUT") or tempfile.mkdtemp(prefix="reranker-launchers-")) \
            / "launcher-selftest.json"
    out = given.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("") if out.suffix == ".json" else out  # side files: <stem>.hang.log ...
    stamps = {key: out.parent / f"{stem.name}.{key}" for key in ("hang", "normal")}
    logs = {key: out.parent / f"{stem.name}.{key}.log" for key in ("hang", "normal")}
    for directory in stamps.values():
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)

    results: list[dict] = []

    def record(case: str, ok: bool, **detail) -> bool:
        results.append({"case": case, "pass": bool(ok), **detail})
        return bool(ok)

    # ---- 2. visible devices: pure function, no process started
    for name, given_env, nproc, expected in VISIBLE_CASES:
        env = dict(given_env)
        mapping = limit_visible_devices(env, nproc)
        record(f"visible_devices:{name}", env == expected, given=given_env, expected=expected, actual=env,
               action=mapping["action"])

    # ---- 1. two jobs from one worker script path
    base_env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "XPU_VISIBLE_DEVICES", JOB_ID_ENV):
        base_env.pop(key, None)
    base_env["CUDA_VISIBLE_DEVICES"] = ""  # CPU jobs: no device is touched either way
    sleep_seconds = args.hang_timeout + args.margin
    job_ids = {"hang": f"selftest-hang-{new_job_id()}", "normal": f"selftest-normal-{new_job_id()}"}
    commands = {
        "hang": [sys.executable, "-m", "torch.distributed.run", "--standalone",
                 f"--nproc_per_node={args.nproc}", str(WORKER), "--mode", "hang",
                 "--stamps", str(stamps["hang"])],
        "normal": [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={args.nproc}", str(WORKER), "--mode", "sleep",
                   "--seconds", str(sleep_seconds), "--stamps", str(stamps["normal"])],
    }
    timeouts = {"hang": args.hang_timeout, "normal": sleep_seconds + 600}
    jobs: dict[str, dict] = {}

    def launch(key: str) -> None:
        try:
            if args.cleanup == "scoped":
                jobs[key] = run_job(commands[key], base_env, logs[key], timeouts[key], cwd=str(HERE),
                                    job_id=job_ids[key])
                jobs[key]["processes_after_return"] = job_processes(job_ids[key])
            else:
                jobs[key] = run_legacy_job(commands[key], base_env, logs[key], timeouts[key], job_ids[key])
                stamps_now = [row for row in read_stamps(stamps[key], "start", args.nproc) if row]
                pids = {row["pid"] for row in stamps_now}
                # the legacy cleanup signals and returns; give the signalled ranks a moment to exit
                jobs[key]["processes_after_return"] = wait_until_gone(job_ids[key], pids)
        except Exception as error:  # surfaced as a failed check below
            jobs[key] = {"job_id": job_ids[key], "error": f"{type(error).__name__}: {error}"}

    hang_thread = threading.Thread(target=launch, args=("hang",))
    hang_thread.start()
    launch("normal")
    hang_thread.join()
    hang, normal = jobs["hang"], jobs["normal"]
    timeout_at = hang.get("started_at", time.time()) + args.hang_timeout

    hang_starts = read_stamps(stamps["hang"], "start", args.nproc)
    normal_starts = read_stamps(stamps["normal"], "start", args.nproc)
    normal_dones = read_stamps(stamps["normal"], "done", args.nproc)
    hang_pids = {row["pid"] for row in hang_starts if row}
    child_pids = {row["child_pid"] for row in hang_starts if row and row.get("child_pid")}
    normal_pids = {row["pid"] for row in normal_starts if row}

    record("hang_job:timed_out", hang.get("timed_out") is True,
           error=hang.get("error"), returncode=hang.get("returncode"))
    record("hang_job:ranks_started_before_timeout",
           all(row is not None and row["time"] < timeout_at for row in hang_starts),
           timeout_at=timeout_at, start_time=[row["time"] if row else None for row in hang_starts],
           note="if false, torchrun start-up exceeded --hang-timeout and nothing was running at the timeout")
    record("hang_job:ranks_carry_job_id",
           all(row is not None and row.get("job_id") == job_ids["hang"] for row in hang_starts),
           job_id=job_ids["hang"])
    left_after_return = hang.get("processes_after_return")
    record("hang_job:fully_gone_when_launcher_returns", left_after_return == [],
           processes_after_return=left_after_return, cleanup=hang.get("cleanup"))

    if args.cleanup == "scoped":
        signalled = {item["pid"] for item in (hang.get("cleanup") or {}).get("signalled", [])}
        record("hang_job:cleanup_covered_ranks_and_children", (hang_pids | child_pids) <= signalled,
               ranks=sorted(hang_pids), children=sorted(child_pids), signalled=sorted(signalled))
        record("normal_job:not_signalled", not (normal_pids & signalled),
               normal_ranks=sorted(normal_pids), signalled=sorted(signalled))
        record("normal_job:no_timeout_cleanup", normal.get("cleanup") is None and normal.get("timed_out") is False,
               error=normal.get("error"))

    record("normal_job:ranks_started_before_hang_timeout",
           all(row is not None and row["time"] < timeout_at for row in normal_starts),
           timeout_at=timeout_at, start_time=[row["time"] if row else None for row in normal_starts])
    record("normal_job:ranks_alive_at_hang_timeout",
           all(row is not None and row["time"] > timeout_at for row in normal_dones),
           done_time=[row["time"] if row else None for row in normal_dones],
           note="a rank that wrote its done record after the timeout was alive at the timeout")
    record("normal_job:exits_zero",
           normal.get("returncode") == 0 and all(row is not None for row in normal_dones),
           returncode=normal.get("returncode"), timed_out=normal.get("timed_out"), error=normal.get("error"))

    sweep = {key: terminate_job(job_ids[key], grace=10.0) for key in job_ids}

    by_case = {item["case"]: item["pass"] for item in results}
    legacy_reproduced = (
        args.cleanup == "legacy"
        and not by_case.get("normal_job:exits_zero", True)
        and by_case.get("normal_job:ranks_started_before_hang_timeout", False)
    )
    failed = [item for item in results if not item["pass"]]
    summary = {
        "passed": not failed, "cleanup": args.cleanup, "nproc": args.nproc,
        "hang_timeout_s": args.hang_timeout, "margin_s": args.margin,
        "legacy_defect_reproduced": legacy_reproduced,
        "cases": len(results), "failed": [item["case"] for item in failed], "results": results,
        "hang_job": hang, "normal_job": normal, "timeout_at": timeout_at,
        "hang_ranks": hang_starts, "normal_ranks": normal_starts, "normal_done": normal_dones,
        "final_sweep": sweep,
        "log": {key: str(path) for key, path in logs.items()},
    }
    print("RESULT_JSON " + json.dumps(summary, ensure_ascii=False, default=str))
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for item in failed:
        print(f"FAIL {item['case']}: "
              f"{json.dumps({k: v for k, v in item.items() if k not in ('case', 'pass')}, default=str)[:700]}")
    if failed:
        print(f"launcher self-test FAILED ({args.cleanup} cleanup): {len(failed)}/{len(results)} checks; exit 1"
              + (" -- this is the expected outcome for --cleanup legacy" if legacy_reproduced else ""))
        return EXIT_FAIL
    print(f"launcher self-test OK ({args.cleanup} cleanup): {len(results)} checks passed; exit 0")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
