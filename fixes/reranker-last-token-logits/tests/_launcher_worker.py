"""Worker for test_reranker_launchers.py (started through torchrun by the self-test).

Lightweight on purpose: no torch import, no device, no process group.  Every rank records
its pid, the job id it inherited and (in ``hang`` mode) the pid of one child it leaves
behind, then either sleeps for ``--seconds`` and exits 0 (``sleep``) or never returns
(``hang``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torchrun_job import JOB_ID_ENV, announce_worker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sleep", "hang"], required=True)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--stamps", required=True, help="directory for the per-rank start / done records")
    args = parser.parse_args()

    stamps = Path(args.stamps)
    stamps.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", "0")
    child_pid = None
    if args.mode == "hang":
        child_pid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(86400)"]).pid
    announce_worker(f"selftest-{args.mode}")
    (stamps / f"rank{rank}.start").write_text(json.dumps({
        "pid": os.getpid(), "child_pid": child_pid, "time": time.time(), "job_id": os.environ.get(JOB_ID_ENV),
    }), encoding="utf-8")

    if args.mode == "sleep":
        time.sleep(args.seconds)
        (stamps / f"rank{rank}.done").write_text(json.dumps({"pid": os.getpid(), "time": time.time()}),
                                                 encoding="utf-8")
        return 0
    while True:
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
