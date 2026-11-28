"""A stand-in for the two real launchers, started through flag-fixes verify by test_reranker_launchers.py.

Same shape as ``e2e_reranker_train.py`` and ``test_reranker_fsdp2_step.py``: a ``--timeout``
that defaults to 600 s, one torchrun job started through the real ``run_job``, then wait and
report.  The job id is printed and filed before the wait, so the test can look for processes
of this job once verify has returned.

``--budget ignored`` is the behaviour the launchers had before they read verify's time budget:
the entry keeps its own ``--timeout`` even when flag-fixes verify gives the command less time.
verify then kills the entry while its job is still running, and the job outlives it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

STARTED_AT = time.time()  # the clock for flag-fixes verify's time budget starts here


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper-dir", required=True, help="directory holding _torchrun_job.py")
    parser.add_argument("--worker", required=True, help="worker script the torchrun job runs")
    parser.add_argument("--stamps", required=True, help="directory for the workers' start records and the job log")
    parser.add_argument("--job-file", required=True, help="where this entry files its job id")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--budget", choices=["honoured", "ignored"], default="honoured")
    args = parser.parse_args()

    sys.path.insert(0, args.helper_dir)
    from _torchrun_job import BUDGET_ENV, job_processes, new_job_id, run_job

    if args.budget == "ignored":
        os.environ.pop(BUDGET_ENV, None)
    stamps = Path(args.stamps)
    stamps.mkdir(parents=True, exist_ok=True)
    job_id = new_job_id()
    Path(args.job_file).write_text(json.dumps({"job_id": job_id, "filed_at": time.time()}), encoding="utf-8")
    print(f"JOB {job_id}", flush=True)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               args.worker, "--mode", "hang", "--stamps", str(stamps)]
    job = run_job(command, dict(os.environ), stamps / "job.log", args.timeout, started_at=STARTED_AT,
                  job_id=job_id)
    job["processes_after_return"] = job_processes(job_id)
    print("RESULT_JSON " + json.dumps(job, default=str), flush=True)
    if not job["launched"] or job["timed_out"] or job["returncode"] != 0:
        print(f"budget probe FAILED (launched={job['launched']}, timed_out={job['timed_out']}, "
              f"returncode={job['returncode']}); exit 1", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
