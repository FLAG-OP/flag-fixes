"""One torchrun job per launcher run, with a timeout cleanup limited to that job.

``e2e_reranker_train.py`` and ``test_reranker_fsdp2_step.py`` start their workers through
``torch.distributed.run``.  torchrun puts every worker into a session of its own, so killing
the agent's process group does not reach a worker that is stuck in a collective.  The
launchers used to remove such workers by entry-point name (``pkill -KILL -f <worker
script>``), which also matches the workers of any other run started from the same checkout
-- a baseline and a patched run in parallel, for instance.

A job is therefore identified before it starts: ``run_job`` puts a random id into the
environment of the torchrun agent, which torchrun hands down to the workers and they to
their children.  After a timeout the cleanup signals exactly the processes whose
environment carries that id, and nothing else.  ``tests/test_reranker_launchers.py`` checks
this with two concurrent jobs sharing one worker script path.

The same module holds ``limit_visible_devices``, the only place the launchers touch the
visible-device variables.
"""
from __future__ import annotations

import os
import secrets
import signal
import subprocess
import time

JOB_ID_ENV = "RERANKER_JOB_ID"


def new_job_id() -> str:
    return f"{os.getpid()}-{int(time.time())}-{secrets.token_hex(6)}"


def announce_worker(tag: str) -> None:
    """Print the job id and pid of the calling worker, so the job's log names its processes."""
    print(f"[job-worker] tag={tag} job={os.environ.get(JOB_ID_ENV)} pid={os.getpid()} "
          f"rank={os.environ.get('RANK', '0')}", flush=True)


def job_processes(job_id: str) -> list[dict]:
    """Processes whose environment carries ``job_id``, lowest pid first.

    That is the torchrun agent (while it lives), every worker and whatever the workers
    spawned, since environments are inherited.  Only processes of the current user are
    looked at.  A process that exits between the listing and the read is skipped.
    """
    wanted = f"{JOB_ID_ENV}={job_id}".encode()
    try:
        names = [entry.name for entry in os.scandir("/proc") if entry.name.isdigit()]
    except OSError as error:
        raise LookupError(f"/proc is not readable ({error}); cannot look up the job's processes") from None
    found = []
    for name in names:
        try:
            if os.stat(f"/proc/{name}").st_uid != os.getuid():
                continue
            with open(f"/proc/{name}/environ", "rb") as handle:
                if wanted not in handle.read().split(b"\0"):
                    continue
            with open(f"/proc/{name}/cmdline", "rb") as handle:
                parts = [part.decode(errors="replace") for part in handle.read().split(b"\0") if part]
        except OSError:
            continue
        found.append({"pid": int(name), "cmdline": " ".join(parts)[:400]})
    return sorted(found, key=lambda item: item["pid"])


def terminate_job(job_id: str, grace: float = 30.0) -> dict:
    """SIGKILL every process that carries ``job_id``; wait up to ``grace`` seconds for them to exit.

    Each pid is taken from a fresh ``job_processes`` scan immediately before the signal, so
    the decision is always based on what the process is now.  A pid that is recycled in the
    instant between the scan and ``os.kill`` is not caught -- pidfd would close that window,
    but this interpreter build does not offer it.
    """
    cleanup = {"job_id": job_id, "signalled": [], "survived": None, "grace_s": grace}
    try:
        signalled: dict[int, dict] = {}
        deadline = time.monotonic() + grace
        left = job_processes(job_id)
        while left:
            for item in left:
                signalled.setdefault(item["pid"], item)
                try:
                    os.kill(item["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)
            left = job_processes(job_id)
        # the last scan above may predate the final signals by one interval: look once more
        left = job_processes(job_id)
        cleanup.update(signalled=list(signalled.values()), survived=left)
    except LookupError as error:
        cleanup["note"] = str(error)
    return cleanup


def run_job(command: list[str], env: dict, log_path, timeout: float, cwd=None, job_id: str | None = None) -> dict:
    """Run ``command`` in a session of its own; on timeout end the agent and the job's processes.

    ``job_id`` is placed in the child's environment under ``JOB_ID_ENV`` and returned with
    the result.  ``cleanup`` is ``None`` unless the timeout fired.
    """
    job_id = job_id or new_job_id()
    env = dict(env)
    env[JOB_ID_ENV] = job_id
    result = {"job_id": job_id, "started_at": time.time(), "timed_out": False, "cleanup": None}
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                                   start_new_session=True)
        try:
            result["returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            try:
                # the agent and anything it forked into its group; workers sit in their own sessions
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            result["returncode"] = process.wait()
            result["cleanup"] = terminate_job(job_id)
    result["elapsed_s"] = round(time.time() - result["started_at"], 1)
    return result


def limit_visible_devices(env: dict, nproc: int) -> dict:
    """Keep at most ``nproc`` cards of an explicit ``CUDA_VISIBLE_DEVICES``; mirror it to XPU.

    Only an explicit ``CUDA_VISIBLE_DEVICES`` is mapped.  With the default visibility (the
    variable is unset), or with only ``XPU_VISIBLE_DEVICES`` set, the environment is left as
    it is: nothing is known about the vendor list's index space, so no CUDA list is invented
    for it.  An ``XPU_VISIBLE_DEVICES`` that is already set is never overwritten.
    """
    record = {
        "nproc": nproc,
        "before": {name: env.get(name) for name in ("CUDA_VISIBLE_DEVICES", "XPU_VISIBLE_DEVICES")},
    }
    cuda = env.get("CUDA_VISIBLE_DEVICES")
    if cuda is None:
        record["action"] = ("XPU_VISIBLE_DEVICES only: left as set" if "XPU_VISIBLE_DEVICES" in env
                            else "default visibility: nothing mapped")
    else:
        actions = ["kept"]
        if nproc > 1:
            entries = cuda.split(",")
            if len(entries) > nproc:
                cuda = ",".join(entries[:nproc])
                env["CUDA_VISIBLE_DEVICES"] = cuda
                actions = [f"cut to the first {nproc} entries"]
        if "XPU_VISIBLE_DEVICES" not in env:
            env["XPU_VISIBLE_DEVICES"] = cuda
            actions.append("mirrored to XPU_VISIBLE_DEVICES")
        else:
            actions.append("XPU_VISIBLE_DEVICES already set")
        record["action"] = "; ".join(actions)
    record["after"] = {name: env.get(name) for name in ("CUDA_VISIBLE_DEVICES", "XPU_VISIBLE_DEVICES")}
    return record
