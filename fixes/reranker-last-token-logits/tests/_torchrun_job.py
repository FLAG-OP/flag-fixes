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

Two layers of timeout meet when ``flag-fixes verify`` runs a launcher: verify kills the command
when its own budget (``timeout_seconds`` or ``--timeout``) ends, and a killed launcher cannot
clean up.  verify therefore passes the budget in ``FLAG_FIXES_COMMAND_TIMEOUT`` and
``resolve_timeout`` keeps the launcher's timeout short enough for the job to end and be
cleaned up before it.
"""
from __future__ import annotations

import os
import secrets
import signal
import subprocess
import time

JOB_ID_ENV = "RERANKER_JOB_ID"
# set by ``flag-fixes verify``: seconds until it kills the command it started
BUDGET_ENV = "FLAG_FIXES_COMMAND_TIMEOUT"
# what a launcher keeps for ending its job when the budget runs out: terminate_job waits up to
# 30 s for signalled processes to exit, the rest covers reading the log and writing the summary
RESERVE_S = 60.0


def new_job_id() -> str:
    return f"{os.getpid()}-{int(time.time())}-{secrets.token_hex(6)}"


def resolve_timeout(requested: float, started_at: float) -> tuple[float | None, str]:
    """The launcher's own timeout, kept inside the time budget ``flag-fixes verify`` gives the command.

    Standalone there is no budget and ``requested`` stands.  Under verify the command is killed
    the moment its budget ends, with no chance to clean up the job it started, so the launcher
    gives back what it has already spent since ``started_at`` and keeps ``RESERVE_S`` for the
    cleanup.  ``None`` means what is left would not fit a run and a cleanup: start nothing.
    """
    budget = os.environ.get(BUDGET_ENV)
    try:
        budget = float(budget) if budget else None
    except ValueError:  # not written by flag-fixes verify; do not let it break the launcher
        budget = None
    if budget is None:
        return requested, "--timeout"
    left = budget - (time.time() - started_at) - RESERVE_S
    if left <= 0:
        return None, (f"flag-fixes verify budget {budget:g} s minus the time already spent and the "
                      f"{RESERVE_S:g} s cleanup reserve leaves nothing for a run")
    return min(requested, left), (f"min(--timeout {requested:g} s, flag-fixes verify budget {budget:g} s minus "
                                  f"start-up and a {RESERVE_S:g} s cleanup reserve)")


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


def run_job(command: list[str], env: dict, log_path, timeout: float, cwd=None, job_id: str | None = None,
            started_at: float | None = None) -> dict:
    """Run ``command`` in a session of its own; on timeout end the agent and the job's processes.

    ``job_id`` is placed in the child's environment under ``JOB_ID_ENV`` and returned with
    the result.  ``cleanup`` is ``None`` unless the timeout fired.  ``started_at`` is when the
    launcher began; under ``flag-fixes verify`` the timeout is cut so that the job is over and
    cleaned up before the budget ends (``resolve_timeout``).  ``launched`` is ``False`` when
    the budget left no room for a run: nothing was started.
    """
    job_id = job_id or new_job_id()
    timeout, timeout_source = resolve_timeout(timeout, time.time() if started_at is None else started_at)
    result = {"job_id": job_id, "started_at": time.time(), "timed_out": False, "cleanup": None,
              "launched": True, "timeout_s": timeout, "timeout_source": timeout_source}
    if timeout is None:
        with open(log_path, "w", encoding="utf-8") as log:  # the launcher still has a log to read
            log.write(f"not launched: {timeout_source}\n")
        result.update(launched=False, returncode=None, elapsed_s=round(time.time() - result["started_at"], 1))
        return result
    env = dict(env)
    env[JOB_ID_ENV] = job_id
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
