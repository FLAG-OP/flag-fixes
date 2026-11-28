#!/usr/bin/env python3
"""Launcher self-test: what the two launchers rely on, checked on the CPU.

The two launchers (``e2e_reranker_train.py``, ``test_reranker_fsdp2_step.py``) share
``_torchrun_job.py`` and are run by ``flag-fixes verify``.  This script exercises those pieces
on their own:

1. Visible devices.  ``limit_visible_devices`` with the default visibility, an explicit CUDA
   list (kept, cut to the rank count, XPU already set) and an XPU-only list.

2. Relative paths of the E2E entry.  ``e2e_reranker_train.prepare`` is called from one
   directory with relative ``--model`` / ``--data`` / ``--out`` (on the command line and through
   the environment defaults).  The worker is started in ``tests/``, so every path that goes
   into the config has to be absolute and readable from there, and the config has to be in the
   directory the caller named.

3. Timeout cleanup scope.  Two torchrun jobs started from one worker script path -- the
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
   from this checkout is running.

4. verify's time budget.  ``flag-fixes verify`` kills a command when its budget ends, and a
   launcher killed that way cannot end the job it started.  verify passes the budget in
   ``FLAG_FIXES_COMMAND_TIMEOUT`` and ``run_job`` keeps its timeout inside it, minus what the
   launcher already spent and a reserve for the cleanup.  Checked from the real entry
   (``flag_fixes.cli.main``) with a short ``--timeout`` (``--verify-budget``) on a temporary
   fix whose declared command is a stand-in launcher (``_verify_budget_entry.py``: the real
   ``run_job``, a torchrun job that never returns, ``--timeout 600``).  Passing means: verify
   reports the command as failed because the launcher's own timeout fired -- not because
   verify killed it -- no process of that job is left when verify returns, and a control job
   started alongside is untouched and exits normally.

   ``--budget ignored`` makes the stand-in ignore the budget, as the launchers did before
   they read it: verify kills the stand-in and its job outlives it.  Expected to FAIL.

Every job the script starts is swept by its own job id at the end, whatever the outcome.

Exit 0  every check passes
Exit 1  a check failed (the expected outcome with ``--cleanup legacy`` or ``--budget ignored``)
Exit 2  precondition missing (torch.distributed.run / flag_fixes not importable), or the time
        budget of a enclosing flag-fixes verify cannot hold the whole run -- SKIP, nothing started
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
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
    BUDGET_ENV, JOB_ID_ENV, RESERVE_S, job_processes, limit_visible_devices, new_job_id, run_job, terminate_job,
)

EXIT_PASS, EXIT_FAIL, EXIT_SKIP = 0, 1, 2
WORKER = HERE / "_launcher_worker.py"
REPO = HERE.parents[2]  # fixes/<id>/tests -> the flag-fixes checkout: flag_fixes, catalog/schemas, templates
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


def torchrun_command(mode: str, nproc: int, stamps: Path, seconds: float | None = None) -> list[str]:
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
               str(WORKER), "--mode", mode, "--stamps", str(stamps)]
    if seconds is not None:
        command += ["--seconds", str(seconds)]
    return command


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

    The line after the group kill is the cleanup the launchers had (E2E and FSDP2 entry
    alike): remove the workers by the path of their entry script.
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
            result["cleanup"] = {"mode": "legacy", "command": f"pkill -KILL -f {WORKER}", "signalled": None}
    result["elapsed_s"] = round(time.time() - result["started_at"], 1)
    return result


def e2e_paths(record, workdir: Path) -> None:
    """Relative --model / --data / --out given from one directory, read back from the worker's directory."""
    import yaml

    import e2e_reranker_train as e2e

    caller = workdir / "caller"
    (caller / "model").mkdir(parents=True)
    (caller / "model" / "config.json").write_text("{}", encoding="utf-8")
    (caller / "data").mkdir()
    (caller / "data" / "train.parquet").write_bytes(b"")
    cases = [
        ("cli_arguments", {}, ["--model", "model", "--data", "data", "--out", "out"], caller / "out"),
        ("environment_defaults", {"RERANKER_MODEL": "model", "RERANKER_DATA": "data", "RERANKER_E2E_OUT": "runs"},
         [], caller / "runs"),
    ]
    saved = {key: os.environ.get(key) for _, env, _, _ in cases for key in env}
    previous = os.getcwd()
    try:
        os.chdir(caller)
        for name, env, argv, named_out in cases:
            for key, value in env.items():
                os.environ[key] = value
            try:
                args = e2e.parse_args(argv)
                prepared = e2e.prepare(args)
            finally:
                for key in env:
                    os.environ.pop(key, None)
            if not isinstance(prepared, tuple):
                record(f"e2e_paths:{name}", False, reason=prepared)
                continue
            out, config_path, config = prepared
            train = config["train"]
            paths = {"model": train["model"]["model_path"], "data": train["data"]["path"],
                     "validation": train["data"]["validation_path"], "test": train["data"]["test_path"],
                     "output_dir": train["output_dir"]}
            # absolute, all inside the directory the caller was in, config inside the named output directory
            absolute_and_named = (
                all(Path(p).is_absolute() and Path(p).resolve().is_relative_to(caller.resolve())
                    for p in paths.values())
                and config_path.resolve().is_relative_to(named_out.resolve()) and out == args.out)
            # the worker is started with cwd=tests/: everything must be readable from there
            os.chdir(e2e.HERE)
            try:
                readable = (yaml.safe_load(config_path.read_text(encoding="utf-8")) == config
                            and (Path(paths["model"]) / "config.json").is_file()
                            and bool(list(Path(paths["data"]).glob("*.parquet")))
                            and Path(paths["output_dir"]).is_dir())
            finally:
                os.chdir(caller)
            record(f"e2e_paths:{name}", absolute_and_named and readable, given=dict(env, argv=argv),
                   paths=paths, config=str(config_path), absolute_and_in_named_directory=absolute_and_named,
                   readable_from_worker_directory=readable)
        reason = e2e.prepare(e2e.parse_args(["--model", "no-such-model", "--data", "data", "--out", "out"]))
        record("e2e_paths:missing_input_is_a_precondition", isinstance(reason, str) and "checkpoint" in reason,
               reason=reason)
    finally:
        os.chdir(previous)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def verify_budget(record, budget_mode: str, budget_seconds: float, base_env: dict, workdir: Path) -> dict:
    """flag-fixes verify, with a short --timeout, on a stand-in launcher that starts a job through run_job."""
    import yaml

    from flag_fixes import catalog, cli

    room = workdir / "verify"
    root = room / "manager"
    source = root / "fixes/budget-probe"
    (source / "tests").mkdir(parents=True)
    (root / "catalog/fixes").mkdir(parents=True)
    shutil.copytree(REPO / "catalog/schemas", root / "catalog/schemas")
    shutil.copyfile(HERE / "_verify_budget_entry.py", source / "tests/entry.py")
    target = room / "target"  # the stand-in "patched tree": one file whose hash the manifest declares
    (target / "pkg").mkdir(parents=True)
    (target / "pkg/fixed.py").write_text("fixed\n", encoding="utf-8")
    stamps, job_file = room / "job", room / "job-id.json"
    command = {
        "name": "budget-probe", "script": "tests/entry.py", "level": "regression", "timeout_seconds": 600,
        "args": ["--helper-dir", str(HERE), "--worker", str(WORKER), "--stamps", str(stamps),
                 "--job-file", str(job_file)],
    }
    if budget_mode == "ignored":
        command["args"] += ["--budget", "ignored"]
    manifest = yaml.safe_load((REPO / "templates/fix/fix.yaml").read_text())
    manifest.update(id="budget-probe", title="timeout budget probe")
    manifest["source"]["path"] = "fixes/budget-probe"
    manifest["targets"] = [{"path": "pkg/fixed.py", "before_sha256": "0" * 64,
                            "after_sha256": catalog.sha256(target / "pkg/fixed.py")}]
    manifest["validation"]["commands"] = [command]
    (source / "fix.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    # verify ends the stand-in after this many seconds unless the launcher's own timeout fires first
    budget = 25.0 if budget_mode == "ignored" else budget_seconds
    # the control job has to be alive when the stand-in's job is ended and finish after verify returns
    sleep_seconds = max(40.0, budget - 35.0)
    control_id = f"selftest-control-{new_job_id()}"
    control_stamps = room / "control"
    control: dict = {}

    def run_control() -> None:
        control.update(run_job(torchrun_command("sleep", 2, control_stamps, sleep_seconds), base_env,
                               room / "control.log", sleep_seconds + 600, cwd=str(HERE), job_id=control_id))

    thread = threading.Thread(target=run_control)
    thread.start()
    buffer = io.StringIO()
    code, report = None, {}
    verify_started = time.time()
    try:
        with contextlib.redirect_stdout(buffer):
            code = cli.main(["--root", str(root), "verify", "budget-probe", "--target", str(target),
                             "--python", sys.executable, "--timeout", str(int(budget))])
    finally:
        returned_at = time.time()
        filed = json.loads(job_file.read_text(encoding="utf-8")) if job_file.is_file() else {}
        # the moment verify is back: is anything left of the job the stand-in started?
        left = job_processes(filed["job_id"]) if filed else None
        thread.join()
    try:
        report = json.loads(buffer.getvalue())
    except ValueError:
        report = {"unparsed": buffer.getvalue()[-2000:]}
    entry = report["results"][0] if isinstance(report.get("results"), list) and report["results"] else {}
    launcher = {}
    for line in (entry.get("stdout") or "").splitlines():
        if line.startswith("RESULT_JSON "):
            launcher = json.loads(line[len("RESULT_JSON "):])
            break

    job_id = filed.get("job_id")
    # when the stand-in's own timeout ended its job: from its record if it returned, otherwise
    # (killed by verify) the moment verify's budget ran out
    timeout_at = ((launcher.get("started_at") or filed.get("filed_at") or returned_at)
                  + launcher.get("timeout_s", budget)) if filed else returned_at
    job_starts = read_stamps(stamps, "start", 2)
    control_starts = read_stamps(control_stamps, "start", 2)
    control_dones = read_stamps(control_stamps, "done", 2)

    record("verify_budget:entry_produced_a_report", code == 1 and bool(entry), exit_code=code,
           report_keys=sorted(report) if isinstance(report, dict) else None)
    record("verify_budget:command_reported_failed", entry.get("passed") is False and report.get("passed") is False,
           returncode=entry.get("returncode"), error=entry.get("error"))
    record("verify_budget:launcher_returned_by_itself",
           entry.get("error") != "timeout" and entry.get("returncode") == 1
           and launcher.get("launched") is True and launcher.get("timed_out") is True,
           returncode=entry.get("returncode"), error=entry.get("error"),
           launcher_timeout_s=launcher.get("timeout_s"), launcher_timeout_source=launcher.get("timeout_source"),
           note="false means verify killed the launcher on its own timeout before the launcher could end its job")
    record("verify_budget:launcher_finished_inside_the_budget",
           launcher.get("timeout_s") is not None and launcher["timeout_s"] + RESERVE_S <= budget
           and (returned_at - verify_started) < budget,
           budget_s=budget, launcher_timeout_s=launcher.get("timeout_s"), reserve_s=RESERVE_S,
           verify_returned_after_s=round(returned_at - verify_started, 1))
    record("verify_budget:job_was_running_when_its_timeout_fired",
           all(row is not None and row["time"] < timeout_at for row in job_starts),
           start_time=[row["time"] if row else None for row in job_starts], timeout_at=timeout_at)
    record("verify_budget:no_process_left_when_verify_returns", left == [], job_id=job_id, processes_left=left)
    record("verify_budget:control_job_unaffected",
           control.get("returncode") == 0 and control.get("timed_out") is False
           and all(row is not None and row["time"] < timeout_at for row in control_starts)
           and all(row is not None and row["time"] > timeout_at for row in control_dones),
           returncode=control.get("returncode"),
           start_time=[row["time"] if row else None for row in control_starts],
           done_time=[row["time"] if row else None for row in control_dones], timeout_at=timeout_at)
    return {
        "budget_s": budget, "budget_mode": budget_mode, "job_id": job_id, "control_job_id": control_id,
        "verify_exit_code": code, "verify_returned_after_s": round(returned_at - verify_started, 1),
        "verify_report": report, "launcher": launcher, "job_ranks": job_starts,
        "processes_left_when_verify_returned": left, "control": control,
        "control_ranks": control_starts, "control_done": control_dones,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cleanup", choices=["scoped", "legacy"], default="scoped")
    parser.add_argument("--budget", choices=["honoured", "ignored"], default="honoured",
                        help="how the stand-in launcher of the verify part treats verify's time budget")
    parser.add_argument("--nproc", type=int, default=2, help="ranks per job (default 2)")
    parser.add_argument("--hang-timeout", type=float, default=60.0,
                        help="launcher timeout of the hang job; must exceed the torchrun start-up")
    parser.add_argument("--margin", type=float, default=30.0,
                        help="the normal ranks sleep this long past the hang job's timeout")
    parser.add_argument("--verify-budget", type=float, default=75.0,
                        help="--timeout given to flag-fixes verify in the budget part; needs the stand-in's "
                             f"start-up plus a run plus the {RESERVE_S:g} s reserve")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if importlib.util.find_spec("torch.distributed.run") is None:
        print("SKIP: torch.distributed.run not importable; exit 2")
        return EXIT_SKIP
    if args.cleanup == "legacy" and shutil.which("pkill") is None:
        print("SKIP: --cleanup legacy needs pkill; exit 2")
        return EXIT_SKIP
    # the real verify entry: flag_fixes from this checkout
    if (REPO / "src/flag_fixes").is_dir() and str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    try:
        import yaml  # noqa: F401  (the E2E entry and the temporary manifest need it)

        from flag_fixes import cli  # noqa: F401
    except Exception as error:
        print(f"SKIP: flag_fixes / yaml not importable from {REPO} ({type(error).__name__}: {error}); exit 2")
        return EXIT_SKIP
    # This script starts jobs of its own, so it lives by the same rule as the launchers: when
    # flag-fixes verify gives it less time than the whole run plus a cleanup reserve takes, it
    # starts nothing.  Run time is the two job pairs (hang timeout + margin, and the control
    # job of the budget part) plus a little for start-up and reporting.
    needed = ((args.hang_timeout + args.margin + 5.0) + (max(40.0, args.verify_budget - 35.0) + 5.0)
              + RESERVE_S)
    if os.environ.get(BUDGET_ENV) and float(os.environ[BUDGET_ENV]) < needed:
        print(f"SKIP: flag-fixes verify gives this self-test {float(os.environ[BUDGET_ENV]):g} s and it needs "
              f"{needed:g} s including the {RESERVE_S:g} s cleanup reserve; nothing was started; exit 2")
        return EXIT_SKIP
    # the jobs this script starts itself are not under the budget verify may have given this script
    # (it was checked above), and nothing here touches a device
    os.environ.pop(BUDGET_ENV, None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

    def record(case: str, ok, **detail) -> bool:
        results.append({"case": case, "pass": bool(ok), **detail})
        return bool(ok)

    base_env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "XPU_VISIBLE_DEVICES", JOB_ID_ENV):
        base_env.pop(key, None)
    base_env["CUDA_VISIBLE_DEVICES"] = ""  # CPU jobs: no device is touched either way

    job_ids: dict[str, str] = {}
    budget_part: dict = {}
    with tempfile.TemporaryDirectory(prefix="reranker-launcher-selftest-") as scratch:
        workdir = Path(scratch)
        # ---- 1. visible devices: pure function, no process started
        for name, given_env, nproc, expected in VISIBLE_CASES:
            env = dict(given_env)
            mapping = limit_visible_devices(env, nproc)
            record(f"visible_devices:{name}", env == expected, given=given_env, expected=expected, actual=env,
                   action=mapping["action"])

        # ---- 2. relative paths of the E2E entry
        e2e_paths(record, workdir / "e2e-paths")

        # ---- 3. two jobs from one worker script path
        sleep_seconds = args.hang_timeout + args.margin
        job_ids = {"hang": f"selftest-hang-{new_job_id()}", "normal": f"selftest-normal-{new_job_id()}"}
        commands = {"hang": torchrun_command("hang", args.nproc, stamps["hang"]),
                    "normal": torchrun_command("sleep", args.nproc, stamps["normal"], sleep_seconds)}
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
                    pids = {row["pid"] for row in read_stamps(stamps[key], "start", args.nproc) if row}
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

        # ---- 4. verify's time budget, from the real entry
        try:
            budget_part = verify_budget(record, args.budget, args.verify_budget, base_env, workdir)
        except Exception as error:  # surfaced as a failed check
            record("verify_budget:ran", False, error=f"{type(error).__name__}: {error}")
        for key, job_id in (("budget-job", budget_part.get("job_id")), ("budget-control", budget_part.get("control_job_id"))):
            if job_id:
                job_ids[key] = job_id

    sweep = {key: terminate_job(job_id, grace=10.0) for key, job_id in job_ids.items()}

    by_case = {item["case"]: item["pass"] for item in results}
    legacy_reproduced = (
        args.cleanup == "legacy"
        and not by_case.get("normal_job:exits_zero", True)
        and by_case.get("normal_job:ranks_started_before_hang_timeout", False)
    )
    budget_reproduced = (
        args.budget == "ignored"
        and by_case.get("verify_budget:launcher_returned_by_itself") is False
        and by_case.get("verify_budget:no_process_left_when_verify_returns") is False
        and by_case.get("verify_budget:control_job_unaffected", False)
    )
    failed = [item for item in results if not item["pass"]]
    summary = {
        "passed": not failed, "cleanup": args.cleanup, "budget": args.budget, "nproc": args.nproc,
        "hang_timeout_s": args.hang_timeout, "margin_s": args.margin, "verify_budget_s": args.verify_budget,
        "legacy_defect_reproduced": legacy_reproduced, "budget_defect_reproduced": budget_reproduced,
        "cases": len(results), "failed": [item["case"] for item in failed], "results": results,
        "hang_job": hang, "normal_job": normal, "timeout_at": timeout_at,
        "hang_ranks": hang_starts, "normal_ranks": normal_starts, "normal_done": normal_dones,
        "verify_budget": budget_part,
        "jobs": job_ids,
        "final_sweep": sweep,
        "log": {key: str(path) for key, path in logs.items()},
    }
    print("RESULT_JSON " + json.dumps(summary, ensure_ascii=False, default=str))
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for item in failed:
        print(f"FAIL {item['case']}: "
              f"{json.dumps({k: v for k, v in item.items() if k not in ('case', 'pass')}, default=str)[:700]}")
    if failed:
        expected = []
        if legacy_reproduced:
            expected.append("--cleanup legacy")
        if budget_reproduced:
            expected.append("--budget ignored")
        print(f"launcher self-test FAILED: {len(failed)}/{len(results)} checks; exit 1"
              + (f" -- this is the expected outcome for {' and '.join(expected)}" if expected else ""))
        return EXIT_FAIL
    print(f"launcher self-test OK: {len(results)} checks passed; exit 0")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
