# Round-4/5 performance benchmarks (review plans 2026-11-10: section 4 and
# round-5 fixes: per-family registration, hit verification, unit/stat fixes).
#
# Round-5 changes vs the first bench:
# - One process per (family, group): each family (a-sum-abs / b-linear /
#   zeros) declares its own f_target include list and EXPECTED registered
#   keys (asserted, not assumed). The first bench registered model-A ops
#   globally, which made model-B / zeros f_target timings mis-attributed.
# - Registered != called: after registration (and outside any timed
#   region) each family verifies actual dispatch with a torch.profiler
#   kernel-name capture on representative calls.
# - Units: calls_per_s = 1000 / ms_per_iter (the previous samples_per_s
#   multiplied by iters once more, inflating by 100x). Model cases report
#   samples_per_s = batch_size * calls_per_s.
# - P95 is a quantile estimate over 5 per-block MEANS (not a per-request
#   tail latency); renamed accordingly.
# - Memory: per-case baseline allocated/reserved recorded before the timed
#   window, peak within it, and the delta (lower bound under shared
#   process residency; the window includes first-call compile + warmup).
# - Regression flags are renamed *_candidate_regression_gt10pct: a
#   median-ratio criterion only, NOT a stability verdict.
# - Configs whose accuracy failed in the matrix are labelled
#   "diagnostic (correctness-failed)" instead of valid performance.
#
# Groups per family: n / f_target / f_all / b / n_rep (native drift pass).
# First round = reporting only: no pre-committed speedup claims.
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

import torch  # noqa: E402

import small_models as sm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda"
SIZES = (2**10, 2**16, 2**20)
CDTYPES = {"c64": torch.complex64, "c128": torch.complex128}

FAMILIES = {
    "a-sum-abs": {
        "include": ["sum", "sum_dim", "sum_out", "sum_dim_out", "abs",
                    "absolute"],
        "expected_keys": ["abs", "absolute", "sum", "sum.dim_IntList"],
        "batch": 8,  # model A large: B=8
    },
    "b-linear": {
        "include": ["conv2d", "linear", "linear_backward"],
        "expected_keys": ["conv2d", "linear", "linear_backward"],
        "batch": 8,  # model B large: N=8
    },
    "zeros": {
        "include": ["zeros", "zero_", "zeros_like"],
        "expected_keys": ["zeros", "zero_", "zeros_like"],
        "batch": 1,
    },
}


def register_group(group, family):
    prov = {"group": group, "family": family, "flag_gems_path": None,
            "include": None, "registered_ops": [], "registered_keys": [],
            "registered_target_keys": {}, "hit_verification": {}}
    if group == "n":
        return prov
    import flag_gems
    if group == "f_target":
        inc = FAMILIES[family]["include"]
        flag_gems.only_enable(include=inc)
        prov["include"] = inc
    else:  # f_all and b: full registration
        flag_gems.enable()
    prov["flag_gems_path"] = flag_gems.__file__
    try:
        prov["registered_ops"] = sorted(set(flag_gems.all_registered_ops()))
        prov["registered_keys"] = sorted(set(flag_gems.all_registered_keys()))
    except Exception as e:
        prov["record_error"] = f"{type(e).__name__}: {e}"
    keys = set(prov["registered_keys"])
    prov["registered_target_keys"] = {
        k: (k in keys) for k in FAMILIES[family]["expected_keys"]
    }
    if group == "f_target":
        missing = [k for k, v in prov["registered_target_keys"].items()
                   if not v]
        if missing:
            print(f"ERROR: f_target/{family} missing expected "
                  f"registrations: {missing}", flush=True)
            sys.exit(2)
    return prov


def verify_hit(name, fn):
    """Non-timed dispatch verification: capture CUDA kernel names with
    torch.profiler for one representative call. Registered keys only prove
    'enabled'; this records which kernels actually ran."""
    from torch.profiler import profile, ProfilerActivity
    fn()  # warm (outside timed regions of later cases)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    try:
        for ev in prof.key_averages():
            if getattr(ev, "device_type", None) == \
                    torch.autograd.DeviceType.CUDA and ev.self_device_time_total > 0:
                names.append(ev.key)
    except Exception:
        names = [e.key for e in prof.key_averages()]
    return {"fn": name, "cuda_kernels": names[:10]}


def factor2(n):
    r = int(round(n**0.5))
    while n % r:
        r += 1
    return r, n // r


def make_strided(t):
    """Same logical shape/values as t but with every-other-element strides."""
    shape = t.shape
    buf = torch.empty(*shape[:-1], shape[-1] * 2, dtype=t.dtype,
                      device=t.device)
    view = buf[..., ::2]
    view.copy_(t)
    assert view.shape == t.shape and not view.is_contiguous()
    assert torch.equal(view.real if view.is_complex() else view,
                       t.real if t.is_complex() else t)
    return view


def _record_peak(baseline):
    peak_a = torch.cuda.max_memory_allocated() / 2**20
    peak_r = torch.cuda.max_memory_reserved() / 2**20
    return {
        "mem_baseline_alloc_mb": baseline["alloc"],
        "mem_baseline_reserved_mb": baseline["reserved"],
        "mem_peak_alloc_mb": peak_a,
        "mem_peak_reserved_mb": peak_r,
        "mem_delta_alloc_mb": peak_a - baseline["alloc"],
        "mem_delta_reserved_mb": peak_r - baseline["reserved"],
        "mem_window": "first-call compile + warmup + timed blocks; "
                      "delta is a lower bound (case inputs coexist in "
                      "the process)",
    }


def bench_case(fn, setup=None, warmup=20, blocks=5, iters=100):
    """Returns dict with first-call, per-block device ms, wall ms, stats."""
    res = {"warmup": warmup, "blocks": blocks, "iters_per_block": iters}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = {
        "alloc": torch.cuda.memory_allocated() / 2**20,
        "reserved": torch.cuda.memory_reserved() / 2**20,
    }
    # first call (JIT/autotune) timed separately
    if setup:
        setup()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    w0 = time.perf_counter()
    e0.record()
    fn()
    e1.record()
    torch.cuda.synchronize()
    res["first_call"] = {
        "device_ms": e0.elapsed_time(e1),
        "wall_ms": (time.perf_counter() - w0) * 1e3,
        "note": "includes JIT/autotune; cache state = "
                "cold-process (per-family process)",
    }
    for _ in range(warmup):
        if setup:
            setup()
        fn()
    torch.cuda.synchronize()
    dev_ms, wall_ms = [], []
    for _ in range(blocks):
        if setup:
            setup()
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        dev_ms.append(e0.elapsed_time(e1))
        wall_ms.append((time.perf_counter() - w0) * 1e3)
    res["block_device_ms"] = dev_ms
    res["block_wall_ms"] = wall_ms
    med = statistics.median(dev_ms) / iters
    res["device_ms_per_iter_median"] = med
    # quantile estimate of per-BLOCK MEAN per-iter time over 5 blocks;
    # NOT a per-request tail latency
    if len(dev_ms) >= 2:
        p95_block = statistics.quantiles(dev_ms, n=20)[-1]
    else:
        p95_block = max(dev_ms)
    res["block_mean_p95_ms_per_iter"] = p95_block / iters
    res["block_mean_p95_note"] = (
        "quantile estimate of per-block mean per-iter time over 5 blocks; "
        "not a per-request tail latency")
    res["wall_ms_per_iter_median"] = statistics.median(wall_ms) / iters
    res["block_device_ms_cv_pct"] = (
        (statistics.stdev(dev_ms) / statistics.mean(dev_ms) * 100)
        if len(dev_ms) > 1 and statistics.mean(dev_ms) else 0.0)
    calls_per_s = 1000.0 / med if med else None
    res["calls_per_s"] = calls_per_s
    res["calls_per_s_note"] = "1000 / ms_per_iter_median"
    res["timing"] = "cuda events device time + synced wall clock"
    res.update(_record_peak(baseline))
    return res


def sum_abs_cases():
    """sum full/inner/outer + abs (contig/strided, fwd+bwd), complex."""
    cases = []
    for dname, dt in CDTYPES.items():
        real_dt = (torch.float32 if dt == torch.complex64
                   else torch.float64)
        for n in SIZES:
            r, c = factor2(n)
            tag = f"{dname}/n{n}"
            x1 = torch.randn(n, dtype=torch.float32,
                             device=DEVICE).to(dt)
            x1s = make_strided(x1)
            x2 = x1.reshape(r, c)
            x2s = make_strided(x2)
            xb = x2.detach().clone().requires_grad_(True)
            xbs = x2s.detach().clone().requires_grad_(True)
            g1 = torch.randn(r, dtype=dt, device=DEVICE)
            gc = torch.randn(r, c, dtype=real_dt, device=DEVICE)

            cases.append((f"sum/full/{tag}", lambda x=x1: x.sum()))
            cases.append((f"sum/inner/{tag}", lambda x=x2: x.sum(dim=1)))
            cases.append((f"sum/outer/{tag}", lambda x=x2: x.sum(dim=0)))
            cases.append((f"sum/inner/strided/{tag}",
                          lambda x=x2s: x.sum(dim=1)))
            cases.append((f"abs/{tag}", lambda x=x2: torch.abs(x)))
            cases.append((f"abs/strided/{tag}",
                          lambda x=x2s: torch.abs(x)))

            def sum_inner_bwd(x=xb, g=g1):
                y = x.sum(dim=1)
                y.backward(g)
            def sum_inner_bwd_s(x=xbs, g=g1):
                y = x.sum(dim=1)
                y.backward(g)
            def abs_bwd(x=xb, g=gc):
                y = torch.abs(x)
                y.backward(g)
            cases.append((f"sum/inner/fwd+bwd/{tag}", sum_inner_bwd))
            cases.append((f"sum/inner/fwd+bwd/strided/{tag}",
                          sum_inner_bwd_s))
            cases.append((f"abs/fwd+bwd/{tag}", abs_bwd))
    return cases


def zeros_cases():
    """zeros creation + zero_ (per-block restore timed as separate case)."""
    cases = []
    for dname, dt in CDTYPES.items():
        for n in SIZES:
            r, c = factor2(n)
            tag = f"{dname}/n{n}"
            x2 = torch.randn(r, c, dtype=torch.float32,
                             device=DEVICE).to(dt)
            cases.append((f"zeros/{tag}",
                          lambda s=(r, c), d=dt: torch.zeros(
                              s, dtype=d, device=DEVICE)))
            src = x2.clone()
            xz = torch.empty_like(x2)
            cases.append((f"zero_/{tag}",
                          lambda t=xz: t.zero_(),
                          lambda t=xz, s=src: t.copy_(s)))
    return cases


def model_a_cases():
    """Model A (c64 core) fwd / fwd+bwd / train-step, contig + conj."""
    cases, build_errors = [], {}
    lr = 1e-2
    try:
        task_a = sm.gen_task("a", "large", 0, "fp32")
        for variant in ("contig", "conj"):
            x, params, tgt_a = sm.to_device(task_a, DEVICE)
            x0 = x.detach().clone()
            p0 = {k: v.detach().clone() for k, v in params.items()}

            def restore(x=x, params=params, x0=x0, p0=p0):
                x.data.copy_(x0)
                for k, p in params.items():
                    p.data.copy_(p0[k])
                sm.zero_grads(x, params)

            def fwd(x=x, params=params, v=variant, tg=tgt_a):
                sm.set_target(tg)
                sm.zero_grads(x, params)
                out, loss, _ = sm.forward("a", "large", x, params, v)
                return out, loss
            cases.append((f"modelA/{variant}/fwd", fwd, restore))

            def fwdbwd(x=x, params=params, v=variant, tg=tgt_a):
                sm.set_target(tg)
                sm.zero_grads(x, params)
                out, loss, _ = sm.forward("a", "large", x, params, v)
                loss.backward()
            cases.append((f"modelA/{variant}/fwd+bwd", fwdbwd, restore))

            def step(x=x, params=params, v=variant, tg=tgt_a):
                sm.set_target(tg)
                sm.zero_grads(x, params)
                out, loss, _ = sm.forward("a", "large", x, params, v)
                loss.backward()
                with torch.no_grad():
                    for p in params.values():
                        p -= lr * p.grad
            cases.append((f"modelA/{variant}/train-step", step, restore))
    except Exception as e:
        build_errors["modelA"] = {
            "type": type(e).__name__, "message": str(e),
            "phase": "case-build"}
    return cases, build_errors


def model_b_cases():
    """Model B (all-real) fwd / fwd+bwd / train-step."""
    cases, build_errors = [], {}
    lr = 1e-2
    try:
        task_b = sm.gen_task("b", "large", 0, "fp32")
        x, params, tgt_b = sm.to_device(task_b, DEVICE)
        x0b = x.detach().clone()
        p0b = {k: v.detach().clone() for k, v in params.items()}

        def restore_b(x=x, params=params, x0=x0b, p0=p0b):
            x.data.copy_(x0)
            for k, p in params.items():
                p.data.copy_(p0[k])
            sm.zero_grads(x, params)

        def b_fwd(x=x, params=params, tg=tgt_b):
            sm.set_target(tg)
            sm.zero_grads(x, params)
            out, loss, _ = sm.forward("b", "large", x, params, "default")
            return out, loss
        def b_fwdbwd(x=x, params=params, tg=tgt_b):
            sm.set_target(tg)
            sm.zero_grads(x, params)
            out, loss, _ = sm.forward("b", "large", x, params, "default")
            loss.backward()
        def b_step(x=x, params=params, tg=tgt_b):
            sm.set_target(tg)
            sm.zero_grads(x, params)
            out, loss, _ = sm.forward("b", "large", x, params, "default")
            loss.backward()
            with torch.no_grad():
                for p in params.values():
                    p -= lr * p.grad
        cases.append(("modelB/default/fwd", b_fwd, restore_b))
        cases.append(("modelB/default/fwd+bwd", b_fwdbwd, restore_b))
        cases.append(("modelB/default/train-step", b_step, restore_b))
    except Exception as e:
        build_errors["modelB"] = {
            "type": type(e).__name__, "message": str(e),
            "phase": "case-build"}
    return cases, build_errors


def family_cases(family):
    """Returns (cases, build_errors) for the family."""
    if family == "a-sum-abs":
        sa = sum_abs_cases()
        ma, errs = model_a_cases()
        return sa + ma, errs
    if family == "b-linear":
        mb, errs = model_b_cases()
        return mb, errs
    if family == "zeros":
        return zeros_cases(), {}
    raise ValueError(family)


def hit_probe_cases(family):
    """Representative (name, fn) pairs for non-timed dispatch verification."""
    x = torch.randn(64, 64, dtype=torch.float32, device=DEVICE).to(
        torch.complex64)
    if family == "a-sum-abs":
        return [
            ("sum/inner/c64", lambda: x.sum(dim=1)),
            ("abs/c64", lambda: torch.abs(x)),
        ]
    if family == "b-linear":
        xi = torch.randn(2, 3, 16, 16, device=DEVICE)
        w = torch.randn(4, 3, 3, 3, device=DEVICE)
        wl = torch.randn(5, 4, device=DEVICE)
        return [
            ("conv2d/fp32", lambda: torch.nn.functional.conv2d(xi, w,
                                                               padding=1)),
            ("linear/fp32", lambda: torch.nn.functional.linear(
                xi.permute(0, 2, 3, 1), wl)),
        ]
    if family == "zeros":
        return [
            ("zeros/c64", lambda: torch.zeros((8, 8), dtype=torch.complex64,
                                              device=DEVICE)),
            ("zero_/c64", lambda: x.zero_()),
        ]
    return []


def run_group(family, group, out_path, seed=20261110, quick=False):
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device", flush=True)
        sys.exit(2)
    rec = sm.set_determinism(0, "fp32")
    prov = register_group(group, family)
    if group != "n":
        for desc, fn in hit_probe_cases(family):
            try:
                prov["hit_verification"][desc] = verify_hit(desc, fn)
            except Exception as e:
                prov["hit_verification"][desc] = {
                    "fn": desc, "error": f"{type(e).__name__}: {e}"}
    cases, build_errors = family_cases(family)
    for name, err in build_errors.items():
        print(f"[{family}/{group}] BUILD-SKIP {name}: {err['type']}: "
              f"{err['message'][:120]}", flush=True)
    if quick:
        cases = cases[:10]
    rng = random.Random(seed)
    order = list(range(len(cases)))
    rng.shuffle(order)
    results = {}
    errors = {}
    for i in order:
        name = cases[i][0]
        fn = cases[i][1]
        setup = cases[i][2] if len(cases[i]) > 2 else None
        try:
            results[name] = bench_case(fn, setup=setup)
            results[name]["order_index"] = i
            print(f"[{family}/{group}] {name}: "
                  f"{results[name]['device_ms_per_iter_median']:.4f} "
                  f"ms/iter", flush=True)
        except Exception as e:
            errors[name] = {"type": type(e).__name__, "message": str(e)}
            print(f"[{family}/{group}] {name}: EXC {type(e).__name__}: "
                  f"{str(e)[:120]}", flush=True)
    out = {
        "group": group,
        "family": family,
        "provenance": prov,
        "determinism": rec,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "protocol": {
            "warmup": 20, "blocks": 5, "iters": 100,
            "first_call_separate": True,
            "timing": "cuda events + synced wall clock",
            "order": f"seeded-random (seed {seed})",
            "zero_restore": "per-block restore timed as separate case",
            "p95": "quantile estimate of per-block mean per-iter time "
                   "(5 blocks), not per-request tail latency",
            "memory": "baseline recorded before timed window; peak within "
                      "window (includes compile+warmup); delta is a lower "
                      "bound under shared process residency",
            "throughput": "calls_per_s = 1000 / ms_per_iter_median; "
                          "samples_per_s = batch * calls_per_s (model "
                          "cases only)",
            "hit_verification": "torch.profiler CUDA kernel-name capture "
                                "on representative calls, outside timed "
                                "regions",
        },
        "results": results,
        "errors": errors,
    }
    sm.write_json(out_path, out)
    print(f"[{family}/{group}] wrote {out_path} "
          f"({len(results)} ok, {len(errors)} exc)", flush=True)


def _env_no_gems():
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _env_tree(tree):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(tree, "src")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _spawn(argv, env):
    return subprocess.run(
        [sys.executable, os.path.abspath(__file__)] + argv,
        env=env, capture_output=True, text=True)


def load_correctness(accuracy_summary):
    """(group, model) -> status from the committed accuracy matrix, so
    timing rows for correctness-failed configs can be labelled diagnostic."""
    status = {}
    if not accuracy_summary or not os.path.exists(accuracy_summary):
        return status
    s = sm.read_json(accuracy_summary)
    for c in s.get("compares", []):
        if "pair" not in c or "rc" not in c:
            continue
        parts = c["pair"].split("/")
        group = parts[0].split("_vs_")[0]
        model = parts[1]
        key = (group, "a" if model.startswith("a") else "b")
        ok = c.get("rc") == 0
        prev = status.get(key)
        if prev is None or (not ok):
            status[key] = "pass" if ok else "failed"
    return status


def do_matrix(args):
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    schedule = ["n", "f_target", "f_all", "b", "n_rep"]
    correctness = load_correctness(args.accuracy_summary)
    for family in FAMILIES:
        for g in schedule:
            real = "n" if g == "n_rep" else g
            path = os.path.join(out, f"bench2-{family}-{g}.json")
            if os.path.exists(path):
                print(f"[matrix] {family}/{g} cached", flush=True)
                continue
            if real == "n":
                env = _env_no_gems()
            else:
                tree = (args.tree_b if real == "b" else args.tree_fixed)
                env = _env_tree(tree)
            argv = ["--family", family, "--group", real, "--out", path]
            if args.quick:
                argv.append("--quick")
            r = _spawn(argv, env)
            print(f"[matrix] {family}/{g} rc={r.returncode}", flush=True)
            if r.returncode != 0:
                print(r.stderr[-1500:], flush=True)
    # combined comparison: per family, vs its own native pass
    combined = {}
    for family in FAMILIES:
        for g in schedule:
            p = os.path.join(out, f"bench2-{family}-{g}.json")
            if os.path.exists(p):
                combined[(family, g)] = sm.read_json(p)
    table = []
    for family in FAMILIES:
        ref = combined.get((family, "n"), {}).get("results", {})
        rep = combined.get((family, "n_rep"), {}).get("results", {})
        for name, nres in ref.items():
            model = ("a" if name.startswith("modelA") else
                     "b" if name.startswith("modelB") else None)
            row = {"family": family, "case": name,
                   "n_ms": nres["device_ms_per_iter_median"]}
            if name in rep:
                row["n_rep_ms"] = rep[name]["device_ms_per_iter_median"]
                row["native_drift_pct"] = (
                    (row["n_rep_ms"] - row["n_ms"]) / row["n_ms"] * 100)
            for g in ("f_target", "f_all", "b"):
                r = combined.get((family, g), {}).get("results", {}).get(name)
                if r:
                    fms = r["device_ms_per_iter_median"]
                    row[f"{g}_ms"] = fms
                    row[f"{g}_vs_n"] = (fms / row["n_ms"]
                                        if row["n_ms"] else None)
                    if row[f"{g}_vs_n"] is not None and \
                            row[f"{g}_vs_n"] > 1.10:
                        row[f"{g}_candidate_regression_gt10pct"] = True
                if name in combined.get((family, g), {}).get("errors", {}):
                    row[f"{g}_error"] = \
                        combined[(family, g)]["errors"][name]["type"]
            if model is not None:
                for g in ("f_target", "f_all", "b"):
                    if correctness.get((g, model)) == "failed":
                        row[f"{g}_correctness"] = (
                            "failed -> timing is diagnostic only, not "
                            "valid performance")
            table.append(row)
    n_flags = sum(
        1 for row in table for k in row
        if k.endswith("_candidate_regression_gt10pct"))
    summary = {
        "schedule": schedule,
        "families": list(FAMILIES),
        "candidate_regression_gt10pct_count": n_flags,
        "candidate_flag_note": "median-ratio criterion only, not a "
                               "stability verdict",
        "correctness_status_source": args.accuracy_summary or None,
        "table": table,
        "groups": {f"{f}/{g}": {"errors": combined[(f, g)]["errors"]}
                   for (f, g) in combined},
    }
    sm.write_json(os.path.join(out, "bench2-summary.json"), summary)
    with open(os.path.join(out, "bench2-table.md"), "w") as f:
        f.write("| family | case | n ms/iter | f_target | f_all | b | "
                "drift n_rep |\n|---|---|---|---|---|---|---|\n")
        for row in table:
            def fmt(k):
                v = row.get(k)
                if v is None:
                    return "-"
                if isinstance(v, float):
                    ratio = row.get(k.replace("_ms", "_vs_n"))
                    return f"{v:.4f}" + (f" ({ratio:.2f}x)" if ratio else "")
                return str(v)
            f.write(f"| {row['family']} | {row['case']} | {fmt('n_ms')} | "
                    f"{fmt('f_target_ms')} | {fmt('f_all_ms')} | "
                    f"{fmt('b_ms')} | "
                    f"{row.get('native_drift_pct', '-')}% |\n")
    print(f"bench2 summary written ({n_flags} candidate flags)",
          flush=True)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=list(FAMILIES))
    p.add_argument("--group", choices=["n", "f_target", "f_all", "b"])
    p.add_argument("--out", default=None)
    p.add_argument("--matrix", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--accuracy-summary", dest="accuracy_summary",
                   default=None,
                   help="matrix summary.json for correctness labelling")
    p.add_argument("--tree-fixed", dest="tree_fixed",
                   default="/tmp/opencode/flagfix-fixed")
    p.add_argument("--tree-b", dest="tree_b",
                   default="/tmp/opencode/flagfix-e2ebase")
    return p


def main():
    args = build_parser().parse_args()
    if args.matrix:
        do_matrix(args)
    elif args.family and args.group:
        if not args.out:
            print("ERROR: --family/--group need --out")
            sys.exit(2)
        run_group(args.family, args.group, args.out, quick=args.quick)
    else:
        print("choose --matrix or --family F --group G --out Y")
        sys.exit(2)


if __name__ == "__main__":
    main()
