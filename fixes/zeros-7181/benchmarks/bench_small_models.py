# Round-4 performance benchmarks (review plan 2026-11-10, section 4).
#
# Process-per-group like the accuracy runner (registration is process
# global): the driver --matrix spawns n / f_target / f_all / b / n_rep with
# the right PYTHONPATH. n_rep is a second native pass used to bound thermal/
# drift effects; case order inside a process is seeded-random.
#
# Protocol: first call timed separately (JIT/autotune, cache state noted),
# warmup 20, 5 independent blocks x 100 iters, CUDA events for device time
# plus sync wall clock per block, median / P95 / per-iter stats / samples/s,
# peak allocated+reserved memory per case, zero_ restore measured separately.
# First round is reporting only: no pre-committed speedup; stable >10%
# regressions are flagged for explanation.
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


def register_group(group, model="a"):
    prov = {"group": group, "flag_gems_path": None, "include": None,
            "registered_ops": [], "registered_keys": []}
    if group == "n":
        return prov
    import flag_gems
    if group == "f_target":
        inc = (["sum", "sum_dim", "sum_out", "sum_dim_out", "abs",
                "absolute"] if model == "a" else
               ["conv2d", "linear", "linear_backward"])
        flag_gems.only_enable(include=inc)
        prov["include"] = inc
    else:
        flag_gems.enable()
    prov["flag_gems_path"] = flag_gems.__file__
    try:
        prov["registered_ops"] = sorted(set(flag_gems.all_registered_ops()))
        prov["registered_keys"] = sorted(set(flag_gems.all_registered_keys()))
    except Exception as e:
        prov["record_error"] = f"{type(e).__name__}: {e}"
    return prov


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


def _record_peak():
    return {
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
    }


def bench_case(fn, setup=None, warmup=20, blocks=5, iters=100):
    """Returns dict with first-call, per-block device ms, wall ms, stats."""
    res = {"warmup": warmup, "blocks": blocks, "iters_per_block": iters}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
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
                + ("cold-process" if warmup == 0 else "warmed-after"),
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
    res["device_ms_per_iter_p95"] = sorted(
        statistics.quantiles(dev_ms, n=20)[-1:] or [max(dev_ms)])[0] / iters \
        if len(dev_ms) >= 2 else max(dev_ms) / iters
    res["wall_ms_per_iter_median"] = statistics.median(wall_ms) / iters
    res["block_device_ms_cv_pct"] = (
        (statistics.stdev(dev_ms) / statistics.mean(dev_ms) * 100)
        if len(dev_ms) > 1 and statistics.mean(dev_ms) else 0.0)
    res["samples_per_s"] = iters / (med / 1e3) if med else None
    res["timing"] = "cuda events device time + synced wall clock"
    res.update(_record_peak())
    return res


def op_cases():
    """Op-level cases: sum full/inner/outer, abs, zeros/zero_, fwd+bwd.

    Built per-dtype-block: under full registration any complex tensor
    creation can crash (pre-existing triton complex64 gap), so each dtype
    block is isolated and a setup failure is recorded instead of killing
    the whole group run."""
    cases, build_errors = [], {}
    for dname, dt in CDTYPES.items():
        try:
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
                # separate leaf tensors for backward cases (forward-only
                # timings must not pay autograd overhead)
                xb = x2.detach().clone().requires_grad_(True)
                xbs = x2s.detach().clone().requires_grad_(True)
                g1 = torch.randn(r, dtype=dt, device=DEVICE)
                gc = torch.randn(r, c, dtype=real_dt, device=DEVICE)

                cases.append((f"sum/full/{tag}",
                              lambda x=x1: x.sum()))
                cases.append((f"sum/inner/{tag}",
                              lambda x=x2: x.sum(dim=1)))
                cases.append((f"sum/outer/{tag}",
                              lambda x=x2: x.sum(dim=0)))
                cases.append((f"sum/inner/strided/{tag}",
                              lambda x=x2s: x.sum(dim=1)))
                cases.append((f"abs/{tag}", lambda x=x2: torch.abs(x)))
                cases.append((f"abs/strided/{tag}",
                              lambda x=x2s: torch.abs(x)))

                # fwd+bwd chains (graph built per iteration); abs maps
                # complex->real so its upstream grad is real-dtyped
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

                # zeros creation and zero_ (restore measured separately)
                cases.append((f"zeros/{tag}",
                              lambda s=(r, c), d=dt: torch.zeros(
                                  s, dtype=d, device=DEVICE)))
                src = x2.clone()
                xz = torch.empty_like(x2)
                cases.append((f"zero_/{tag}",
                              lambda t=xz: t.zero_(),
                              lambda t=xz, s=src: t.copy_(s)))
        except Exception as e:
            build_errors[f"opblock/{dname}"] = {
                "type": type(e).__name__, "message": str(e),
                "phase": "case-build",
            }
    return cases, build_errors


def model_cases():
    """Model-level cases: A (c64 core) and B (fp32) at medium sizes.

    Built per-model: the complex-core model A can fail to even construct
    under full registration (complex gap); the all-real model B stays
    measurable, which is exactly the comparison the review asks for."""
    cases, build_errors = [], {}
    lr = 1e-2
    try:
        # Model A medium: B=8, L=256, D=64, H=64, out=8 (A_CFGS["large"])
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
                sm.set_target(tg)  # per-model target (shared module global)
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
            "phase": "case-build",
        }
    try:
        # Model B medium: N=8, Cin=3, H=W=32, Cout=32 (B_CFGS["large"])
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
            sm.set_target(tg)  # per-model target (shared module global)
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
            "phase": "case-build",
        }
    return cases, build_errors


def run_group(group, out_path, seed=20261110, quick=False):
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device", flush=True)
        sys.exit(2)
    rec = sm.set_determinism(0, "fp32")
    model = "a"
    prov = register_group(group, model)
    op_c, op_err = op_cases()
    md_c, md_err = model_cases()
    cases = op_c + md_c
    build_errors = {**op_err, **md_err}
    for name, err in build_errors.items():
        print(f"[{group}] BUILD-SKIP {name}: {err['type']}: "
              f"{err['message'][:120]}", flush=True)
    if quick:
        cases = [c for c in cases if "/n2^20/" not in c[0]][:12]
    rng = random.Random(seed)
    order = list(range(len(cases)))
    rng.shuffle(order)
    results = {}
    errors = dict(build_errors)
    for i in order:
        name = cases[i][0]
        fn = cases[i][1]
        setup = cases[i][2] if len(cases[i]) > 2 else None
        try:
            results[name] = bench_case(fn, setup=setup)
            results[name]["order_index"] = i
            print(f"[{group}] {name}: "
                  f"{results[name]['device_ms_per_iter_median']:.4f} ms/iter",
                  flush=True)
        except Exception as e:
            errors[name] = {"type": type(e).__name__, "message": str(e)}
            print(f"[{group}] {name}: EXC {type(e).__name__}: {e}",
                  flush=True)
    out = {
        "group": group,
        "provenance": prov,
        "determinism": rec,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "protocol": {
            "warmup": 20, "blocks": 5, "iters": 100,
            "first_call_separate": True,
            "timing": "cuda events + synced wall clock",
            "order": "seeded-random (seed 20261110)",
            "zero_restore": "per-block restore timed as separate case",
        },
        "results": results,
        "errors": errors,
    }
    sm.write_json(out_path, out)
    print(f"[{group}] wrote {out_path} "
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


def do_matrix(args):
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    schedule = ["n", "f_target", "f_all", "b", "n_rep"]
    for g in schedule:
        real = "n" if g == "n_rep" else g
        path = os.path.join(out, f"bench-{g}.json")
        if os.path.exists(path):
            print(f"[matrix] {g} cached", flush=True)
            continue
        if real == "n":
            env = _env_no_gems()
        else:
            tree = (args.tree_b if real == "b" else args.tree_fixed)
            env = _env_tree(tree)
        argv = ["--group", real, "--out", path]
        if args.quick:
            argv.append("--quick")
        r = _spawn(argv, env)
        print(f"[matrix] {g} rc={r.returncode}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-2000:], flush=True)
    # combined comparison table
    combined = {}
    for g in schedule:
        p = os.path.join(out, f"bench-{g}.json")
        if os.path.exists(p):
            combined[g] = sm.read_json(p)
    table = []
    ref = combined.get("n", {}).get("results", {})
    rep = combined.get("n_rep", {}).get("results", {})
    for name, nres in ref.items():
        row = {"case": name, "n_ms": nres["device_ms_per_iter_median"]}
        if name in rep:
            row["n_rep_ms"] = rep[name]["device_ms_per_iter_median"]
            row["native_drift_pct"] = (
                (row["n_rep_ms"] - row["n_ms"]) / row["n_ms"] * 100)
        for g in ("f_target", "f_all", "b"):
            r = combined.get(g, {}).get("results", {}).get(name)
            if r:
                fms = r["device_ms_per_iter_median"]
                row[f"{g}_ms"] = fms
                row[f"{g}_vs_n"] = fms / row["n_ms"] if row["n_ms"] else None
                if row[f"{g}_vs_n"] is not None and row[f"{g}_vs_n"] > 1.10:
                    row[f"{g}_flag"] = "regression>10%"
        if name in combined.get("b", {}).get("errors", {}):
            row["b_error"] = combined["b"]["errors"][name]["type"]
        table.append(row)
    summary = {"schedule": schedule, "table": table,
               "groups": {g: {"errors": combined[g]["errors"]}
                          for g in combined}}
    sm.write_json(os.path.join(out, "bench-summary.json"), summary)
    with open(os.path.join(out, "bench-table.md"), "w") as f:
        f.write("| case | n ms/iter | f_target | f_all | b | drift n_rep |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in table:
            def fmt(k):
                v = row.get(k)
                if v is None:
                    return "-"
                if isinstance(v, float):
                    return f"{v:.4f}" + (
                        f" ({row.get(k.replace('_ms', '_vs_n')):.2f}x)"
                        if row.get(k.replace("_ms", "_vs_n")) else "")
                return str(v)
            f.write(f"| {row['case']} | {fmt('n_ms')} | "
                    f"{fmt('f_target_ms')} | {fmt('f_all_ms')} | "
                    f"{fmt('b_ms')} | {row.get('native_drift_pct', '-')}"
                    f"% |\n")
    print("bench summary written", flush=True)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["n", "f_target", "f_all", "b"])
    p.add_argument("--out", default=None)
    p.add_argument("--matrix", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--tree-fixed", dest="tree_fixed",
                   default="/tmp/opencode/flagfix-fixed")
    p.add_argument("--tree-b", dest="tree_b",
                   default="/tmp/opencode/flagfix-e2ebase")
    return p


def main():
    args = build_parser().parse_args()
    if args.matrix:
        do_matrix(args)
    elif args.group:
        if not args.out:
            print("ERROR: --group needs --out")
            sys.exit(2)
        run_group(args.group, args.out, quick=args.quick)
    else:
        print("choose --group X --out Y or --matrix")
        sys.exit(2)


if __name__ == "__main__":
    main()
