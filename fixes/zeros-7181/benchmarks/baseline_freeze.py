# Round-10 (review 6787395475 section 1): freeze the current baseline for
# the zero_/zeros complex path — layouts (contig / slice / narrow /
# transpose / storage offset), c64/c128, edge inputs (conj / empty /
# 0-dim; plus sum/abs numeric edges NaN/Inf/extreme), across three paths
# (native / current fix / pre-fix control). One process per group; the
# driver combines and cross-checks.
#
# Per case records: status (ok/exception), selected-zero + unselected-
# storage self-checks, output storage checksum (bitwise), timing for
# perf-eligible cases, and profiler kernel names for a representative call.
#
# Run (one group):  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<tree>/src \
#                     python3 benchmarks/baseline_freeze.py --group fix --out <path>
# Driver:           python3 benchmarks/baseline_freeze.py --matrix --out-dir <dir>
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

import torch  # noqa: E402
import small_models as sm  # noqa: E402

DEVICE = "cuda"
DTYPES = {"c64": torch.complex64, "c128": torch.complex128}
LAYOUTS = ("contig", "slice_stride", "narrow", "transpose", "offset")
PERF_LAYOUTS = ("contig", "slice_stride", "transpose")


def make_view(layout, dtype, n=1024):
    """Complex tensors are ASSEMBLED via torch.complex (cast via .to()
    would hit the pre-existing to_copy/fill gap under registered groups —
    see FIX-ITEM-2)."""
    def mk(shape):
        p = torch.randn(shape, dtype=torch.float32, device=DEVICE)
        q = torch.randn(shape, dtype=torch.float32, device=DEVICE)
        return torch.complex(p, q)
    if layout == "contig":
        base = mk((n, n))
        return base, base
    if layout == "slice_stride":
        base = mk((n, 2 * n))
        return base[:, ::2], base
    if layout == "narrow":
        base = mk((4 * n,))
        return base[2:2 + n], base
    if layout == "transpose":
        base = mk((n, n))
        return base.t(), base
    if layout == "offset":
        base = mk((2 * n + 5,))
        return base[5:5 + n], base
    raise ValueError(layout)


def checksum(t):
    """Bitwise checksum of the tensor's storage (NaN-safe; conj bits
    materialized; 0-dim handled via reshape(-1))."""
    t = t.resolve_conj() if t.is_conj() else t
    u8 = t.contiguous().reshape(-1).view(torch.uint8)
    return u8.cpu().numpy().tobytes().hex()[:64], int(
        u8.sum(dtype=torch.int64).item())  # int64: uint8 sum wraps under FlagGems registration (FIX-ITEM-4)


def kernel_capture(fn):
    from torch.profiler import profile, ProfilerActivity
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evs = []
    for ev in prof.key_averages():
        dev = getattr(ev, "self_device_time_total", 0) or 0
        if dev > 0:
            evs.append(ev.key[:110])
    return evs[:6]


def zero_cases():
    cases = []
    for dname, dt in DTYPES.items():
        for layout in LAYOUTS:
            cases.append((f"zero_/{dname}/{layout}", "zero_", dt, layout))
            cases.append((f"zeros/{dname}/{layout}", "zeros", dt, layout))
    # edges: conj view, empty, 0-dim
    for dname, dt in DTYPES.items():
        cases.append((f"zero_/conj/{dname}", "zero_", dt, "conj"))
        cases.append((f"zeros/conj/{dname}", "zeros", dt, "conj"))
        cases.append((f"zero_/empty/{dname}", "zero_", dt, "empty"))
        cases.append((f"zeros/empty/{dname}", "zeros", dt, "empty"))
        cases.append((f"zero_/zerodim/{dname}", "zero_", dt, "zerodim"))
    return cases


def run_zero_case(op, dt, layout_kind):
    """Returns result dict for one zero case in this group. All tensor
    creation stays inside the guarded region (registered groups can crash
    on complex creation — that is recorded, never fatal)."""
    res = {"op": op, "layout": layout_kind, "dtype": str(dt)}
    try:
        if layout_kind == "conj":
            p = torch.randn(16, 16, dtype=torch.float32, device=DEVICE)
            q = torch.randn(16, 16, dtype=torch.float32, device=DEVICE)
            base = torch.complex(p, q)
            x = base.conj()
        elif layout_kind == "empty":
            x = torch.zeros(0, dtype=dt, device=DEVICE)
            base = x
        elif layout_kind == "zerodim":
            p = torch.full((), 1.0, dtype=torch.float32, device=DEVICE)
            q = torch.full((), 0.5, dtype=torch.float32, device=DEVICE)
            x = torch.complex(p, q)
            base = x
        else:
            x, base = make_view(layout_kind, dt, n=256)
        base_before = base.clone()
        target = x if op == "zero_" else None
        if op == "zero_":
            x.zero_()
        else:
            # zeros creation: the created tensor is the object under test
            target = torch.zeros(x.shape, dtype=dt, device=DEVICE)
    except Exception as e:
        res.update({"status": "exception",
                    "error": f"{type(e).__name__}: {str(e)[:140]}"})
        return res
    res["status"] = "ok"
    res["outcome_target"] = "input-view" if op == "zero_" else "created"
    try:
        if target.numel():
            res["all_zero"] = bool((target == 0).all().item())
        else:
            res["all_zero"] = True
        if op == "zero_" and x.numel():
            changed = (base != base_before)
            res["unselected_ok"] = not bool(
                (changed & (base != 0)).any().item())
        res["x_checksum"], res["x_byte_sum"] = checksum(target)
    except Exception as e:  # belt-and-braces: self-check escapes
        res.update({"status": "post-op-error",
                    "error": f"{type(e).__name__}: {str(e)[:140]}"})
    return res


def edge_numeric_cases():
    """sum/abs numeric edges (NaN/Inf/extreme/empty/0-dim/conj)."""
    cases = []
    payloads = {
        "nan": float("nan"), "inf": float("inf"),
        "neginf": float("-inf"),
        "extreme_high": 1e38, "extreme_low": 1e-38,
    }
    for dname, dt in DTYPES.items():
        for pname, pv in payloads.items():
            for op in ("sum", "abs"):
                cases.append((f"{op}/{dname}/{pname}", op, dt, pv))
        cases.append((f"sum/{dname}/empty", "sum", dt, "empty"))
        cases.append((f"abs/{dname}/empty", "abs", dt, "empty"))
        cases.append((f"sum/{dname}/conj", "sum", dt, "conj"))
        cases.append((f"abs/{dname}/conj", "abs", dt, "conj"))
    return cases


def run_edge_case(op, dt, payload):
    res = {"op": op, "payload": str(payload), "dtype": str(dt)}
    try:
        if payload == "empty":
            x = torch.zeros(0, dtype=dt, device=DEVICE)
        elif payload == "conj":
            p = torch.randn(8, 8, dtype=torch.float32, device=DEVICE)
            q = torch.randn(8, 8, dtype=torch.float32, device=DEVICE)
            x = torch.complex(p, q).conj()
        else:
            v = payload if isinstance(payload, float) else float(payload)
            p = torch.full((16, 16), v, dtype=torch.float32,
                           device=DEVICE)
            q = torch.full((16, 16), v * 0.5, dtype=torch.float32,
                           device=DEVICE)
            x = torch.complex(p, q)
        if op == "sum":
            y = x.sum()
        else:
            xg = x.clone().requires_grad_(True) if not x.is_conj() \
                else x
            y = torch.abs(xg if x.is_conj() else x)
        res["out_checksum"], res["out_byte_sum"] = checksum(
            y.detach() if not isinstance(y, torch.Tensor) else y)
        if op == "abs" and not payload == "empty" and not x.is_conj():
            y.backward(torch.ones_like(y))
            res["grad_checksum"], res["grad_byte_sum"] = checksum(xg.grad)
    except Exception as e:
        res.update({"status": "exception",
                    "error": f"{type(e).__name__}: {str(e)[:140]}"})
        return res
    res["status"] = "ok"
    return res


def run_group(group, out_path):
    if not torch.cuda.is_available():
        print("no device")
        sys.exit(2)
    rec = sm.set_determinism(0, "fp32")
    prov = {"group": group, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0)}
    if group != "n":
        import flag_gems
        if group == "fix":
            flag_gems.only_enable(include=["zeros", "zero_", "zeros_like",
                                           "sum", "sum_dim", "sum_out",
                                           "sum_dim_out", "abs", "absolute"])
            prov["include"] = ["zeros", "zero_", "zeros_like", "sum",
                               "sum_dim", "sum_out", "sum_dim_out", "abs",
                               "absolute"]
        else:  # prefix: full enable on pre-fix tree
            flag_gems.enable()
        prov["flag_gems_path"] = flag_gems.__file__
        prov["registered_keys_sample"] = sorted(
            set(flag_gems.all_registered_keys()))[:12]
    zero_res = []
    for (name, op, dt, layout) in zero_cases():
        r = run_zero_case(op, dt, layout)
        r["case"] = name
        zero_res.append(r)
        print(f"[{group}] {name}: {r.get('status', '?')} "
              f"{r.get('error', '')[:60]}", flush=True)
    edge_res = []
    for (name, op, dt, payload) in edge_numeric_cases():
        r = run_edge_case(op, dt, payload)
        r["case"] = name
        edge_res.append(r)
        print(f"[{group}] {name}: {r.get('status', '?')} "
              f"{r.get('error', '')[:60]}", flush=True)
    # perf-eligible: zero_ contig/slice_stride/transpose at full n
    # (crashes on the pre-fix tree are recorded per-entry, not fatal)
    perf = []
    for dname, dt in DTYPES.items():
        for layout in PERF_LAYOUTS:
            x, _ = make_view(layout, dt, n=1024)
            try:
                t = bench_zero(lambda x=x: x.zero_())
                perf.append({"op": "zero_", "dtype": dname,
                             "layout": layout, **t})
                print(f"[{group}] PERF zero_/{dname}/{layout}: "
                      f"{t['ms_per_iter_median']:.4f}", flush=True)
            except Exception as e:
                perf.append({"op": "zero_", "dtype": dname,
                             "layout": layout,
                             "error": f"{type(e).__name__}: "
                                      f"{str(e)[:100]}"})
                print(f"[{group}] PERF zero_/{dname}/{layout}: EXC "
                      f"{type(e).__name__}", flush=True)
    out = {"group": group, "provenance": prov, "determinism": rec,
           "zero_cases": zero_res, "edge_numeric": edge_res,
           "perf": perf}
    sm.write_json(out_path, out)
    print(f"[{group}] wrote {out_path}", flush=True)


def bench_zero(fn, iters=100, blocks=5, warmup=20):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    w0 = time.perf_counter()
    e0.record()
    fn()
    e1.record()
    torch.cuda.synchronize()
    first = e0.elapsed_time(e1)
    first_wall = (time.perf_counter() - w0) * 1e3
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dev = []
    torch.cuda.reset_peak_memory_stats()
    base_a = torch.cuda.memory_allocated() / 2**20
    for _ in range(blocks):
        torch.cuda.synchronize()
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        dev.append(e0.elapsed_time(e1))
    med = statistics.median(dev) / iters
    return {"first_call_ms": first, "first_call_wall_ms": first_wall,
            "block_device_ms": dev, "ms_per_iter_median": med,
            "mem_baseline_alloc_mb": base_a}


def _env(group, tree_fixed, tree_b):
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if group == "n":
        env["PYTHONPATH"] = ""
    elif group == "fix":
        env["PYTHONPATH"] = os.path.join(tree_fixed, "src")
    else:
        env["PYTHONPATH"] = os.path.join(tree_b, "src")
    return env


def do_matrix(args):
    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)
    for g in ("n", "fix", "prefix"):
        p = os.path.join(out, f"baseline-{g}.json")
        if os.path.exists(p):
            print(f"[matrix] {g} cached")
            continue
        env = _env(g, args.tree_fixed, args.tree_b)
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--group", g,
             "--out", p], env=env, capture_output=True, text=True)
        print(f"[matrix] {g} rc={r.returncode}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-1200:], flush=True)
    groups = {}
    for g in ("n", "fix", "prefix"):
        p = os.path.join(out, f"baseline-{g}.json")
        if os.path.exists(p):
            groups[g] = sm.read_json(p)
    # cross-group numeric comparison: fix vs native on zero cases + edges
    compare = []
    nmap = {c["case"]: c for c in groups.get("n", {}).get("zero_cases", [])}
    for c in groups.get("fix", {}).get("zero_cases", []):
        nc = nmap.get(c["case"])
        if nc is None:
            continue
        same = (c.get("status") == nc.get("status") and
                c.get("x_checksum") == nc.get("x_checksum"))
        compare.append({"case": c["case"], "fix_status": c.get("status"),
                        "native_status": nc.get("status"),
                        "bitwise_equal": same})
    emap = {c["case"]: c for c in groups.get("n", {}).get("edge_numeric",
                                                          [])}
    for c in groups.get("fix", {}).get("edge_numeric", []):
        nc = emap.get(c["case"])
        if nc is None:
            continue
        keys = ("out_checksum", "grad_checksum")
        same = all(c.get(k) == nc.get(k) for k in keys) and \
            c.get("status") == nc.get("status")
        compare.append({"case": c["case"], "fix_status": c.get("status"),
                        "native_status": nc.get("status"),
                        "bitwise_equal": same})
    n_equal = sum(1 for c in compare if c.get("bitwise_equal"))
    summary = {
        "purpose": "round-10 baseline freeze (zero path layouts/edges "
                   "across native/fix/prefix + sum/abs numeric edges)",
        "groups": {g: v.get("provenance") for g, v in groups.items()},
        "compare_fix_vs_native": compare,
        "bitwise_equal_count": n_equal,
        "compare_count": len(compare),
        "prefix_crashes": [
            {"group": g, "case": c["case"],
             "error": c.get("error", "")[:120]}
            for g in ("prefix",) for c in
            groups.get(g, {}).get("zero_cases", [])
            + groups.get(g, {}).get("edge_numeric", [])
            if c.get("status") == "exception"],
    }
    sm.write_json(os.path.join(out, "baseline-summary.json"), summary)
    with open(os.path.join(out, "baseline-table.md"), "w") as f:
        f.write("| case | fix | native | bitwise_equal |\n|---|---|---|---|\n")
        for c in compare:
            f.write(f"| {c['case']} | {c['fix_status']} | "
                    f"{c['native_status']} | {c['bitwise_equal']} |\n")
    print(json.dumps({"bitwise_equal": n_equal, "total": len(compare),
                      "prefix_crashes": len(summary["prefix_crashes"])},
                     indent=1))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["n", "fix", "prefix"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--out-dir", dest="out_dir", default=None)
    ap.add_argument("--tree-fixed", dest="tree_fixed",
                    default="/tmp/opencode/flagfix-fixed")
    ap.add_argument("--tree-b", dest="tree_b",
                   default="/tmp/opencode/flagfix-e2ebase")
    return ap


def main():
    args = build_parser().parse_args()
    if args.matrix:
        do_matrix(args)
    elif args.group:
        run_group(args.group, args.out)
    else:
        print("choose --matrix --out-dir or --group G --out P")
        sys.exit(2)


if __name__ == "__main__":
    main()
