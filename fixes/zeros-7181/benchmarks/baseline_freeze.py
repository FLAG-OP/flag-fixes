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
    """Full-byte reliable hash + int64 byte-sum (NaN-safe; conj bits
    materialized; 0-dim handled; int64 sum — uint8 sum wraps under
    FlagGems registration, FIX-ITEM-4). Round-12: full sha256 replaces
    the truncated-64-hex-digit checksum that could hide post-byte-32
    differences."""
    t = t.resolve_conj() if t.is_conj() else t
    u8 = t.contiguous().reshape(-1).view(torch.uint8).cpu()
    import hashlib
    h = hashlib.sha256(u8.numpy().tobytes()).hexdigest()
    return h, int(u8.sum(dtype=torch.int64).item())


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
            # round-12 fix: the forward MUST use the grad-enabled clone
            # (previously used x, so backward had no grad_fn -> spurious
            # exceptions in 10 committed edge rows)
            xg = x.clone().requires_grad_(True)
            y = torch.abs(xg)
        res["out_checksum"], res["out_byte_sum"] = checksum(
            y.detach() if not isinstance(y, torch.Tensor) else y)
        if res["grad_applicable"]:
            y.backward(torch.ones_like(y))
            res["grad_checksum"], res["grad_byte_sum"] = checksum(xg.grad)
    except Exception as e:
        res.update({"status": "exception",
                    "error": f"{type(e).__name__}: {str(e)[:140]}"})
        return res
    res["status"] = "ok"
    res["shape"] = list(y.shape)
    # grad applicability is EXPLICIT: abs carries an input-grad checkpoint,
    # sum does not (round-12: missing fields may not stand in for a pass)
    res["grad_applicable"] = (op == "abs" and payload != "empty"
                              and not x.is_conj())
    if not res["grad_applicable"]:
        res["grad"] = "N/A"
    return res



# ------------------------------------------------------------- comparator --
# Round-12: strict case comparator. Categories are explicit; exceptions and
# missing fields are NEVER counted as numeric agreement.

def compare_case(fc, nc):
    """fc/nc: per-case result dicts (fix, native). Returns
    {case, category, detail} with category one of:
      numeric_equal | numeric_diff | both_exception | one_exception |
      missing_reference | missing_fields | dtype_or_shape_mismatch
    Rules (round-12):
      - numeric comparison requires BOTH status==ok AND all applicable
        checksum fields present in BOTH dicts;
      - grad_checksum is applicable only when grad_applicable is True;
        otherwise it is explicit N/A and ignored;
      - exceptions form their own category (both_exception/one_exception);
      - missing native reference is its own category (never skipped
        silently)."""
    case = (fc or nc).get("case")
    out = {"case": case}
    if nc is None:
        out.update({"category": "missing_reference",
                    "detail": "no native reference case"})
        return out
    fs, ns = fc.get("status"), nc.get("status")
    if fs == "exception" and ns == "exception":
        out.update({"category": "both_exception",
                    "detail": {"fix": fc.get("error", "")[:80],
                               "native": nc.get("error", "")[:80]}})
        return out
    if fs != "ok" or ns != "ok":
        out.update({"category": "one_exception",
                    "detail": f"fix={fs} native={ns}"})
        return out
    # applicable checksum fields
    fields = [k for k in ("x_checksum", "out_checksum")
              if k in fc or k in nc]
    if fc.get("grad_applicable") or nc.get("grad_applicable"):
        fields.append("grad_checksum")
    missing = [k for k in fields if k not in fc or k not in nc]
    if missing:
        out.update({"category": "missing_fields",
                    "detail": f"missing in {'fix/native':s}: {missing}"})
        return out
    for k in fields:
        if fc[k] != nc[k]:
            out.update({"category": "numeric_diff",
                        "detail": f"{k} differs (full-byte sha256)"})
            return out
    # dtype must match; shape compared when both recorded
    if fc.get("dtype") != nc.get("dtype"):
        out.update({"category": "dtype_or_shape_mismatch",
                    "detail": f"dtype {fc.get('dtype')} vs "
                              f"{nc.get('dtype')}"})
        return out
    if "shape" in fc and "shape" in nc and fc["shape"] != nc["shape"]:
        out.update({"category": "dtype_or_shape_mismatch",
                    "detail": f"shape {fc.get('shape')} vs "
                              f"{nc.get('shape')}"})
        return out
    out.update({"category": "numeric_equal",
                "detail": "full-byte sha256 + dtype/shape match"})
    return out


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



def do_selftest(out_path):
    """Device-free comparator negative tests (round-12): both-exception,
    missing fields, differences beyond byte 32, dtype mismatch, missing
    reference must NEVER be reported as numeric agreement."""
    import hashlib
    def H(b):
        return hashlib.sha256(bytes(b)).hexdigest()
    good = H(range(64))
    def case(status="ok", fields=None):
        c = {"case": "t", "status": status, "dtype": "c64"}
        if status == "ok":
            c.setdefault("x_checksum", good)
        if fields:
            c.update(fields)
        return c
    tests = []
    def t(name, fc, nc, want_cat):
        r = compare_case(fc, nc)
        ok = r["category"] == want_cat
        tests.append({"name": name, "want": want_cat,
                      "got": r["category"], "ok": ok})
        print(f"{'OK ' if ok else 'BAD'} {name}: {r['category']}")

    t("numeric_equal", case(), case(), "numeric_equal")
    # difference ONLY after byte 32 must be numeric_diff (old checksum hid it)
    t("diff_after_byte32",
      case(fields={"x_checksum": H([0] * 32 + [1] + [0] * 31)}),
      case(), "numeric_diff")
    t("dtype_mismatch",
      case(fields={"dtype": "c128"}), case(), "dtype_or_shape_mismatch")
    both_exc = case(status="exception", fields={"error": "e"})
    t("both_exception_not_equal", both_exc,
      case(status="exception", fields={"error": "e"}),
      "both_exception")
    t("one_exception", case(status="exception",
                            fields={"error": "e"}), case(),
      "one_exception")
    t("missing_grad_field",
      case(fields={"grad_applicable": True}),
      case(fields={"grad_applicable": True, "grad_checksum": good}),
      "missing_fields")
    t("missing_reference", case(), None, "missing_reference")
    # explicit-N/A grad must NOT be required
    t("grad_na_not_required",
      case(fields={"grad_applicable": False}),
      case(fields={"grad_applicable": False}), "numeric_equal")
    all_ok = all(x["ok"] for x in tests)
    out = {"all_ok": all_ok, "tests": tests}
    if out_path:
        sm.write_json(out_path, out)
    print(f"COMPARATOR SELFTEST: {'PASS' if all_ok else 'FAIL'} "
          f"({len(tests)} cases)", flush=True)
    return all_ok


def do_selftest_checked():
    return do_selftest("/tmp/opencode/baseline-selftest.json")


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
    # cross-group comparison: fix vs native via the strict comparator
    # (round-12: exceptions/missing fields are explicit categories and
    # never count as numeric agreement; checksums are full-byte sha256)
    compare = []
    nmap = {c["case"]: c for c in groups.get("n", {}).get("zero_cases", [])}
    for c in groups.get("fix", {}).get("zero_cases", []):
        compare.append(compare_case(c, nmap.get(c["case"])))
    emap = {c["case"]: c for c in groups.get("n", {}).get("edge_numeric",
                                                          [])}
    for c in groups.get("fix", {}).get("edge_numeric", []):
        compare.append(compare_case(c, emap.get(c["case"])))
    cats = {}
    for c in compare:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
    n_equal = cats.get("numeric_equal", 0)
    summary = {
        "purpose": "round-10 baseline freeze (zero path layouts/edges "
                   "across native/fix/prefix + sum/abs numeric edges) — "
                   "round-12 strict comparator",
        "groups": {g: v.get("provenance") for g, v in groups.items()},
        "compare_fix_vs_native": compare,
        "category_counts": cats,
        "numeric_equal_count": n_equal,
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
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tree-fixed", dest="tree_fixed",
                    default="/tmp/opencode/flagfix-fixed")
    ap.add_argument("--tree-b", dest="tree_b",
                   default="/tmp/opencode/flagfix-e2ebase")
    return ap


def main():
    args = build_parser().parse_args()
    if args.selftest:
        do_selftest(args.out or "/tmp/opencode/baseline-selftest.json")
    elif args.matrix:
        # comparator negatives must pass before any cross-group compare
        do_selftest("/tmp/opencode/baseline-selftest.json") if False else None
        st = do_selftest_checked()
        if not st:
            print("comparator selftest failed; aborting matrix")
            sys.exit(1)
        do_matrix(args)
    elif args.group:
        run_group(args.group, args.out)
    else:
        print("choose --matrix --out-dir or --group G --out P")
        sys.exit(2)


if __name__ == "__main__":
    main()
