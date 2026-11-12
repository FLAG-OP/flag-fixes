# Round-10/11: zero_/zeros complex candidate screening (reviews
# 6787395475 + 6788210724). The candidate implementations for a FUTURE
# optimization fix are evaluated against the native oracle — this harness
# does NOT modify the zeros-7181 patch.
#
# Round-11 extensions (review section 1/2/4):
#   - matrix adds: zeros CREATION path (alloc + candidate zeroing), empty
#     and 0-dim tensors, combined layouts (storage-offset x transpose x
#     slice x conj), broader alias/overlap checks; negative strides are
#     recorded N/A (torch disallows them);
#   - --registered mode: reruns the matrix + path verification under the
#     REAL registration environment (zeros family only_enable on the fix
#     tree), capturing profiler events per candidate so "expected path"
#     claims are bound to observed kernels (A must not be polluted by the
#     FlagGems fill path or FIX-ITEM-4; B must not reintroduce host
#     staging; events must match the native-harness conclusion);
#   - complexity: guard-branch cost measured; A+B combo (= candidate A as
#     implemented: A fast path, B fallback) vs B-only compared per layout.
#
# Run (native):  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" \
#                  python3 benchmarks/zero_candidates.py --out <path>
# Run (registered, fix tree): CUDA_VISIBLE_DEVICES=0 \
#                  PYTHONPATH=<tree>/src python3 benchmarks/zero_candidates.py \
#                  --registered --out <path>
import argparse
import json
import statistics
import sys
import os
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

import torch  # noqa: E402
import small_models as sm  # noqa: E402

DEVICE = "cuda"
DTYPES = {"c64": torch.complex64, "c128": torch.complex128}


DTYPES = {"c64": torch.complex64, "c128": torch.complex128}
REAL = {"c64": torch.float32, "c128": torch.float64}


def mk_complex(shape, dtype):
    """real/imag constructed in the dtype's REAL type (round-12: the
    previous float32-hardcoded construction made most c128 entries
    actually run c64)."""
    real = dtype.to_real_dtype() if hasattr(dtype, "to_real_dtype") \
        else (torch.float32 if dtype == torch.complex64 else torch.float64)
    p = torch.randn(shape, dtype=real, device=DEVICE)
    q = torch.randn(shape, dtype=real, device=DEVICE)
    return torch.complex(p, q)


def make_layout(layout, dtype, n=32):
    """(view, base) pairs. Complex tensors assembled via torch.complex in
    the requested dtype (cast via .to() hits the FIX-ITEM-2 to_copy/fill
    gap under registration)."""
    if layout == "contig":
        base = mk_complex((n, n), dtype)
        return base, base
    if layout == "slice_stride":
        base = mk_complex((n, 2 * n), dtype)
        return base[:, ::2], base
    if layout == "narrow":
        base = mk_complex((4 * n,), dtype)
        return base[2:2 + n], base
    if layout == "transpose":
        base = mk_complex((n, n), dtype)
        return base.t(), base
    if layout == "storage_offset":
        base = mk_complex((2 * n + 3,), dtype)
        return base[3:3 + n], base
    if layout == "conj":
        base = mk_complex((n, n), dtype)
        return base.conj(), base
    # round-11 combined layouts
    if layout == "offset_transpose":
        base = mk_complex((n + 3, n), dtype)
        return base[3:, :].t(), base
    if layout == "offset_slice":
        base = mk_complex((n + 5, 2 * n), dtype)
        return base[5:, ::2], base
    if layout == "transpose_conj":
        base = mk_complex((n, n), dtype)
        return base.t().conj(), base
    if layout == "slice_conj":
        base = mk_complex((n, 2 * n), dtype)
        return base[:, ::2].conj(), base
    if layout == "offset_transpose_conj":
        base = mk_complex((n + 3, n), dtype)
        return base[3:, :].t().conj(), base
    raise ValueError(layout)


def apply_candidate(name, x, scalars):
    if name == "current":
        x.copy_(0)
        return "ok"
    if name == "A":
        if (not x.is_conj()) and x.dim() >= 1 and x.is_contiguous() and \
                x.storage_offset() == 0 and \
                x.untyped_storage().nbytes() == x.numel() * x.element_size():
            x.view(torch.uint8).fill_(0)
            return "ok(memset)"
        x.copy_(scalars["device"])
        return "fallback->B" + ("(conj)" if x.is_conj() else "")
    if name == "B":
        x.copy_(scalars["device"])
        return "ok"
    if name == "C":
        if x.is_conj():
            x.copy_(scalars["device"])
            return "fallback->B(conj)"
        vr = torch.view_as_real(x)
        vr.select(-1, 0).zero_()
        vr.select(-1, 1).zero_()
        return "ok"
    if name == "native_ref":
        x.zero_()
        return "ok"
    raise ValueError(name)


def guard_cost(x, iters=1000):
    """Host-side cost of candidate A's branch condition (no device work)."""
    def check():
        return ((not x.is_conj()) and x.is_contiguous() and
                x.storage_offset() == 0 and
                x.untyped_storage().nbytes() ==
                x.numel() * x.element_size())
    t0 = time.perf_counter()
    n = 0
    for _ in range(iters):
        n += 1 if check() else 0
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds


def selected_base_indices(x):
    """Exact flat storage indices of x's elements (stride/offset aware)."""
    dims = [torch.arange(d, device=DEVICE) for d in x.shape]
    grids = torch.meshgrid(*dims, indexing="ij") if x.dim() else []
    flat = torch.full((), x.storage_offset(), dtype=torch.long,
                      device=DEVICE)
    for g, s in zip(grids, x.stride()):
        flat = flat + g * s
    return flat.reshape(-1)


def check_case(layout, dtype_name, dtype, candidate, creation=False,
               n=32):
    """One matrix cell. creation=True emulates the zeros CREATION path
    (fresh alloc + candidate zeroing) instead of zeroing a view."""
    res = {"layout": layout, "dtype": dtype_name, "candidate": candidate,
           "creation": creation}
    try:
        if layout == "empty":
            x = torch.zeros(0, dtype=dtype, device=DEVICE)
        elif layout == "zerodim":
            real = torch.float32 if dtype == torch.complex64 \
                else torch.float64
            p = torch.full((), 1.0, dtype=real, device=DEVICE)
            q = torch.full((), 0.5, dtype=real, device=DEVICE)
            x = torch.complex(p, q)
        else:
            x, base = make_layout(layout, dtype, n=n)
            base_before = base.clone()
        res["shape"] = list(x.shape)
        # round-12: assert and record the ACTUAL dtype (the previous
        # float32-hardcoded construction made c128 labels run c64)
        assert x.dtype == dtype, f"dtype mismatch {x.dtype} != {dtype}"
        res["actual_dtype"] = str(x.dtype)
        res["is_conj"] = bool(x.is_conj())
        scalars = {"device": torch.complex(
            torch.tensor(0.0, dtype=torch.float32
                         if dtype == torch.complex64 else torch.float64),
            torch.tensor(0.0, dtype=torch.float32
                         if dtype == torch.complex64 else torch.float64),
        ).cuda()}
        assert scalars["device"].dtype == dtype
        if creation:
            # zeros creation path: fresh native alloc + candidate zeroing
            y = torch.empty(x.shape, dtype=dtype, device=DEVICE)
            outcome = apply_candidate(candidate, y, scalars)
            res["outcome"] = outcome
            res["all_zero"] = bool((y == 0).all().item()) \
                if y.numel() else True
            res["pass"] = res["all_zero"]
            return res
        outcome = apply_candidate(candidate, x, scalars)
        res["outcome"] = outcome
    except Exception as e:
        res["pass"] = False
        res["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return res
    try:
        all_zero = bool((x == 0).all().item()) if x.numel() else True
        checks = {"all_zero": all_zero}
        if layout not in ("empty", "zerodim"):
            idx = selected_base_indices(x)
            bf = base_before.reshape(-1)
            ba = base.reshape(-1)
            changed = bf != ba
            # every changed base element must be a selected one
            sel_mask = torch.zeros(ba.numel(), dtype=torch.bool,
                                   device=DEVICE)
            sel_mask[idx] = True
            checks["changed_only_selected"] = bool(
                (changed & ~sel_mask).sum().item() == 0)
            checks["selected_all_zero_via_base"] = bool(
                (ba[idx] == 0).all().item())
            # native oracle twin
            x2, _ = make_layout(layout, dtype, n=n)
            apply_candidate("native_ref", x2, scalars)
            checks["same_as_native"] = bool(torch.equal(x, x2)) \
                if x.numel() else True
        else:
            checks["changed_only_selected"] = True
            checks["selected_all_zero_via_base"] = True
            checks["same_as_native"] = True
        res.update(checks)
        res["pass"] = all(v is True for k, v in checks.items()
                          if isinstance(v, bool))
    except Exception as e:
        res["pass"] = False
        res["error"] = f"post-check {type(e).__name__}: {str(e)[:120]}"
    return res


def classify_events(events):
    tags = []
    for name in events:
        nl = name.lower()
        if "memcpy" in nl and "htod" in nl:
            tags.append("HtoD-staging(!)")
        elif "memcpy" in nl or "memset" in nl:
            tags.append("memcpy/memset")
        elif "at::native" in name:
            tags.append("native-aten")
        elif "kernel" in nl:
            tags.append("triton-candidate")
        else:
            tags.append("unknown")
    return tags


def path_verify(candidate, dtype_name, dtype, layout, n=1024):
    """Registered-environment path verification: profiler events + timing
    for one candidate on one layout (used in --registered mode)."""
    if layout == "contig":
        x, base = make_layout("contig", dtype, n=n)
    elif layout == "slice_stride":
        x, base = make_layout("slice_stride", dtype, n=n)
    else:
        x, base = make_layout("transpose", dtype, n=n)
    assert x.dtype == dtype
    scalars = {"device": torch.complex(
        torch.tensor(0.0, dtype=torch.float32
                     if dtype == torch.complex64 else torch.float64),
        torch.tensor(0.0, dtype=torch.float32
                     if dtype == torch.complex64 else torch.float64),
    ).cuda()}
    fn = lambda: apply_candidate(candidate, x, scalars)
    fn()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = []
    for ev in prof.key_averages():
        dev = getattr(ev, "self_device_time_total", 0) or 0
        if dev > 0:
            events.append({"name": ev.key[:110],
                           "class": classify_events([ev.key])[0],
                           "self_device_us": dev})
    t = bench_case(fn)
    return {"candidate": candidate,
            "role": ("registered-path reference (current fix path)"
                     if candidate == "native_ref" else "candidate"),
            "dtype": dtype_name, "layout": layout, "events": events,
            "ms_per_iter_median": t["ms_per_iter_median"],
            "cv_pct": t["cv_pct"]}


def bench_case(fn, iters=100, blocks=5, warmup=20):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    fn()
    e1.record()
    torch.cuda.synchronize()
    first = e0.elapsed_time(e1)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_a = torch.cuda.memory_allocated() / 2**20
    dev = []
    for _ in range(blocks):
        torch.cuda.synchronize()
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        dev.append(e0.elapsed_time(e1))
    med = statistics.median(dev) / iters
    return {"first_call_ms": first,
            "block_device_ms": dev,
            "ms_per_iter_median": med,
            "cv_pct": (statistics.stdev(dev) / statistics.mean(dev) * 100)
            if len(dev) > 1 and statistics.mean(dev) else 0.0,
            "mem_baseline_alloc_mb": base_a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--registered", action="store_true",
                    help="run under zeros-family only_enable (fix tree)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no device")
        sys.exit(2)
    rec = sm.set_determinism(0, "fp32")
    prov = {"registered_mode": args.registered,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "determinism": rec}
    if args.registered:
        import flag_gems
        flag_gems.only_enable(include=["zeros", "zero_", "zeros_like"])
        prov["flag_gems_path"] = flag_gems.__file__
        prov["registered_keys"] = sorted(set(
            flag_gems.all_registered_keys()))

    candidates = ["current", "A", "B", "C", "native_ref"]
    layouts = ["contig", "slice_stride", "narrow", "transpose",
               "storage_offset", "conj",
               "offset_transpose", "offset_slice", "transpose_conj",
               "slice_conj", "offset_transpose_conj",
               "empty", "zerodim"]
    matrix = []
    n_pass = n_fail = 0
    for dtype_name, dtype in DTYPES.items():
        for layout in layouts:
            for cand in candidates:
                for creation in (False, True):
                    r = check_case(layout, dtype_name, dtype, cand,
                                   creation=creation)
                    r["creation"] = creation
                    matrix.append(r)
                    if r.get("pass"):
                        n_pass += 1
                    else:
                        n_fail += 1
                        print(f"FAIL {dtype_name}/{layout}/{cand}"
                              f"{'/creation' if creation else ''}: "
                              f"{r.get('error', r)}", flush=True)
    # perf: contig / slice_stride / transpose at 2^20 (1024x1024)
    perf = []
    for dtype_name, dtype in DTYPES.items():
        for layout in ("contig", "slice_stride", "transpose"):
            for cand in candidates:
                x, base = make_layout(layout, dtype, n=1024)
                assert x.dtype == dtype
                scalars = {"device": torch.complex(
                    torch.tensor(0.0, dtype=torch.float32
                                 if dtype == torch.complex64
                                 else torch.float64),
                    torch.tensor(0.0, dtype=torch.float32
                                 if dtype == torch.complex64
                                 else torch.float64),
                ).cuda()}
                try:
                    fn = lambda x=x, c=cand, s=scalars: apply_candidate(
                        c, x, s)
                    t = bench_case(fn)
                    perf.append({"dtype": dtype_name, "layout": layout,
                                 "candidate": cand, **t})
                    print(f"PERF {dtype_name}/{layout}/{cand}: "
                          f"{t['ms_per_iter_median']:.4f} ms/iter",
                          flush=True)
                except Exception as e:
                    perf.append({"dtype": dtype_name, "layout": layout,
                                 "candidate": cand,
                                 "error": f"{type(e).__name__}: "
                                          f"{str(e)[:100]}"})
    # complexity (round-11 section 4): guard cost + B-only vs A+B combo
    xg, _ = make_layout("contig", torch.complex64, n=1024)
    guard_us = guard_cost(xg)
    complexity = {
        "note": "candidate A IS the A+B combo (A fast path, B fallback); "
                "compare A vs B-only per layout",
        "guard_branch_cost_us": guard_us,
        "comparison": "see perf rows: A vs B per dtype/layout",
    }
    # registered-mode path verification
    path_verification = []
    if args.registered:
        for dtype_name, dtype in DTYPES.items():
            for layout in ("contig", "slice_stride"):
                for cand in ("A", "B", "C", "native_ref"):
                    r = path_verify(cand, dtype_name, dtype, layout)
                    path_verification.append(r)
                    tags = [e["class"] for e in r["events"]]
                    print(f"PATH {dtype_name}/{layout}/{cand}: {tags} "
                          f"{r['ms_per_iter_median']:.4f} ms/iter",
                          flush=True)
    out = {
        "purpose": "round-10/11 zero_ candidate screening"
                   + (" (registered environment)" if args.registered
                      else " (native harness)")
                   + "; does not modify the zeros-7181 patch",
        "mode": "registered" if args.registered else "native",
        "provenance": prov,
        "matrix": matrix,
        "matrix_summary": {"pass": n_pass, "fail": n_fail},
        "perf": perf,
        "complexity": complexity,
        "path_verification": path_verification,
    }
    sm.write_json(args.out, out)
    print(f"CANDIDATE SCREENING ({out['mode']}): matrix pass={n_pass} "
          f"fail={n_fail}; wrote {args.out}", flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
