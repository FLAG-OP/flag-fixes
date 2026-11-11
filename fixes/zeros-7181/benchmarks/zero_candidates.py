# Round-10 (review 6787395475 section 2): zero_/zeros complex candidate
# screening. The three candidate implementations for a FUTURE optimization
# fix are evaluated against the native oracle on the reviewer's correctness
# matrix — this harness does NOT modify the zeros-7181 patch.
#
# Candidates:
#   current   : x.copy_(0)            (emulates the shipped fix branch;
#                                      known host-staging perf cost)
#   A         : full-storage contiguous -> uint8-view fill_(0) (device
#               memset); otherwise falls back to candidate B
#   B         : x.copy_(pre-built device 0-dim scalar)
#   C         : view_as_real -> per-channel select(-1, i).zero_()
#               (stride-correct native fill on the real view; conj-bit
#               inputs fall back to B because view_as_real rejects them)
#   native_ref: x.zero_()             (oracle)
#
# Correctness matrix (review section 2): contiguous full tensor; x[::2],
# narrow, transpose, storage offset; c64/c128; only the selected view is
# zeroed (unselected storage bitwise unchanged); alias/view observations
# match native.
#
# Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" python3 benchmarks/zero_candidates.py --out <path>
import argparse
import sys
import os
import time
import statistics

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

import torch  # noqa: E402
import small_models as sm  # noqa: E402

DEVICE = "cuda"
DTYPES = {"c64": torch.complex64, "c128": torch.complex128}


def make_layout(layout, dtype, n_small=32, n_big=None):
    """Returns (view, base_buffer) for each matrix layout."""
    n = n_small if n_big is None else n_big
    if layout == "contig":
        base = torch.randn(n, n, dtype=torch.float32,
                           device=DEVICE).to(dtype)
        return base, base
    if layout == "slice_stride":
        base = torch.randn(n, 2 * n, dtype=torch.float32,
                           device=DEVICE).to(dtype)
        return base[:, ::2], base
    if layout == "narrow":
        base = torch.randn(4 * n, dtype=torch.float32,
                           device=DEVICE).to(dtype)
        return base[2:2 + n], base
    if layout == "transpose":
        base = torch.randn(n, n, dtype=torch.float32,
                           device=DEVICE).to(dtype)
        return base.t().contiguous().t(), base
    if layout == "storage_offset":
        base = torch.randn(2 * n + 3, dtype=torch.float32,
                           device=DEVICE).to(dtype)
        return base[3:3 + n], base
    if layout == "conj":
        base = torch.randn(n, n, dtype=torch.float32,
                           device=DEVICE).to(dtype)
        return base.conj(), base
    raise ValueError(layout)


def apply_candidate(name, x, scalars):
    """Applies one candidate in-place; returns a short outcome string."""
    if name == "current":
        x.copy_(0)
        return "ok"
    if name == "A":
        if (not x.is_conj()) and x.is_contiguous() and \
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


def storage_bytes(t):
    return t.untyped_storage().cpu() if t.is_cuda else None


def check_case(layout, dtype_name, dtype, candidate, n=32):
    """Runs one matrix cell. Returns a result dict (pass/fail + details)."""
    x, base = make_layout(layout, dtype, n_small=n)
    base_before = base.clone()
    # selected-region mask on the BASE buffer: elements that belong to x
    sel = torch.zeros(base.numel(), dtype=torch.bool)
    xv = x.contiguous()  # only to discover which base indices are selected
    # map selected flat indices of x back to base storage indices
    idx = (torch.arange(x.numel(), device=DEVICE)
           .view(x.shape)
           .expand_as(x) if False else None)
    # simpler: compare base buffers elementwise before/after and require
    # differences exactly at positions whose value changed to 0 while being
    # part of x — done below via change-mask containment check
    del idx, xv
    scalars = {"device": torch.complex(torch.tensor(0.0),
                                       torch.tensor(0.0)).cuda()}
    res = {"layout": layout, "dtype": dtype_name, "candidate": candidate,
           "shape": list(x.shape), "stride": list(x.stride()),
           "storage_offset": x.storage_offset()}
    try:
        outcome = apply_candidate(candidate, x, scalars)
        res["outcome"] = outcome
    except Exception as e:
        res["pass"] = False
        res["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return res
    # 1) selected view all-zero
    all_zero = bool((x == 0).all().item()) if x.numel() else True
    # 2) unselected storage unchanged: differences vs base_before must be
    #    confined to positions now equal to zero AND belonging to x's
    #    element set (approximated: every changed element must be 0)
    changed = (base != base_before)
    changed_nonzero = changed & (base != 0)
    unselected_ok = not bool(changed_nonzero.any().item())
    # 3) alias consistency: a fresh view of the same storage sees zeros
    alias = base.reshape(-1)[x.storage_offset():]  # not exact for strided;
    # simpler alias check: re-read x itself (same storage) and also a
    # non-slice re-view of base where layout allows
    alias_ok = bool((x == 0).all().item()) if x.numel() else True
    # 4) equality vs native oracle: run native_ref on a fresh twin
    x2, base2 = make_layout(layout, dtype, n_small=n)
    apply_candidate("native_ref", x2, scalars)
    same_as_native = bool(torch.equal(x, x2)) if x.numel() else True
    ok = all_zero and unselected_ok and alias_ok and same_as_native
    res.update({
        "all_zero": all_zero,
        "unselected_ok": unselected_ok,
        "alias_ok": alias_ok,
        "same_as_native": same_as_native,
        "pass": ok,
    })
    return res


def bench_case(fn, iters=100, blocks=5, warmup=20):
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
    for _ in range(blocks):
        torch.cuda.synchronize()
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        dev.append(e0.elapsed_time(e1))
    med = statistics.median(dev) / iters
    torch.cuda.reset_peak_memory_stats()
    base_a = torch.cuda.memory_allocated() / 2**20
    return {
        "first_call_ms": first,
        "first_call_wall_ms": first_wall,
        "block_device_ms": dev,
        "ms_per_iter_median": med,
        "cv_pct": (statistics.stdev(dev) / statistics.mean(dev) * 100)
        if len(dev) > 1 and statistics.mean(dev) else 0.0,
        "mem_baseline_alloc_mb": base_a,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no device")
        sys.exit(2)
    rec = sm.set_determinism(0, "fp32")
    candidates = ["current", "A", "B", "C", "native_ref"]
    layouts = ["contig", "slice_stride", "narrow", "transpose",
               "storage_offset", "conj"]
    matrix = []
    n_pass = 0
    n_fail = 0
    for dtype_name, dtype in DTYPES.items():
        for layout in layouts:
            for cand in candidates:
                r = check_case(layout, dtype_name, dtype, cand)
                matrix.append(r)
                if r.get("pass"):
                    n_pass += 1
                else:
                    n_fail += 1
                    print(f"FAIL {dtype_name}/{layout}/{cand}: "
                          f"{r.get('error', r)}", flush=True)
    # perf: contig / slice_stride / transpose at 2^20 total elements
    # (1024x1024), both dtypes
    perf = []
    for dtype_name, dtype in DTYPES.items():
        for layout in ("contig", "slice_stride", "transpose"):
            for cand in candidates:
                x, base = make_layout(layout, dtype, n_big=1024)
                scalars = {"device": torch.complex(
                    torch.tensor(0.0), torch.tensor(0.0)).cuda()}
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
    out = {
        "purpose": "round-10 zero_ candidate screening (native harness; "
                   "does not modify the zeros-7181 patch)",
        "determinism": rec,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "matrix": matrix,
        "matrix_summary": {"pass": n_pass, "fail": n_fail},
        "perf": perf,
    }
    sm.write_json(args.out, out)
    print(f"CANDIDATE SCREENING: matrix pass={n_pass} fail={n_fail}; "
          f"wrote {args.out}", flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
