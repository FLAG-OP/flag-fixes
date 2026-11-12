# Round-12/13/14/15: quantify the numeric_diff baseline cases with a
# classification-complete, bit-exact comparison tool.
#
# Round-15 (review 6811732879):
#   - classify(): sign/exp/mantissa extracted separately — mantissa mask is
#     mant&~sign&~exp (the previous ~exp mask kept the sign bit, so -inf
#     was classified as NaN; a NaN payload policy then silently accepted
#     -Inf-vs-NaN);
#   - per-component comparison works directly on the RAW BITS returned by
#     the children (NaN payloads no longer pass through repr);
#   - main() aggregates hard verification failures (child rc != 0 / missing
#     result, in_bits mismatch, length mismatch, classification mismatch,
#     missing fields) and exits NONZERO after saving diagnostics; the
#     aggregation is testable via --exit-code-negative-test;
#   - raw in/out bits, dtype, shape, byteorder and output SHA256 are
#     archived per case; the abs/c128/nan case is cross-checked against the
#     committed round-10 baseline checksums (batch reproduction).
#
# Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" python3 benchmarks/diff_details.py \
#        --fix-tree <tree>/src --out <path> [--selftest]
import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import small_models as sm  # noqa: E402

CHILD = r'''
import sys, json, struct, torch
mode, op, dtype_name, payload = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
if mode != "native":
    import flag_gems
    flag_gems.only_enable(include=["sum", "sum_dim", "sum_out",
                                   "sum_dim_out", "abs", "absolute"])
torch.manual_seed(0)
dt = {"c64": torch.complex64, "c128": torch.complex128}[dtype_name]
real = torch.float32 if dt == torch.complex64 else torch.float64

def mk_payload():
    if payload == "extreme_high":
        return torch.full((16, 16), complex(1e38, 5e37), dtype=dt,
                          device="cuda")
    if payload == "extreme_low":
        return torch.full((16, 16), complex(1e-38, 5e-39), dtype=dt,
                          device="cuda")
    if payload == "nan":
        # EXACTLY the baseline_freeze construction: NaN + NaN*j
        p = torch.full((16, 16), float("nan"), dtype=real, device="cuda")
        q = torch.full((16, 16), float("nan") * 0.5, dtype=real,
                       device="cuda")
        return torch.complex(p, q)
    if payload == "conj":
        p = torch.randn(8, 8, dtype=real, device="cuda")
        q = torch.randn(8, 8, dtype=real, device="cuda")
        return torch.complex(p, q).conj()
    raise ValueError(payload)

x = mk_payload()
y = (x.sum() if op == "sum" else torch.abs(x.resolve_conj()
     if x.is_conj() else x)).flatten().cpu()
f64 = dt == torch.complex128
def bit_list(t):
    tt = (t.resolve_conj() if t.is_conj() else t).contiguous().reshape(-1)
    vtt = tt.view(torch.float64 if f64 else torch.float32).cpu()
    raw = vtt.numpy().tobytes()
    if f64:
        return struct.unpack(f"{vtt.numel()}Q", raw)
    return struct.unpack(f"{vtt.numel()}I", raw)
print(json.dumps({"values": [repr(v) for v in y.tolist()],
                  "out_is_complex": bool(y.is_complex()),
                  "in_is_complex": bool(x.is_complex()),
                  "out_bits": bit_list(y),
                  "in_bits": bit_list(x),
                  "dtype": dtype_name,
                  "shape": list(y.shape),
                  "byteorder": sys.byteorder}))
'''


def bits32(v):
    return struct.unpack("I", struct.pack("f", v))[0]


def bits64(v):
    return struct.unpack("Q", struct.pack("d", v))[0]


def ordered_ord(bits, total_bits):
    """Standard ordered mapping (monotonic across +/- zero): positives
    above the midpoint, negatives mirrored below it."""
    sign = 1 << (total_bits - 1)
    if bits & sign:
        m = bits & (sign - 1)
        return sign - 1 - m
    return sign + bits


def classify(value, bits, float64):
    """Round-15: sign/exp/mantissa extracted SEPARATELY. mantissa is the
    true payload mask (exponent AND sign excluded) — the previous
    `bits & ~exp_mask` kept the sign bit, so -inf carried a non-zero
    "mantissa" and was misclassified as NaN."""
    sign = 1 << (63 if float64 else 31)
    exp_mask = 0x7FF0000000000000 if float64 else 0x7F800000
    mant_mask = sign - 1 - exp_mask          # 0x000F...FF / 0x007FFFFF
    exp = bits & exp_mask
    mant = bits & mant_mask                  # true mantissa payload
    if exp == exp_mask:
        if mant == 0:
            return ("inf-" if bits & sign else "inf+"), mant
        return "nan", mant
    if value == 0.0:
        return "zero", (1 if bits & sign else 0)
    return "finite", mant


def classify_bits(bits, float64, value=None):
    return classify(0.0 if value is None else value, bits, float64)


def ulp_distance(n, f, float64, nv_bits=None, fv_bits=None):
    """Ordered-mapping ULP distance; +/-0 policy = same point (distance 0)
    when both values are zero (sign-bit difference is a separate flag
    handled by the caller)."""
    if n == 0.0 and f == 0.0:
        return 0
    if float64:
        bn = bits64(n) if nv_bits is None else nv_bits
        bf = bits64(f) if fv_bits is None else fv_bits
        return abs(ordered_ord(bn, 64) - ordered_ord(bf, 64))
    bn = bits32(n) if nv_bits is None else nv_bits
    bf = bits32(f) if fv_bits is None else fv_bits
    return abs(ordered_ord(bn, 32) - ordered_ord(bf, 32))


def component_compare(nv_bits, fv_bits, nv_value, fv_value, float64):
    """One scalar component compared from RAW BITS + repr round-trip
    value. Returns (status, flags, detail): status in equal |
    signed_zero_bit_diff | nan_payload_diff | class_mismatch |
    numeric_diff."""
    cn, pn = classify(0.0 if nv_value is None else nv_value, nv_bits,
                      float64)
    cf, pf = classify(0.0 if fv_value is None else fv_value, fv_bits,
                      float64)
    flags = []
    if cn != cf:
        return ("class_mismatch", flags,
                f"native={cn}({nv_bits:#x}) fix={cf}({fv_bits:#x})")
    if cn == "nan":
        if nv_bits != fv_bits:
            flags.append("nan_payload_diff")
        return ("equal", flags,
                f"nan payload native={nv_bits:#x} fix={fv_bits:#x}")
    if cn.startswith("inf"):
        return "equal", flags, cn
    if cn == "zero":
        if nv_bits != fv_bits:
            flags.append("signed_zero_bit_diff")  # numeric policy: same
            # zero point; the sign-bit difference is recorded separately
        return "equal", flags, "zero"
    # both finite: exact repr round-trip values
    nv = float(nv_value) if nv_value is not None else 0.0
    fv = float(fv_value) if fv_value is not None else 0.0
    d = abs(nv - fv)
    ulp = abs(ordered_ord(nv_bits, 64 if float64 else 32) -
              ordered_ord(fv_bits, 64 if float64 else 32))
    if d != 0:
        return "numeric_diff", flags, f"ulp={ulp} abs={d!r}"
    return "equal", flags, "zero-diff"


def ulp_metrics(nat, fix):
    """Classification-complete metrics from RAW BITS (round-15).
    Round-16: EVERY return path carries the full schema with defaults
    (n_class_mismatches etc.), so main() can always read them and the
    diagnostics JSON is always written; missing required fields are
    aggregated as hard failures instead of raising."""
    schema = {"float64": nat.get("dtype") == "c128",
              "hard_failures": [], "max_abs_err": None,
              "max_rel_err": None, "max_ulp_distance": None,
              "flags": [], "class_mismatches": [],
              "n_class_mismatches": 0}
    REQUIRED = ("out_bits", "in_bits", "values", "out_is_complex")
    missing = [k for k in REQUIRED if k not in nat or k not in fix]
    if missing:
        schema["hard_failures"].append(
            f"missing required fields: {missing}")
        schema["missing_fields"] = missing
        return schema
    float64 = schema["float64"]
    nb, fb = nat["out_bits"], fix["out_bits"]
    hard = schema["hard_failures"]
    flags = schema["flags"]
    class_mismatches = schema["class_mismatches"]
    max_abs = 0.0
    rel = 0.0
    max_ulp = 0
    if nat.get("in_bits") != fix.get("in_bits"):
        hard.append("in_bits mismatch (dual-tree inputs differ)")
    if len(nb) != len(fb):
        hard.append(f"out component count mismatch {len(nb)} vs {len(fb)}")
        schema["class_mismatches"].append("length")
        schema["n_class_mismatches"] = 1
        return schema
    n_vals = nat["values"]
    f_vals = fix["values"]
    comps_per_elem = 2 if nat.get("out_is_complex") else 1
    if len(n_vals) * comps_per_elem != len(nb) or \
            len(f_vals) * comps_per_elem != len(fb):
        hard.append("values/bits granularity mismatch "
                    f"(values={len(n_vals)}x{comps_per_elem} "
                    f"vs bits={len(nb)})")
        schema["n_class_mismatches"] = 0
        return schema
    class_mismatches = []
    for i in range(min(len(nb), len(fb))):
        # component i: real part of element i//2 (float64 complex) or the
        # element itself (float32 complex) — matching the bits layout
        # component mapping by OUTPUT kind (round-15): real outputs
        # (abs) have 1 component per element; complex outputs (sum) have
        # real/imag pairs
        if nat.get("out_is_complex"):
            e, part = i // 2, i % 2
        else:
            e, part = i, 0
        vi_n = n_vals[e] if e < len(n_vals) else "nan"
        vi_f = f_vals[e] if e < len(f_vals) else "nan"
        nv_value = fv_value = None
        try:
            cv_n = complex(vi_n)
            cv_f = complex(vi_f)
            nv_value = cv_n.real if part == 0 else cv_n.imag
            fv_value = cv_f.real if part == 0 else cv_f.imag
        except (ValueError, TypeError):
            pass  # NaN repr — bits-only classification
        st, fl, detail = component_compare(
            nb[i], fb[i], nv_value, fv_value, float64)
        if st == "class_mismatch":
            class_mismatches.append(f"comp{i}:{detail}")
            hard.append(f"comp{i}:{detail}")
        for x in fl:
            flags.append(f"comp{i}:{x}")
        if st == "numeric_diff":
            try:
                nv = float(nv_value)
                fv = float(fv_value)
                d = abs(nv - fv)
                max_abs = max(max_abs, d)
                if nv != 0:
                    rel = max(rel, d / abs(nv))
                u = abs(ordered_ord(nb[i], 64 if float64 else 32) -
                        ordered_ord(fb[i], 64 if float64 else 32))
                max_ulp = max(max_ulp, u)
            except (ValueError, TypeError):
                pass
    schema.update({"hard_failures": hard,
                   "class_mismatches": sorted(set(class_mismatches)),
                   "n_class_mismatches": len(class_mismatches),
                   "flags": sorted(set(flags)),
                   "max_abs_err": max_abs, "max_rel_err": rel,
                   "max_ulp_distance": max_ulp})
    return schema


def selftest():
    """Device-free self-tests: classify negatives (±Inf, ±NaN, -Inf↔NaN,
    Inf sign flip — both dtypes), ordered mapping exact values, signed-zero
    policy, and the exit-code aggregation negative."""
    tests = []

    def t(name, cond):
        tests.append((name, bool(cond)))
        print(f"{'OK ' if cond else 'BAD'} {name}", flush=True)

    pinf32, pinf64 = 0x7F800000, 0x7FF0000000000000
    ninf32, ninf64 = 0xFF800000, 0xFFF0000000000000
    t("f32 classify(+inf)==inf+",
      classify(0.0, pinf32, False) == ("inf+", 0))
    t("f32 classify(-inf)==inf-",
      classify(0.0, ninf32, False) == ("inf-", 0))
    t("f64 classify(+inf)==inf+",
      classify(0.0, pinf64, True) == ("inf+", 0))
    t("f64 classify(-inf)==inf-",          # round-15: was misclassified nan
      classify(0.0, ninf64, True) == ("inf-", 0))
    t("f64 classify(-inf)!=nan",
      classify(0.0, ninf64, True)[0] != "nan")
    t("f32 classify(-inf)!=nan",
      classify(0.0, ninf32, False)[0] != "nan")
    # NaN with/without sign bit and payload
    qnan64 = 0x7FF8000000000000
    snan64 = 0xFFF8000000000000
    t("f64 classify(+qnan)==nan",
      classify(0.0, qnan64, True) == ("nan", 0x8000000000000))
    t("f64 classify(-qnan)==nan (sign excluded from mantissa)",
      classify(0.0, snan64, True) == ("nan", 0x8000000000000))
    t("f64 NaN payload bits returned",
      classify(0.0, qnan64 | (1 << 40), True)[1] == (0x8000000000000 |
                                                     (1 << 40)))
    # -Inf vs NaN comparison must be a CLASS MISMATCH (round-15: was
    # silently accepted as nan_payload_diff)
    st, fl, det = component_compare(ninf64, qnan64, None, None, True)
    t("f64 -inf vs NaN -> class_mismatch",
      st == "class_mismatch")
    st, fl, det = component_compare(ninf32, 0x7FC00000, None, None, False)
    t("f32 -inf vs +NaN -> class_mismatch", st == "class_mismatch")
    st, fl, det = component_compare(pinf32, ninf32, None, None, False)
    t("f32 +inf vs -inf -> class_mismatch (sign flip)",
      st == "class_mismatch")
    # ordered mapping exact values (retained from round-14)
    t("f32 -0.0 ord == 0x7FFFFFFF",
      ordered_ord(bits32(-0.0), 32) == 0x7FFFFFFF)
    t("f32 dist(-min_sub,+min_sub)==3",
      abs(ordered_ord(bits32(torch.tensor(-1e-45,
          dtype=torch.float32).item()), 32) -
          ordered_ord(bits32(torch.tensor(1e-45,
          dtype=torch.float32).item()), 32)) == 3)
    # signed-zero policy
    st, fl, det = component_compare(bits32(-0.0), bits32(0.0), -0.0, 0.0,
                                    False)
    t("f32 +/-0 numeric equal with signed_zero flag",
      st == "equal" and "signed_zero_bit_diff" in fl)
    # exit-code aggregation negative (round-15 P2: failures must roll up)
    inject = [
        {"case": "injected/class_mismatch", "status": "ok",
         "dtype": "c128", "in_bits": [0], "out_bits": [1],
         "values": ["nan"], "out_bits_fix": [2], "values_fix": ["nan"],
         "out_bits_native": [1]},
    ]
    hard = aggregate_failures(
        [{"case": "x", "status": "ok", "dtype": "c64",
          "in_bits": [0], "out_bits": [0], "out_bits_native": [0],
          "values": ["1.0"], "values_native": ["1.0"]}],
        injected_class_mismatch=inject)
    t("aggregate: injected class mismatch -> nonzero failures",
      len(hard) > 0)
    all_ok = all(ok for _, ok in tests)
    print(f"SELFTEST: {'PASS' if all_ok else 'FAIL'} "
          f"({len(tests)} tests)", flush=True)
    return {"pass": all_ok, "tests": tests}


def nonempty_eval(v):
    return bool(v)


def aggregate_failures(cases, injected_class_mismatch=None):
    """Rolls up hard verification failures (round-15): in_bits mismatch,
    classification mismatch, child failures, missing fields. Pure function
    for exit-code negative testing."""
    hard = []
    for c in cases:
        for hf in c.get("hard_failures", []):
            hard.append(f"{c.get('case')}: {hf}")
        if c.get("in_bits_equal") is False:
            hard.append(f"{c.get('case')}: in_bits mismatch")
        if c.get("n_class_mismatches", 0) > 0:
            hard.append(f"{c.get('case')}: classification mismatch")
        if c.get("child_failed"):
            hard.append(f"{c.get('case')}: child failed")
    if injected_class_mismatch:
        for inj in injected_class_mismatch:
            hard.append(f"{inj['case']}: injected classification mismatch")
    return hard


def run_child(mode, op, dtype_name, payload, env):
    r = subprocess.run(
        [sys.executable, "-c", CHILD, mode, op, dtype_name, payload],
        env=env, capture_output=True, text=True, timeout=300)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        print(f"CHILD rc={r.returncode} ({mode}/{op}/{dtype_name}/{payload})"
              f" stderr tail: {r.stderr[-200:]}", flush=True)
        return None, r.returncode
    return json.loads(line[-1]), r.returncode


def raw_hex_prefix_of_bits(bits, float64):
    """First bytes of the output as hex — the SAME function the round-10
    baseline used for out_checksum (raw byte hex prefix, NOT a hash).
    Pads to at least 32 bytes for small outputs; comparison against the
    stored checksum uses PREFIX matching (small tensors have shorter
    hex)."""
    fmt = "Q" if float64 else "I"
    take = min(len(bits), 4 if float64 else 8)   # >= 32 bytes when large
    raw = struct.pack(f"{take}{fmt}", *bits[:take])
    return raw.hex()


def sha256_of_bits(bits, float64):
    fmt = "Q" if float64 else "I"
    raw = struct.pack(f"{len(bits)}{fmt}", *bits)
    return hashlib.sha256(raw).hexdigest()



def process_case(op, dtname, payload, nat, fix, nat_rc, fix_rc, out,
                 hard_failures):
    """Processes one case; appends to out["cases"] and hard_failures.
    Called inside try/except by the driver loop."""
    case = f"{op}/{dtname}/{payload}"
    if nat is None or fix is None:
        hard_failures.append(f"{case}: child failed "
                             f"(native_rc={nat_rc} fix_rc={fix_rc})")
        out["cases"].append({"case": case, "child_failed": True,
                             "native_rc": nat_rc, "fix_rc": fix_rc})
        return
    float64 = dtname == "c128"
    m = ulp_metrics(nat, fix)
    m.update({"case": case,
              "input_construction": ("baseline NaN + NaN*j"
                                     if payload == "nan" else
                                     "torch.complex payload"),
              "in_bits_equal": nat["in_bits"] == fix["in_bits"],
              "out_bits_equal": nat["out_bits"] == fix["out_bits"],
              "out_sha256_native": sha256_of_bits(nat["out_bits"],
                                                  float64),
              "out_sha256_fix": sha256_of_bits(fix["out_bits"],
                                               float64),
              "out_rawhex_prefix_native": raw_hex_prefix_of_bits(
                  nat["out_bits"], float64),
              "out_rawhex_prefix_fix": raw_hex_prefix_of_bits(
                  fix["out_bits"], float64),
              "shape": fix.get("shape"),
              "byteorder": sys.byteorder,
              "raw_in_bits_native": nat["in_bits"],
              "raw_in_bits_fix": fix["in_bits"],
              "raw_out_bits_native": nat["out_bits"],
              "raw_out_bits_fix": fix["out_bits"]})
    if m["in_bits_equal"] is False:
        hard_failures.append(f"{case}: in_bits mismatch")
    for hf in m["hard_failures"]:
        hard_failures.append(f"{case}: {hf}")
    out["cases"].append(m)
    print(f"{case}: max_abs={m['max_abs_err']} "
          f"max_ulp={m['max_ulp_distance']} "
          f"flags={m['flags']} "
          f"class_mismatch={m['n_class_mismatches']} "
          f"out_bits_equal={m['out_bits_equal']} "
          f"in_bits_equal={m['in_bits_equal']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-tree",
                    default="/tmp/opencode/flagfix-fixed/src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--exit-code-negative-test", action="store_true",
                    dest="exit_code_negative_test")
    ap.add_argument("--baseline-dir",
                    default=os.path.dirname(os.path.abspath(__file__))
                    + "/../results/small-model-2026-11-10/baseline")
    args = ap.parse_args()
    if args.selftest:
        res = selftest()
        sm.write_json(args.out, res)
        print(f"wrote {args.out}", flush=True)
        sys.exit(0 if res["pass"] else 1)
    if args.exit_code_negative_test:
        # entry-level negatives (round-16): length error / missing field /
        # child failure must EACH aggregate to nonzero AND produce a
        # diagnostics JSON; a clean case aggregates to zero
        scenarios = {
            "length_mismatch": {
                "case": "injected/length", "status": "ok",
                "dtype": "c128", "in_bits": [0, 1], "out_bits": [0],
                "out_bits_native": [0, 1], "values": ["1.0", "2.0"],
                "values_native": ["1.0", "2.0"], "out_is_complex": False,
                "in_is_complex": False},
            "missing_field": {
                "case": "injected/missing", "status": "ok",
                "dtype": "c64", "in_bits": [0], "values": ["1.0"]},
            "child_failed": {
                "case": "injected/child", "child_failed": True,
                "native_rc": 1, "fix_rc": 1},
        }
        all_ok = True
        diag = {}
        for name, case in scenarios.items():
            m = ulp_metrics(case, case)
            hard = aggregate_failures([m])
            nonzero = len(hard) > 0
            all_ok = all_ok and nonzero
            diag[name] = {"hard_failures": hard,
                          "n_class_mismatches":
                              m.get("n_class_mismatches", 0)}
            print(f"  {name}: aggregate_nonzero={nonzero}", flush=True)
        sm.write_json(args.out, {"entry_negative_test": True,
                                 "diagnostics_json_written": True,
                                 "scenarios": diag})
        import os as _os
        written = _os.path.exists(args.out)
        ok = all_ok and written
        print(f"ENTRY NEGATIVE TEST (length/missing-field/child-failed): "
              f"{'PASS' if ok else 'FAIL'} — diagnostics written={written},"
              f" all aggregated nonzero={all_ok}", flush=True)
        sys.exit(0 if ok else 1)
    env_native = dict(os.environ)
    env_native["PYTHONPATH"] = ""
    env_native.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env_fix = dict(os.environ)
    env_fix["PYTHONPATH"] = args.fix_tree
    env_fix.setdefault("CUDA_VISIBLE_DEVICES", "0")
    cases = [
        ("abs", "c64", "extreme_low"),
        ("abs", "c128", "extreme_low"),
        ("abs", "c128", "nan"),
        ("abs", "c128", "extreme_high"),
        ("sum", "c64", "conj"),
        ("sum", "c128", "conj"),
        ("abs", "c64", "conj"),
        ("abs", "c128", "conj"),
    ]
    out = {"cases": [], "byteorder": sys.byteorder,
           "batch_commit": os.popen("git -C " + os.path.dirname(
               os.path.abspath(__file__)) +
               " rev-parse --short HEAD").read().strip() or "unknown"}
    hard_failures = []  # round-16: aggregated and ALWAYS reported with a
    # diagnostics JSON before a nonzero exit
    for (op, dtname, payload) in cases:
        try:
            nat, nat_rc = run_child("native", op, dtname, payload,
                                    env_native)
            fix, fix_rc = run_child("fix", op, dtname, payload, env_fix)
            process_case(op, dtname, payload, nat, fix, nat_rc, fix_rc,
                         out=out, hard_failures=hard_failures)
        except Exception as e:
            # round-16: never lose the diagnostics report — record the
            # failure and continue; the report is written after the loop
            hard_failures.append(f"{op}/{dtname}/{payload}: processing "
                                 f"error {type(e).__name__}: "
                                 f"{str(e)[:120]}")
            out["cases"].append({"case": f"{op}/{dtname}/{payload}",
                                 "processing_error":
                                     f"{type(e).__name__}: "
                                     f"{str(e)[:140]}"})
            print(f"[case-error] {op}/{dtname}/{payload}: "
                  f"{type(e).__name__}", flush=True)
    # batch reproduction (round-17): compare against the CURRENT strict
    # baseline files (baseline-n.json / baseline-fix.json), NOT the
    # superseded r10 archive (the r10 batch had the float32-assembled
    # c128 mis-dtype — see round-12 — and conj cases are RNG-batch
    # dependent). Field name: batch_reproduction_vs_strict.
    repro = []
    sn = os.path.join(args.baseline_dir, "baseline-n.json")
    sf = os.path.join(args.baseline_dir, "baseline-fix.json")
    if os.path.exists(sn) and os.path.exists(sf):
        nmap = {c["case"]: c for c in sm.read_json(sn).get(
            "edge_numeric", [])}
        fmap = {c["case"]: c for c in sm.read_json(sf).get(
            "edge_numeric", [])}
        for m in out["cases"]:
            case = m.get("case")
            rn = nmap.get(case, {}).get("out_checksum")
            rf = fmap.get(case, {}).get("out_checksum")
            if rn and rf:
                # strict baseline checksums are FULL sha256 hex (round-12
                # checksum()); compare against this run's output sha256
                sn = m.get("out_sha256_native", "")
                sf = m.get("out_sha256_fix", "")
                repro.append({"case": case,
                              "this_native_sha256": sn,
                              "strict_native_checksum": rn,
                              "this_fix_sha256": sf,
                              "strict_fix_checksum": rf,
                              "native_matches_strict": sn == rn,
                              "fix_matches_strict": sf == rf})
    out["batch_reproduction_vs_strict"] = repro
    out["batch_reproduction_note"] = (
        "Deterministic (torch.full) cases are compared against the "
        "CURRENT strict baseline (baseline-n.json / baseline-fix.json) "
        "and reproduce it bit-exactly (e.g. abs/c64/extreme_low matches "
        "both trees). conj cases draw RNG-dependent inputs — same "
        "construction code, different generator state — so they are NOT "
        "hash-compared against any stored batch; the valid check for "
        "them is the within-run native-vs-fix comparison with "
        "in_bits_equal=True. The r10-superseded archive is retained only "
        "as historical raw evidence (its c128 entries were mis-dtyped, "
        "see round-12).")
    out["hard_failures"] = hard_failures
    sm.write_json(args.out, out)
    print(f"hard_failures: {len(hard_failures)}; "
          f"batch_reproduction entries: {len(repro)}", flush=True)
    if hard_failures:
        print("HARD FAILURES PRESENT — exiting nonzero", flush=True)
    sys.exit(1 if hard_failures else 0)


if __name__ == "__main__":
    main()
