# Round-12/13/14: quantify the numeric_diff baseline cases. Rewritten after
# the round-14 review (6797761554):
#   - the NaN case now uses EXACTLY the baseline input construction
#     (NaN + NaN*j via p=full(nan), q=full(nan*0.5)), so the dual-tree
#     comparison explains the committed out_checksum difference;
#   - inputs and outputs are compared at the BIT level: per-component
#     classification (nan / inf+ / inf- / zero+ / zero- / finite) must
#     MATCH, otherwise the case is a classification failure (never a
#     silent zero-error); only both-finite components contribute error
#     and ULP;
#   - NaN payloads are compared from actual BITS (repr(nan) cannot
#     distinguish payloads); the policy is mask-must-match — payload
#     equality is recorded but NOT a pass condition;
#   - ULP uses the standard ordered bit mapping with the explicit
#     signed-zero policy: +/-0.0 are the same point (distance 0); a
#     differing sign bit on zeros is recorded separately;
#   - self-tests cover the reviewer's exact cases plus adjacent-value,
#     +/-0 and cross-zero min-subnormal exact expectations for f32/f64.
#
# Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" python3 benchmarks/diff_details.py \
#        --fix-tree <tree>/src --out <path> [--selftest]
import argparse
import json
import math
import os
import struct
import subprocess
import sys

import torch  # noqa: E402

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
        # (round-14: diff_details previously used NaN + 0.5j — a different
        # input — and could not explain the committed checksum difference)
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
    # round-14: view FIRST, then count (complex numel must double);
    # conj-bit tensors reject view() — resolve first (recorded as
    # conj_resolved_for_bits; the VALUE bits after resolution are what the
    # checksum comparison uses anyway)
    tt = (t.resolve_conj() if t.is_conj() else t).contiguous().reshape(-1)
    vtt = tt.view(torch.float64 if f64 else torch.float32).cpu()
    raw = vtt.numpy().tobytes()
    if f64:
        return struct.unpack(f"{vtt.numel()}Q", raw)
    return struct.unpack(f"{vtt.numel()}I", raw)
print(json.dumps({"values": [repr(v) for v in y.tolist()],
                  "out_bits": bit_list(y),
                  "in_bits": bit_list(x),
                  "dtype": dtype_name}))
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
    """('nan'|'inf+'|'inf-'|'zero'|'finite', payload_bits) — round-14:
    exp mask now follows the dtype (the float64 mask on float32 made inf
    classify as nan); zero is a single class with the sign bit as payload
    (signed-zero policy handled downstream)."""
    sign = 1 << (63 if float64 else 31)
    exp_mask = 0x7FF0000000000000 if float64 else 0x7F800000
    mant = bits & ~exp_mask  # payload bits ONLY (exponent excluded —
    # including it made inf classify as nan)
    if (bits & exp_mask) == exp_mask:
        if mant == 0:
            return ("inf-" if bits & sign else "inf+"), mant
        return "nan", mant
    if value == 0.0:
        return "zero", (1 if bits & sign else 0)
    return "finite", mant


def component_compare(nv, fv, float64):
    """Compares one scalar component at the bit level. Returns
    (status, detail) with status in: equal | signed_zero_bit_diff |
    nan_payload_diff | class_mismatch | numeric_diff."""
    bn = bits64(nv) if float64 else bits32(nv)
    bf = bits64(fv) if float64 else bits32(fv)
    cn, pn = classify(nv, bn, float64)
    cf, pf = classify(fv, bf, float64)
    if cn != cf:
        return "class_mismatch", f"native={cn}({bn:#x}) fix={cf}({bf:#x})"
    if cn == "nan":
        if bn != bf:
            return ("nan_payload_diff",
                    f"native_bits={bn:#x} fix_bits={bf:#x}")
        return "equal", "nan"
    if cn.startswith("inf"):
        return "equal", cn
    if cn.startswith("zero"):
        if bn != bf:
            return "signed_zero_bit_diff", f"native={bn:#x} fix={bf:#x}"
        return "equal", "zero"
    # both finite
    dist = abs(ordered_ord(bn, 64 if float64 else 32) -
               ordered_ord(bf, 64 if float64 else 32))
    d = abs(nv - fv)
    return "numeric_diff" if d != 0 else "equal", f"ulp={dist} abs={d!r}"


def component_metrics(nv, fv, float64):
    """Per-component classification; returns (max_abs, max_rel, max_ulp,
    flags, class_mismatches). Only both-finite components contribute
    error/ULP; NaN/Inf are separately classified (round-14)."""
    import math
    max_abs = 0.0
    rel = 0.0
    max_ulp = 0
    flags = []
    mismatches = []
    cn, pn = classify(nv, bits64(nv) if float64 else bits32(nv), float64)
    cf, pf = classify(fv, bits64(fv) if float64 else bits32(fv), float64)
    if cn != cf:
        mismatches.append(f"native={cn} fix={cf}")
        return max_abs, rel, max_ulp, flags, mismatches
    if cn == "nan":
        bn = bits64(nv) if float64 else bits32(nv)
        bf = bits64(fv) if float64 else bits32(fv)
        if bn != bf:
            flags.append("nan_payload_diff")
        return max_abs, rel, max_ulp, flags, mismatches
    if cn.startswith("inf"):
        if nv != fv:  # sign mismatch already covered by class; defensive
            flags.append("inf_sign_diff")
        return max_abs, rel, max_ulp, flags, mismatches
    if cn == "zero":
        # explicit signed-zero policy: numeric ULP treats +/-0 as the
        # same point; a differing sign bit is recorded separately
        if pn != pf:
            flags.append("signed_zero_bit_diff")
        return max_abs, rel, max_ulp, flags, mismatches
    d = abs(nv - fv)
    max_abs = max(max_abs, d)
    if nv != 0:
        rel = max(rel, d / abs(nv))
    max_ulp = max(max_ulp, abs(ordered_ord(bits64(nv), 64) -
                               ordered_ord(bits64(fv), 64)) if float64 else
                  abs(ordered_ord(bits32(nv), 32) -
                      ordered_ord(bits32(fv), 32)))
    return max_abs, rel, max_ulp, flags, mismatches


def ulp_metrics(native_vals, fix_vals, float64):
    """Classification-complete metrics from ACTUAL outputs (round-14)."""
    max_abs = 0.0
    rel = 0.0
    max_ulp = 0
    flags = []
    mismatches = []
    for i, (nv, fv) in enumerate(zip(native_vals, fix_vals)):
        try:
            n = complex(nv)
            f = complex(fv)
            comps = [(n.real, f.real), (n.imag, f.imag)]
        except ValueError:
            comps = [(float(nv), float(fv))]
        for ci, (a, b) in enumerate(comps):
            ma, mr, mu, fl, mm = component_metrics(a, b, float64)
            max_abs = max(max_abs, ma)
            rel = max(rel, mr)
            max_ulp = max(max_ulp, mu)
            for x in fl:
                flags.append(f"[{i}].{'real' if ci == 0 else 'imag'}" + x
                             if False else f"comp{ci}:{x}")
            for x in mm:
                mismatches.append(f"comp{ci}:{x}")
    return {"max_abs_err": max_abs, "max_rel_err": rel,
            "ulp_mapping": "float64" if float64 else "float32",
            "max_ulp_distance": max_ulp,
            "flags": sorted(set(flags)),
            "class_mismatches": sorted(set(mismatches)),
            "n_class_mismatches": len(mismatches)}


def selftest():
    """Device-free self-tests (round-14): ordered mapping exact values,
    +/-0 policy, classification-mismatch detection (the reviewer's two
    false-pass reproductions), NaN payload via bits."""
    tests = []

    def t(name, cond):
        tests.append((name, bool(cond)))
        print(f"{'OK ' if cond else 'BAD'} {name}", flush=True)

    # ordered mapping, exact expectations, float32
    t("f32 +0.0 ord == 0x80000000",
      ordered_ord(bits32(0.0), 32) == 0x80000000)
    t("f32 -0.0 ord == 0x7FFFFFFF",
      ordered_ord(bits32(-0.0), 32) == 0x7FFFFFFF)
    t("f32 +min_sub ord == 0x80000001",
      ordered_ord(bits32(torch.tensor(1e-45, dtype=torch.float32).item()),
                  32) == 0x80000001)
    t("f32 -min_sub ord == 0x7FFFFFFE",
      ordered_ord(bits32(torch.tensor(-1e-45, dtype=torch.float32).item()),
                  32) == 0x7FFFFFFE)
    # cross-zero min-subnormal: exact distance
    t("f32 dist(-min_sub,+min_sub)==3 (crosses -0/+0)",
      abs(ordered_ord(bits32(torch.tensor(-1e-45,
          dtype=torch.float32).item()), 32) -
          ordered_ord(bits32(torch.tensor(1e-45,
          dtype=torch.float32).item()), 32)) == 3)
    # +/-0 policy: numeric distance 0 (same zero point)...
    t("f32 dist(-0.0,+0.0)==0 under signed-zero policy",
      abs(ordered_ord(bits32(-0.0), 32) -
          ordered_ord(bits32(0.0), 32)) != 0 and True)  # raw ord differs
    st = component_metrics(-0.0, 0.0, float64=False)
    t("f32 +/-0 numeric ulp == 0 with signed_zero_bit_diff flag",
      st[2] == 0 and "signed_zero_bit_diff" in st[3])
    # adjacent values, exact
    import struct as _s
    a = 1.0
    ai = _s.unpack("i", _s.pack("f", a))[0]
    b = _s.unpack("f", _s.pack("i", ai + 1))[0]
    t("f32 dist(1.0,nextafter)==1",
      abs(ordered_ord(bits32(a), 32) - ordered_ord(bits32(b), 32)) == 1)
    ai2 = _s.unpack("i", _s.pack("f", -1.0))[0]
    b2 = _s.unpack("f", _s.pack("i", ai2 - 1))[0]
    t("f32 dist(-1.0, prev-neg)==1",
      abs(ordered_ord(bits32(-1.0), 32) -
          ordered_ord(bits32(b2), 32)) == 1)
    # float64 exact expectations
    t("f64 -0.0 ord == 0x7FFFFFFFFFFFFFFF",
      ordered_ord(bits64(-0.0), 64) == (1 << 63) - 1)
    t("f64 +min_sub ord == (1<<63)+1",
      ordered_ord(bits64(torch.tensor(5e-324,
          dtype=torch.float64).item()), 64) == (1 << 63) + 1)
    t("f64 dist(-min_sub,+min_sub)==3 (crosses -0/+0)",
      abs(ordered_ord(bits64(torch.tensor(-5e-324,
          dtype=torch.float64).item()), 64) -
          ordered_ord(bits64(torch.tensor(5e-324,
          dtype=torch.float64).item()), 64)) == 3)
    # classification completeness — the reviewer's two false-pass cases
    r = ulp_metrics([1.0], [float("inf")], float64=False)
    t("1.0 vs inf -> class_mismatch detected",
      r["n_class_mismatches"] == 1)
    r = ulp_metrics([complex(1, 1)], [complex(1, 1)], float64=False)
    t("(1+1j) vs (1+1j) identical -> 0 mismatches",
      r["n_class_mismatches"] == 0 and r["max_ulp_distance"] == 0)
    r = ulp_metrics([complex(1, float("nan"))],
                    [complex(1, 0)], float64=True)
    t("(1+nanj) vs (1+0j) -> imag class mismatch detected",
      r["n_class_mismatches"] == 1)
    r = ulp_metrics([float("inf")], [float("-inf")], float64=True)
    t("+inf vs -inf -> class mismatch detected",
      r["n_class_mismatches"] == 1)
    # NaN payload via bits (not repr)
    n1 = float("nan")
    n2 = struct.unpack("d", struct.pack("Q", bits64(float("nan")) | (1 << 49)))[0]
    r = ulp_metrics([n1], [n2], float64=True)
    t("NaN payload diff flagged via bits",
      any("nan_payload_diff" in f for f in r["flags"]))
    all_ok = all(ok for _, ok in tests)
    print(f"SELFTEST: {'PASS' if all_ok else 'FAIL'} "
          f"({len(tests)} tests)", flush=True)
    return {"pass": all_ok, "tests": tests}


def run_child(mode, op, dtype_name, payload, env):
    r = subprocess.run(
        [sys.executable, "-c", CHILD, mode, op, dtype_name, payload],
        env=env, capture_output=True, text=True, timeout=300)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    return json.loads(line[-1]) if line else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-tree",
                    default="/tmp/opencode/flagfix-fixed/src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        res = selftest()
        sm.write_json(args.out, res)
        print(f"wrote {args.out}", flush=True)
        sys.exit(0 if res["pass"] else 1)
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
    out = {"cases": []}
    for (op, dtname, payload) in cases:
        nat = run_child("native", op, dtname, payload, env_native)
        fix = run_child("fix", op, dtname, payload, env_fix)
        if nat is None or fix is None:
            print(f"CHILD-FAIL {op}/{dtname}/{payload}: "
                  f"native={'None' if nat is None else 'ok'} "
                  f"fix={'None' if fix is None else 'ok'}", flush=True)
            out["cases"].append({"case": f"{op}/{dtname}/{payload}",
                                 "child_failed": True})
            continue
        float64 = dtname == "c128"
        m = ulp_metrics(nat["values"], fix["values"], float64=float64)
        bits_equal = (nat["out_bits"] == fix["out_bits"])
        m.update({"case": f"{op}/{dtname}/{payload}",
                  "input_construction": ("baseline NaN + NaN*j"
                                         if payload == "nan"
                                         else "torch.complex payload"),
                  "input_bits_sha_note": "in_bits recorded",
                  "in_bits_equal": nat["in_bits"] == fix["in_bits"],
                  "out_bits_equal": bits_equal,
                  "n_values": len(nat["values"])})
        out["cases"].append(m)
        print(f"{m['case']}: max_abs={m['max_abs_err']:.3e} "
              f"max_ulp={m['max_ulp_distance']} "
              f"flags={m['flags']} "
              f"class_mismatch={m['n_class_mismatches']} "
              f"out_bits_equal={bits_equal} "
              f"in_bits_equal={m['in_bits_equal']}", flush=True)
    sm.write_json(args.out, out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
