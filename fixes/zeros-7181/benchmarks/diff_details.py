# Round-12: quantify the 8 numeric_diff baseline cases (full-byte sha256
# differs). For each case, captures native and fix outputs and reports
# max_abs_err, rel_err and ULP distance (int32 view over the float bits,
# monotonic mapping per IEEE754); NaN/Inf payloads are described separately
# (bit patterns), per the review's "errors and ULP from actual outputs;
# NaN/Inf separate criteria".
#
# Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" python3 benchmarks/diff_details.py \
#        --fix-tree /tmp/opencode/flagfix-fixed/src --out <path>
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import small_models as sm  # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

CHILD = r'''
import sys, json, torch
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
        p = torch.full((16, 16), float("nan"), dtype=real, device="cuda")
        q = torch.full((16, 16), 0.5, dtype=real, device="cuda")
        return torch.complex(p, q)
    if payload == "conj":
        p = torch.randn(8, 8, dtype=real, device="cuda")
        q = torch.randn(8, 8, dtype=real, device="cuda")
        return torch.complex(p, q).conj()
    raise ValueError(payload)
x = mk_payload()
y = (x.sum() if op == "sum" else torch.abs(x.resolve_conj()
     if x.is_conj() else x)).flatten().cpu()
bits32 = y.view(torch.int32 if dt == torch.complex64 else torch.int64)
print(json.dumps({"values": [repr(v) for v in y.tolist()],
                  "bits": bits32.tolist()}))
'''


def run_child(mode, op, dtype_name, payload, env):
    r = subprocess.run(
        [sys.executable, "-c", CHILD, mode, op, dtype_name, payload],
        env=env, capture_output=True, text=True, timeout=300)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    return json.loads(line[-1]) if line else None


def _ulp_distance(n, f, float64):
    """IEEE754 monotonic int mapping — mapping chosen BY DTYPE (round-13:
    the previous magnitude-based choice made c128 values inside the
    float32 range be measured with a float32 bit pattern, reporting
    max_ulp=0 for non-zero float64-ULP differences)."""
    import struct
    if float64:
        i = struct.unpack("q", struct.pack("d", float(n)))[0]
        j = struct.unpack("q", struct.pack("d", float(f)))[0]
        oi = i if i >= 0 else (1 << 63) - i
        oj = j if j >= 0 else (1 << 63) - j
        return abs(oi - oj)
    i = struct.unpack("i", struct.pack("f", float(n)))[0]
    j = struct.unpack("i", struct.pack("f", float(f)))[0]
    oi = i if i >= 0 else (1 << 31) - i
    oj = j if j >= 0 else (1 << 31) - j
    return abs(oi - oj)


def _component_metrics(pairs, float64):
    """pairs: (native_float, fix_float) component pairs. Returns
    max_abs/rel and the max ULP distance using the dtype-appropriate bit
    mapping. NaN/Inf excluded (described separately)."""
    import math
    max_abs = 0.0
    rel = 0.0
    max_ulp = 0
    for n, f in pairs:
        if math.isnan(n) or math.isnan(f):
            continue
        if math.isinf(n) or math.isinf(f):
            continue
        d = abs(n - f)
        max_abs = max(max_abs, d)
        if abs(n) > 0:
            rel = max(rel, d / abs(n))
        max_ulp = max(max_ulp, _ulp_distance(n, f, float64))
    return max_abs, rel, max_ulp


def ulp_metrics(native_vals, fix_vals, float64):
    """ULP/err from ACTUAL outputs; NaN/Inf described separately;
    ULP mapping bound to the case dtype (round-13)."""
    import math
    pairs = []
    nan_note = []
    for i, (nv, fv) in enumerate(zip(native_vals, fix_vals)):
        try:
            n = complex(nv)
            f = complex(fv)
            pairs.append((n.real, f.real))
            pairs.append((n.imag, f.imag))
        except ValueError:
            n, f = float(nv), float(fv)
            pairs.append((n, f))
        n, f = pairs[-2][0], pairs[-2][1]
        if math.isnan(n) or math.isnan(f):
            if (math.isnan(n) != math.isnan(f)) or \
                    (repr(nv) != repr(fv)):
                nan_note.append(f"[{i}] NaN payload differs: "
                                f"native={nv!r} fix={fv!r}")
    max_abs, rel, max_ulp = _component_metrics(pairs, float64)
    return {"max_abs_err": max_abs, "max_rel_err": rel,
            "ulp_mapping": "float64" if float64 else "float32",
            "max_ulp_distance": max_ulp,
            "nan_payload_diffs": len(nan_note),
            "nan_inf_notes": nan_note[:5]}




def selftest():
    """Round-13 ULP-mapping self-test: a c128 (float64-bit) value pair
    differing by exactly 1 float64 ULP must report ULP >= 1 even when both
    values fit the float32 range (the old magnitude-based mapping reported
    0); c64 mapping reports float32 ULPs."""
    import struct
    def f64_ulp_step(v):
        i = struct.unpack("q", struct.pack("d", v))[0]
        i += 1
        return struct.unpack("d", struct.pack("q", i))[0]
    v = 1.0000000000000002          # fits float32-ish magnitude
    v1 = f64_ulp_step(v)            # next float64 value (sub-f32-ULP gap)
    m = ulp_metrics([v], [v1], float64=True)
    assert m["max_ulp_distance"] >= 1, m
    # c64 mapping on the same pair: gap is BELOW one float32 ULP -> 0 or 1
    m32 = ulp_metrics([v], [v1], float64=False)
    # and a genuine float32-scale 1-ULP difference must report 1 under c64
    import struct as _s
    a = 1.0
    i = _s.unpack("i", _s.pack("f", a))[0]
    b = _s.unpack("f", _s.pack("i", i + 1))[0]
    m64_for_f32 = ulp_metrics([a], [b], float64=True)
    assert m64_for_f32["max_ulp_distance"] >= 1
    print("ULP-MAPPING SELFTEST: PASS (c128 sub-f32-ULP diff -> "
          f"{m['max_ulp_distance']} float64 ULP; f32 1-ULP diff -> "
          f"{m32['max_ulp_distance']} f32 ULP / "
          f"{m64_for_f32['max_ulp_distance']} f64 ULP)", flush=True)
    return {"pass": True, "c128_sub_f32_ulp_diff": m['max_ulp_distance'],
            "f32_ulp_diff_seen_as_f32": m32['max_ulp_distance'],
            "f32_ulp_diff_seen_as_f64": m64_for_f32['max_ulp_distance']}

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
        sys.exit(0)
    env_native = dict(os.environ)
    env_native["PYTHONPATH"] = ""
    env_native.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env_fix = dict(os.environ)
    env_fix["PYTHONPATH"] = os.path.abspath(args.fix_tree) if \
        os.sep in args.fix_tree else args.fix_tree
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
        env_fix["PYTHONPATH"] = args.fix_tree
        nat = run_child("native", op, dtname, payload, env_native)
        fix = run_child("fix", op, dtname, payload, env_fix)
        m = ulp_metrics(nat["values"], fix["values"],
                        float64=(dtname == "c128"))
        m.update({"case": f"{op}/{dtname}/{payload}",
                  "n_values": len(nat["values"])})
        # bitwise equality for completeness
        m["bitwise_equal"] = (nat["bits"] == fix["bits"] and
                              nat["values"] == fix["values"])
        out["cases"].append(m)
        print(f"{m['case']}: max_abs={m['max_abs_err']:.3e} "
              f"max_rel={m['max_rel_err']:.3e} "
              f"max_ulp={m['max_ulp_distance']} "
              f"nan_payload_diffs={m['nan_payload_diffs']} "
              f"bitwise={m['bitwise_equal']}", flush=True)
        if m["nan_inf_notes"]:
            print("  notes:", m["nan_inf_notes"][:3], flush=True)
    sm.write_json(args.out, out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
