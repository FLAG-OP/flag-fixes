"""NaN-aware comparator for the gelu boundary dumps (base vs fixed FlagGems).

Comparison layers (per output tensor, after structural checks):
  1. NaN mask equality            (payloads not compared -- reported separately)
  2. Inf mask + Inf sign equality
  3. zero sign-bit equality       (from saved raw bit patterns)
  4. finite values: bitwise equality required; where bitwise differs, the
     frozen per-case tolerance must hold (documented below) and the worst
     absolute/relative deviation is reported.

Frozen tolerances (predetermined, not derived from this run):
  * fp32 gelu_tanh_bwd: atol 2e-7   (observed pow-vs-x*x rounding ~1 ulp)
  * every other tensor: bitwise (no numeric tolerance)

Exit codes: 0 = all checks pass; 1 = any check fails; 2 = usage error.
--self-test corrupts copies of the fixed dump (finite value change, NaN->Inf
reclassification, missing key) and asserts the comparator rejects each.
"""

import argparse
import json
import math
import sys

TOL = {"gelu_tanh_bwd:torch.float32": 2e-7}


def is_nan(v):
    return v != v


def sign_bit(bits_hex):
    b = int(bits_hex, 16)
    return (b >> 31) & 1 if len(bits_hex) == 8 else (b >> 15) & 1


def classify(values, bits_list):
    nan = [is_nan(v) for v in values]
    inf = [(math.isinf(v), math.copysign(1.0, v) if math.isinf(v) else 0.0) for v in values]
    zero_sign = [(v == 0.0, sign_bit(b)) for v, b in zip(values, bits_list)]
    finite = [i for i in range(len(values)) if not nan[i] and not math.isinf(values[i]) and values[i] != 0.0]
    zero = [i for i in range(len(values)) if values[i] == 0.0 and not nan[i]]
    return nan, inf, zero_sign, finite, zero


def compare_tensor(label, a_row, b_row, failures, report):
    va, ba = a_row["values"], a_row["bits"]
    vb, bb = b_row["values"], b_row["bits"]
    if len(va) != len(vb) or len(ba) != len(bb):
        failures.append(f"{label}: element/bit count differs ({len(va)}/{len(ba)})")
        return
    n_nan = n_inf = n_zero = n_bit = 0
    rounding = []
    na, ia, za, fa, _ = classify(va, ba)
    nb, ib, zb, fb, _ = classify(vb, bb)
    for i in range(len(va)):
        if na[i] or nb[i]:
            if na[i] != nb[i]:
                failures.append(f"{label}[{i}]: NaN classification differs "
                                f"(base={va[i]!r} fixed={vb[i]!r})")
            else:
                n_nan += 1          # NaN mask equal; payloads not compared
            continue
        if ia[i][0] or ib[i][0]:
            if ia[i] != ib[i]:
                failures.append(f"{label}[{i}]: Inf mask/sign differs "
                                f"(base={va[i]!r} fixed={vb[i]!r})")
            else:
                n_inf += 1
            continue
        if va[i] == 0.0:
            if za[i] != zb[i]:
                failures.append(f"{label}[{i}]: signed-zero sign bit differs "
                                f"(base bits={ba[i]} fixed bits={bb[i]})")
            else:
                n_zero += 1
            continue
        if ba[i] == bb[i]:
            n_bit += 1
            continue
        # finite rounding difference: check frozen tolerance
        key = f"{label.split(':')[0]}:{label.split(':')[1].strip()}" if ":" in label else label
        tol = TOL.get(key)
        abs_diff = abs(va[i] - vb[i])
        denom = max(abs(va[i]), abs(vb[i]), 1e-30)
        rel = abs_diff / denom
        rounding.append((i, va[i], vb[i], abs_diff, rel))
        if tol is None:
            failures.append(f"{label}[{i}]: bitwise differs and no frozen tolerance "
                            f"is registered (base={va[i]!r} fixed={vb[i]!r})")
        elif abs_diff > tol:
            failures.append(f"{label}[{i}]: rounding diff {abs_diff:.3e} exceeds "
                            f"frozen atol {tol} (base={va[i]!r} fixed={vb[i]!r})")
    report[label] = {
        "elements": len(va),
        "bitwise_identical": n_bit,
        "nan_mask_equal": n_nan,
        "inf_mask_sign_equal": n_inf,
        "zero_sign_equal": n_zero,
        "rounding_within_frozen_tol": len(rounding),
        "worst_rounding": max(rounding, key=lambda r: r[3])[:4] if rounding else None,
    }


def compare_payload(base, fixed, failures, report):
    if set(base["inputs"]) != set(fixed["inputs"]):
        failures.append("dtype keys differ between dumps")
        return
    for dt in base["inputs"]:
        if base["inputs"][dt]["bits"] != fixed["inputs"][dt]["bits"]:
            failures.append(f"input bits differ for {dt} -- the two runs were not "
                            "fed identical inputs")
    if set(base["outputs"]) != set(fixed["outputs"]):
        failures.append("output dtype keys differ")
        return
    for dt in base["outputs"]:
        if set(base["outputs"][dt]) != set(fixed["outputs"][dt]):
            failures.append(f"output key set differs for {dt}")
            continue
        for key in sorted(k for k in base["outputs"][dt] if not k.startswith("bits")):
            compare_tensor(f"{key}:{dt}", base["outputs"][dt][key],
                           fixed["outputs"][dt][key], failures, report)
        for k in ("nan_propagates",):
            if base["sanity"][dt][k] != fixed["sanity"][dt][k]:
                failures.append(f"sanity {k} differs for {dt}")
    if base["ftz"] != fixed["ftz"]:
        failures.append(f"FTZ records differ: {base['ftz']} vs {fixed['ftz']}")


def self_test(base_path):
    """Negative cases: the comparator must REJECT corrupted copies."""
    base = json.load(open(base_path))
    import copy
    ok = True

    def rejects(payload, tag):
        f = []
        report = {}
        compare_payload(base, payload, f, report)
        passed = len(f) > 0
        print(f"  {'ok  ' if passed else 'FAIL'} self-test {tag}: rejected={passed}"
              + ("" if passed else ""))
        return passed

    # 1. finite value corrupted
    p = copy.deepcopy(base)
    dt = "torch.float32"
    key = "gelu_none_fwd"
    p["outputs"][dt][key]["values"][15] *= 1.01
    ok &= rejects(p, "finite value corrupted")

    # 2. NaN reclassified as Inf
    p = copy.deepcopy(base)
    idx = 4  # nan input position
    p["outputs"][dt][key]["values"][idx] = float("inf")
    ok &= rejects(p, "NaN -> Inf reclassification")

    # 3. missing structure
    p = copy.deepcopy(base)
    del p["outputs"][dt][key]
    ok &= rejects(p, "missing output key")

    # 4. signed-zero sign bit flipped
    p = copy.deepcopy(base)
    i0 = next(i for i, v in enumerate(p["outputs"][dt][key]["values"]) if v == 0.0)
    bits_hex = p["outputs"][dt][key]["bits"][i0]
    flipped = int(bits_hex, 16) ^ (1 << 31)
    p["outputs"][dt][key]["bits"][i0] = f"0x{flipped:08x}"
    p["outputs"][dt][key]["values"][i0] = -0.0
    ok &= rejects(p, "signed-zero sign bit flipped")

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--fixed", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="run negative cases against the comparator")
    a = ap.parse_args()

    if a.self_test:
        if not a.base:
            print('--base required for --self-test', file=sys.stderr)
            return 2
        if not self_test(a.base):
            print("SELF-TEST FAILED", file=sys.stderr)
            return 1
        print("SELF-TEST PASS: comparator rejects all corrupted variants")
        return 0

    base = json.load(open(a.base))
    fixed = json.load(open(a.fixed))
    failures, report = [], {}
    compare_payload(base, fixed, failures, report)

    print(f"{'tensor':<44} {'elem':>5} {'bit':>6} {'nan':>5} {'inf':>5} {'zero':>5} {'round':>6}")
    for label, r in report.items():
        print(f"{label:<44} {r['elements']:>5} {r['bitwise_identical']:>6} "
              f"{r['nan_mask_equal']:>5} {r['inf_mask_sign_equal']:>5} "
              f"{r['zero_sign_equal']:>5} {r['rounding_within_frozen_tol']:>6}")
        wr = r["worst_rounding"]
        if wr:
            print(f"    worst rounding: idx={wr[0]} base={wr[1]!r} fixed={wr[2]!r} "
                  f"abs={wr[3]:.3e}")

    if failures:
        print(f"\nCOMPARATOR FAILED: {len(failures)} issue(s)", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        if a.report:
            json.dump({"passed": False, "failures": failures, "report": report},
                      open(a.report, "w"), indent=2)
        return 1

    print("\nCOMPARATOR PASS: structural, NaN-mask, Inf-sign, signed-zero and "
          "frozen-tolerance checks all satisfied")
    if a.report:
        json.dump({"passed": True, "failures": [], "report": report},
                  open(a.report, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
