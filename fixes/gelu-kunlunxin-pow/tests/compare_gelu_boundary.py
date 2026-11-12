"""NaN/signed-zero/bit-aware comparator for the gelu boundary dumps.

Comparison layers (per output tensor, after structural checks):
  1. NaN mask equality            (payloads not compared -- noted as such)
  2. Inf mask + Inf sign equality (math.copysign; not affected by the
     sign-bit extraction bug fixed in this revision)
  3. zero sign-BIT equality       (explicit per-width sign-bit extraction)
  4. finite values: bitwise equality required; where bit patterns differ,
     the FROZEN per-case tolerance must hold and the worst absolute
     deviation is reported.

values <-> bits consistency is verified for every element via the recorded
dtype (NaN payloads are exempt from the float round-trip check).

Frozen tolerance (predetermined, not derived from the run under test):
  * fp32 gelu_tanh_bwd: per-element atol 2e-7 (pow-vs-x*x rounding class)

Exit codes: 0 = all checks pass; 1 = any check fails; 2 = usage error.
--self-test: negative cases (finite corruption beyond tolerance, within
tolerance accepted, signed-zero sign-bit flip, NaN->Inf, missing key, Inf
sign flip) must ALL be rejected by the comparator; the within-tolerance
case must be ACCEPTED (validates the tolerance branch actually works).
"""

import argparse
import copy
import json
import math
import struct
import sys

import torch

TOL = {"gelu_tanh_bwd:torch.float32": 2e-7}

_DTYPE_OBJ = {"torch.float32": torch.float32,
              "torch.bfloat16": torch.bfloat16,
              "torch.float16": torch.float16}


def digits_of(bits_hex):
    return bits_hex[2:] if bits_hex.startswith("0x") else bits_hex


def sign_bit(bits_hex):
    """Sign bit with EXPLICIT width detection (round-3 fix: fp32 hex is 8
    digits including the 0x prefix, previously misrouted to the 16-bit path)."""
    digits = digits_of(bits_hex)
    if len(digits) == 8:
        return (int(digits, 16) >> 31) & 1
    if len(digits) == 4:
        return (int(digits, 16) >> 15) & 1
    raise ValueError(f"unknown bit width: {bits_hex}")


def bits_to_float(bits_hex, dtype_str):
    v = int(digits_of(bits_hex), 16)
    dtype_obj = _DTYPE_OBJ[dtype_str]
    if dtype_obj == torch.float32:
        v32 = v - (1 << 32) if v > (1 << 31) - 1 else v   # 负 float 位型 ≥ 0x80000000
        return torch.tensor([v32], dtype=torch.int32).view(torch.float32).item()
    v16 = v & 0xFFFF
    v16s = v16 - 65536 if v16 > 32767 else v16
    return torch.tensor([v16s], dtype=torch.int16).view(dtype_obj).item()


def float_to_bits(value, dtype_str):
    if dtype_str == "torch.float32":
        i = struct.unpack(">I", struct.pack(">f", value))[0]
        return f"0x{i:08x}"
    if dtype_str == "torch.float16":
        i = struct.unpack(">H", struct.pack(">e", value))[0]
        return f"0x{i:04x}"
    if dtype_str == "torch.bfloat16":
        import numpy as np
        i = int(np.frombuffer(np.array(value, dtype=np.float32).tobytes(), dtype=np.uint32)[0] >> 16)
        return f"0x{i:04x}"
    raise ValueError(dtype_str)


def values_bits_consistent(values, bits_list, dtype_str):
    """Every finite element's recorded bits must round-trip to the recorded
    float value (NaN payloads exempt -- only the NaN class is checked)."""
    for i, (v, b) in enumerate(zip(values, bits_list)):
        try:
            f = bits_to_float(b, dtype_str)
        except Exception as exc:
            return False, i, f"bits decode failed: {type(exc).__name__}: {exc}"
        if math.isnan(f) and math.isnan(v):
            continue
        if f != v:
            return False, i, f"bits decode to {f!r} but value is {v!r}"
    return True, None, None


def is_nan(v):
    return v != v


def classify(values, bits_list):
    nan = [is_nan(v) for v in values]
    inf = [(math.isinf(v), math.copysign(1.0, v)) for v in values]
    zero_sign = [(v == 0.0, sign_bit(b)) for v, b in zip(values, bits_list)]
    return nan, inf, zero_sign


def compare_tensor(label, dtype_str, a_row, b_row, failures, report):
    va, ba = a_row["values"], a_row["bits"]
    vb, bb = b_row["values"], b_row["bits"]
    if len(va) != len(vb) or len(ba) != len(bb):
        failures.append(f"{label}: element/bit count differs ({len(va)}/{len(ba)})")
        return
    # values <-> bits consistency (both trees)
    for tag, vals, bl in (("base", va, ba), ("fixed", vb, bb)):
        consistent, i, why = values_bits_consistent(vals, bl, dtype_str)
        if not consistent:
            failures.append(f"{label}: {tag} values/bits inconsistent at [{i}]: {why}")
            return

    tol = TOL.get(f"{label.split(':')[0]}:{dtype_str}")
    n_nan = n_inf = n_zero = n_bit = 0
    rounding = []
    na, ia, za = classify(va, ba)
    nb, ib, zb = classify(vb, bb)
    for i in range(len(va)):
        if na[i] or nb[i]:
            if na[i] != nb[i]:
                failures.append(f"{label}[{i}]: NaN classification differs "
                                f"(base={va[i]!r} fixed={vb[i]!r})")
            else:
                n_nan += 1  # NaN mask equal; payloads intentionally not compared
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
        abs_diff = abs(va[i] - vb[i])
        denom = max(abs(va[i]), abs(vb[i]), 1e-30)
        rounding.append((i, va[i], vb[i], abs_diff, abs_diff / denom))
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
        "zero_sign_bit_equal": n_zero,
        "rounding_within_frozen_atol": len(rounding),
        "worst_rounding_abs": max((r[3] for r in rounding), default=None),
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
    for dt in sorted(base["outputs"]):
        keys_a = set(k for k in base["outputs"][dt] if not k.startswith("bits"))
        keys_b = set(k for k in fixed["outputs"][dt] if not k.startswith("bits"))
        if keys_a != keys_b:
            failures.append(f"output key set differs for {dt}")
            continue
        for key in sorted(keys_a):
            compare_tensor(f"{key}:{dt}", dt, base["outputs"][dt][key],
                           fixed["outputs"][dt][key], failures, report)
        for k in ("nan_propagates",):
            if base["sanity"][dt][k] != fixed["sanity"][dt][k]:
                failures.append(f"sanity {k} differs for {dt}")
    if base["ftz"] != fixed["ftz"]:
        failures.append(f"FTZ records differ: {base['ftz']} vs {fixed['ftz']}")


# ---------------- self-test（无设备依赖；负例必须全部被拒绝） ----------------

def self_test(base_path):
    base = json.load(open(base_path))
    import copy
    dt = "torch.float32"
    key = "gelu_none_fwd"

    def rejects(payload, tag, must_accept=False, expect_reason=None):
        f, report = [], {}
        compare_payload(base, payload, f, report)
        passed = (len(f) == 0)
        reason_ok = True
        if expect_reason is not None:
            reason_ok = any(expect_reason in x for x in f)
        if must_accept:
            ok = passed
            if not passed:
                print(f"    (unexpected failures: {f[:2]})", file=sys.stderr)
        else:
            ok = (not passed) and reason_ok
            if not reason_ok:
                print(f"    (rejection reason mismatch: expected {expect_reason!r} "
                      f"in {f[:2]})", file=sys.stderr)
        print(f"  {'ok  ' if ok else 'FAIL'} self-test {tag}: "
              f"{'accepted' if passed else 'rejected'} (expected "
              f"{'accept' if must_accept else 'reject'})")
        return ok

    ok = True

    # 1. 有限值改坏（values+bits 一致地改，超出冻结容差）-> 必须拒绝
    p = copy.deepcopy(base)
    i = 15  # x=3.0
    new_v = float(torch.tensor(base["outputs"][dt][key]["values"][i] * 1.01,
                               dtype=torch.float32))   # 先量化到 fp32
    p["outputs"][dt][key]["values"][i] = new_v
    p["outputs"][dt][key]["bits"][i] = float_to_bits(new_v, dt)
    ok &= rejects(p, "finite corrupted beyond tolerance (values+bits coherent)",
                      expect_reason="bitwise differs and no frozen tolerance")

    # 2. 容差内修改（gelu_tanh_bwd fp32，atol 冻结为 2e-7；取 |v| 接近 1 的
    #    元素加 1.5e-7：>=1 fp32 ulp 但 <= atol）-> 必须接受（验证容差分支）
    p = copy.deepcopy(base)
    bt = "gelu_tanh_bwd"
    vals_bwd = p["outputs"][dt][bt]["values"]
    idx_bwd = min((i for i, v in enumerate(vals_bwd) if math.isfinite(v) and abs(v) > 0.5),
                  key=lambda i: abs(abs(vals_bwd[i]) - 1.0), default=None)
    if idx_bwd is None:
        print("  ok   case 2 vacuous: no finite gelu_tanh_bwd element near |v|=1")
    else:
        new_v = float(torch.tensor(vals_bwd[idx_bwd] + 1.5e-7, dtype=torch.float32))
        p["outputs"][dt][bt]["values"][idx_bwd] = new_v
        p["outputs"][dt][bt]["bits"][idx_bwd] = float_to_bits(new_v, dt)
        ok &= rejects(p, "finite within atol 2e-7 but >=1 ulp (must ACCEPT)",
                      must_accept=True)

    # 3. signed-zero 符号位翻转（values 与 bits 一致地翻转）-> 必须拒绝
    p = copy.deepcopy(base)
    i0 = next(i for i, v in enumerate(p["outputs"][dt][key]["values"]) if v == 0.0)
    old_bits = p["outputs"][dt][key]["bits"][i0]
    flipped = int(digits_of(old_bits), 16) ^ (1 << 31)
    p["outputs"][dt][key]["bits"][i0] = f"0x{flipped:08x}"
    p["outputs"][dt][key]["values"][i0] = struct.unpack(">f", struct.pack(">I", flipped))[0]
    ok &= rejects(p, "signed-zero sign-bit flip (values+bits coherent)",
                      expect_reason="signed-zero sign bit differs")

    # 4. NaN -> Inf 重分类 -> 必须拒绝
    p = copy.deepcopy(base)
    idx = next(i for i, v in enumerate(p["outputs"][dt][key]["values"]) if math.isnan(v))
    p["outputs"][dt][key]["values"][idx] = float("inf")
    p["outputs"][dt][key]["bits"][idx] = "0x7f800000"
    ok &= rejects(p, "NaN -> Inf reclassification",
                      expect_reason="NaN classification differs")

    # 5. Inf 符号翻转 -> 必须拒绝（评审建议补充的覆盖；按实际 dtype 写回
    #    位型，并断言拒绝原因是 Inf mask/sign，而非 schema 不一致）
    p = copy.deepcopy(base)
    done = False
    for dtk in sorted(p["outputs"]):
        key5 = "gelu_tanh_fwd"
        if key5 not in p["outputs"][dtk]:
            continue
        vals = p["outputs"][dtk][key5]["values"]
        idx = next((i for i, v in enumerate(vals) if math.isinf(v)), None)
        if idx is None:
            continue
        vals[idx] = -vals[idx]
        bits5 = p["outputs"][dtk][key5]["bits"]
        bits5[idx] = float_to_bits(vals[idx], dtk)
        ok &= rejects(p, "Inf sign flip",
                      expect_reason="Inf mask/sign differs")
        done = True
        break
    if not done:
        print("  ok   Inf sign flip: no Inf output in dumps (case vacuous)")


    # 6. 缺输出键 -> 必须拒绝
    p = copy.deepcopy(base)
    del p["outputs"][dt][key]
    ok &= rejects(p, "missing output key",
                      expect_reason="output key set differs")

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--fixed", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="negative cases; only --base is used (device-free)")
    a = ap.parse_args()

    if a.self_test:
        if not self_test(a.base):
            print("SELF-TEST FAILED", file=sys.stderr)
            return 1
        print("SELF-TEST PASS: all negative cases rejected; tolerance branch validated")
        return 0

    if not a.fixed:
        print("--fixed required without --self-test", file=sys.stderr)
        return 2
    base = json.load(open(a.base))
    fixed = json.load(open(a.fixed))
    failures, report = [], {}
    compare_payload(base, fixed, failures, report)

    print(f"{'tensor':<44} {'elem':>5} {'bit':>6} {'nan':>5} {'inf':>5} {'zero':>5} {'round':>6}")
    for label, r in report.items():
        print(f"{label:<44} {r['elements']:>5} {r['bitwise_identical']:>6} "
              f"{r['nan_mask_equal']:>5} {r['inf_mask_sign_equal']:>5} "
              f"{r['zero_sign_bit_equal']:>5} {r['rounding_within_frozen_atol']:>6}")
        wr = r["worst_rounding_abs"]
        if wr is not None:
            print(f"    worst rounding abs: {wr:.3e}")

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
