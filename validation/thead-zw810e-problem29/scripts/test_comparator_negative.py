"""Comparator negative tests (device-independent, pure CPU).

Verifies the v3/v4 accuracy gates REJECT: classification changes (NaN/Inf
appearance), Inf sign flips, out-of-tolerance errors, grad presence mismatch;
accepts gems-better-than-native within tolerance; treats FTZ-zone bitwise
differences and zero-sign differences as DIFF (recorded), not PASS.
Also asserts the issue-required boundary bit patterns decode to subnormals:
fp16 0x03FF, bf16 0x007F, fp32 0x007FFFFF.

Mirrors gates in accuracy_matrix.py v4 (kept in sync by this self-test).
Exit code 0 only if every expectation holds; any mismatch -> non-zero.
"""
import torch

EPS = {"float32": 1.1920929e-07, "bfloat16": 0.0078125, "float16": 0.0009765625}
MIN_NORMAL = {"float32": 1.1754943508222875e-38, "bfloat16": 1.1754943508222875e-38, "float16": 6.103515625e-05}

failures = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


def m_nan(t):
    return torch.isnan(t.float())


def m_inf(t):
    return torch.isinf(t.float())


def m_inf_sign(t):
    f = t.float()
    return torch.isposinf(f), torch.isinf(f) & torch.signbit(f)


def rel_err(a, oracle):
    af = a.double()
    fin = torch.isfinite(af) & torch.isfinite(oracle)
    if not fin.any():
        return 0.0
    denom = oracle[fin].abs().clamp_min(1e-300)
    return float(((af[fin] - oracle[fin]).abs() / denom).max())


def gate(native_out, gems_out, native_grad_present, gems_grad_present, oracle, dtype, ftz_bitwise_equal=True, zero_sign_equal=True):
    pin, nin = m_inf_sign(native_out)
    pig, nig = m_inf_sign(gems_out)
    hard = not (
        bool(torch.equal(m_nan(native_out), m_nan(gems_out)))
        and bool(torch.equal(m_inf(native_out), m_inf(gems_out)))
        and bool(torch.equal(pin, pig) and torch.equal(nin, nig))
        and native_grad_present == gems_grad_present
        and rel_err(gems_out.float(), oracle) <= max(4 * EPS[dtype], 1.5 * rel_err(native_out.float(), oracle))
    )
    if hard:
        return "FAIL"
    if not (ftz_bitwise_equal and zero_sign_equal):
        return "DIFF"
    return "PASS"


# boundary bit patterns (issue #5 requirement)
x16 = torch.tensor([0x03FF], dtype=torch.int16).view(torch.float16).item()
xb16 = torch.tensor([0x007F], dtype=torch.int16).view(torch.bfloat16).item()
x32 = torch.tensor([0x007FFFFF], dtype=torch.int32).view(torch.float32).item()
check("fp16 0x03FF is max subnormal (< min normal 0x0400)", 0 < x16 < MIN_NORMAL["float16"])
check("bf16 0x007F is max subnormal (< min normal 0x0080)", 0 < xb16 < MIN_NORMAL["bfloat16"])
check("fp32 0x007FFFFF is max subnormal (< min normal 0x00800000)", 0 < x32 < MIN_NORMAL["float32"])

o = torch.tensor([1.0, 2.0, 3.0])
# 1 NaN appearance where native finite -> must FAIL
g = torch.tensor([1.0, float("nan"), 3.0])
check("NaN appearance rejected", gate(o, g, True, True, o.double(), "float32") == "FAIL")
# 2 Inf sign flip (+inf vs -inf) -> must FAIL (sign-aware; regression for v4 gap)
g = torch.tensor([1.0, float("-inf"), 3.0])
check("+Inf vs -Inf rejected", gate(torch.tensor([1.0, float("inf"), 3.0]), g, True, True, o.double(), "float32") == "FAIL")
# 3 Inf appearance -> must FAIL
check("Inf appearance rejected", gate(o, torch.tensor([1.0, float("inf"), 3.0]), True, True, o.double(), "float32") == "FAIL")
# 4 grad presence mismatch -> must FAIL
check("grad presence mismatch rejected", gate(o, o.clone(), True, False, o.double(), "float32") == "FAIL")
# 5 out-of-tolerance error (10% off, native near-perfect) -> must FAIL
g = torch.tensor([1.1, 2.0, 3.0])
check("out-of-tolerance error rejected", gate(o, g, True, True, o.double(), "float32") == "FAIL")
# 6 within tolerance (gems 1.5 ulp-ish off, native slightly worse) -> PASS
n2 = torch.tensor([1.0 + 2e-7, 2.0, 3.0])
g2 = torch.tensor([1.0 + 1e-7, 2.0, 3.0])
check("gems within tol vs native accepted", gate(n2, g2, True, True, o.double(), "float32") == "PASS")
# 7 gems LESS accurate than native beyond 1.5x native error -> FAIL
n3 = torch.tensor([1.0 + 1e-7, 2.0, 3.0])
g3 = torch.tensor([1.0 + 5e-6, 2.0, 3.0])
check("gems less accurate than native rejected", gate(n3, g3, True, True, o.double(), "float32") == "FAIL")
# 8 FTZ zone bitwise mismatch only -> DIFF (recorded, not FAIL, not PASS)
check("FTZ-only mismatch -> DIFF", gate(o, o.clone(), True, True, o.double(), "float32", ftz_bitwise_equal=False) == "DIFF")
# 9 zero-sign only -> DIFF with note (never silently PASS)
check("zero-sign-only -> DIFF", gate(o, o.clone(), True, True, o.double(), "float32", zero_sign_equal=False) == "DIFF")
# 10 bitwise identical -> PASS
check("bitwise identical -> PASS", gate(o, o.clone(), True, True, o.double(), "float32") == "PASS")

print(f"\n{len(failures)} failure(s)")
raise SystemExit(1 if failures else 0)
