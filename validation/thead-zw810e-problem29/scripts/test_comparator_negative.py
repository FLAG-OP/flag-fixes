"""Comparator negative tests v2 (device-independent, review round 2).

v2: calls the PRODUCTION comparator (scripts/comparator.py compare_case) —
the same gate accuracy_matrix.py v6 uses — instead of a duplicated copy.
Also runs comparator.selftest() (ULP ordinal exactness, ±0 raw-bit handling,
the review's dilution counter-example) and asserts issue-required boundary
bit patterns.

Run together with accuracy_matrix.py --inject-fail (archived evidence:
results/accuracy_matrix_inject.json — corrupted case FAIL + process exit 1),
which covers the production ENTRY (not just the gate function).
Exit code 0 only if every expectation holds.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import comparator as C

failures = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        failures.append(name)


# 0. production unit selftest (ULP mapping, raw-bit ±0, dilution counter-example)
for n, c in C.selftest():
    check("comparator.selftest: " + n, c)

# 1. issue-required boundary bit patterns decode to max subnormals
x16 = torch.tensor([0x03FF], dtype=torch.int16).view(torch.float16).item()
xb16 = torch.tensor([0x007F], dtype=torch.int16).view(torch.bfloat16).item()
x32 = torch.tensor([0x007FFFFF], dtype=torch.int32).view(torch.float32).item()
check("fp16 0x03FF is max subnormal", 0 < x16 < C.MIN_NORMAL["float16"])
check("bf16 0x007F is max subnormal", 0 < xb16 < C.MIN_NORMAL["bfloat16"])
check("fp32 0x007FFFFF is max subnormal", 0 < x32 < C.MIN_NORMAL["float32"])

o = torch.tensor([1.0, 2.0, 3.0])
od = o.double()
V = lambda t, g=None, ng=None, og=None, dt="float32": C.compare_case(
    t[0], t[1], ng, g, od, og, dt
) if False else C.compare_case(t[0], t[1], ng, g, od, og, dt)

# 2 NaN appearance -> FAIL (production)
r = C.compare_case(o, torch.tensor([1.0, float("nan"), 3.0]), None, None, od, None, "float32")
check("production: NaN appearance -> FAIL", r["verdict"] == "FAIL")
# 3 Inf sign flip (+inf native vs -inf gems) -> FAIL (production, sign-aware)
r = C.compare_case(torch.tensor([1.0, float("inf"), 3.0]), torch.tensor([1.0, float("-inf"), 3.0]), None, None, od, None, "float32")
check("production: +Inf vs -Inf -> FAIL", r["verdict"] == "FAIL")
# 4 grad presence mismatch -> FAIL
r = C.compare_case(o, o.clone(), o.clone(), None, od, od.clone(), "float32")
check("production: grad presence mismatch -> FAIL", r["verdict"] == "FAIL")
# 5 out-of-tolerance (100% error at normal position, native perfect) -> FAIL
r = C.compare_case(o, torch.tensor([1.1, 2.0, 3.0]), None, None, od, None, "float32")
check("production: out-of-tolerance -> FAIL", r["verdict"] == "FAIL" and r["normal_gate_violations"] >= 1)
# 6 gems slightly better than native -> PASS
r = C.compare_case(torch.tensor([1.0 + 2e-7, 2.0, 3.0]), torch.tensor([1.0 + 1e-7, 2.0, 3.0]), None, None, od, None, "float32")
check("production: gems better than native -> PASS", r["verdict"] == "PASS")
# 7 gems worse than native beyond 1.5x at same position -> FAIL
r = C.compare_case(torch.tensor([1.0 + 1e-7, 2.0, 3.0]), torch.tensor([1.0 + 5e-6, 2.0, 3.0]), None, None, od, None, "float32")
check("production: gems worse than native -> FAIL", r["verdict"] == "FAIL")
# 8 FTZ-only bit mismatch -> DIFF (production soft rule)
orc = torch.tensor([1e-40, 1.0], dtype=torch.float64)
n8 = torch.tensor([1.1754944e-38, 1.0])
g8 = torch.tensor([0.0, 1.0])  # gems flushes, native keeps min normal
r = C.compare_case(n8, g8, None, None, orc, None, "float32")
check("production: FTZ-only mismatch -> DIFF (recorded)", r["verdict"] == "DIFF" and r["ftz_bit_diff_count"] >= 1)
# 9 zero-sign-only -> DIFF via RAW BITS (production)
r = C.compare_case(torch.tensor([0.0, 1.0]), torch.tensor([-0.0, 1.0]), None, None, torch.tensor([0.0, 1.0], dtype=torch.float64), None, "float32")
check("production: zero-sign-only -> DIFF (raw bits)", r["verdict"] == "DIFF" and r["signed_zero_diff_count"] == 1)
# 10 bitwise identical -> PASS
r = C.compare_case(o, o.clone(), None, None, od, None, "float32")
check("production: bitwise identical -> PASS", r["verdict"] == "PASS")

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
