"""Negative/positive tests for validate_ab_json.py — assert exit code AND reason.
Covers review bd1eea8 P1 variants + earlier gaps. Exit 0 iff all pass."""
import json, subprocess, tempfile, os, sys, copy

BASE = {"correctness": {
            "losses": [0.7] * 20, "grad_norms": [10.0] * 3,
            "losses_finite": True, "grad_norms_finite": True,
            "nonfinite_param_count": 0, "peak_mem_bytes": 26_092_513_280},
        "wall_ms": 2130.6, "segments_ms_median": {"forward": 687.1}}

def run(d):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    if isinstance(d, str): f.write(d)
    else: json.dump(d, f)
    f.close()
    r = subprocess.run([sys.executable, "scripts/validate_ab_json.py", f.name],
                       capture_output=True, text=True)
    os.unlink(f.name)
    return r.returncode, r.stderr

cases = []
def case(name, mutate, exp_code, exp_reason):
    d = mutate if isinstance(mutate, str) else copy.deepcopy(BASE)
    if not isinstance(mutate, str): mutate(d)
    cases.append((name, d, exp_code, exp_reason))

case("valid baseline", lambda d: None, 0, "")
case("nonfinite_param_count=1 -> FAIL", lambda d: d["correctness"].__setitem__("nonfinite_param_count", 1), 1, "nonfinite_param_count must be 0")
case("nonfinite_param_count deleted -> FAIL", lambda d: d["correctness"].pop("nonfinite_param_count"), 1, "nonfinite_param_count must be int")
case("grad_norms_finite=false -> FAIL", lambda d: d["correctness"].__setitem__("grad_norms_finite", False), 1, "grad_norms_finite must be literal true")
case("losses_finite=null -> FAIL", lambda d: d["correctness"].__setitem__("losses_finite", None), 1, "losses_finite must be literal true")
case("only 4 losses -> FAIL", lambda d: d["correctness"].__setitem__("losses", [0.7]*4), 1, "losses must be list len>=20")
case("empty grads -> FAIL", lambda d: d["correctness"].__setitem__("grad_norms", []), 1, "grad_norms must be list len>=3")
case("inf loss -> FAIL", lambda d: d["correctness"]["losses"].__setitem__(19, float("inf")), 1, "losses contain non-finite")
case("wall missing -> FAIL", lambda d: d.pop("wall_ms"), 1, "wall_ms must be finite")
case("no segments -> FAIL", lambda d: d.pop("segments_ms_median"), 1, "missing segments")
case("corrupt JSON -> FAIL", "{corrupt!", 1, "unparseable")

bad = []
for name, d, exp, reason in cases:
    code, err = run(d)
    ok = code == exp and (reason in err if reason else True)
    if not ok: bad.append(f"{name}: code={code} (exp {exp}) err={err.strip()[:80]}")
    print(f"{'PASS' if ok else 'FAIL'} {name}")
print(f"{len(cases)-len(bad)}/{len(cases)} pass")
sys.exit(1 if bad else 0)
