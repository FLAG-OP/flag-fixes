"""Negative/positive tests for check_reuse.py (review 9fe0a05 item 2):
- modify one finite loss, keep wall -> REJECT (result_sha256 mismatch)
- change config (env) only -> REJECT
- valid full match -> REUSE-OK"""
import hashlib, json, subprocess, sys, copy, os

BASE = {"correctness": {"losses": [0.7]*20, "grad_norms": [10.0]*3,
                        "losses_finite": True, "grad_norms_finite": True,
                        "nonfinite_param_count": 0},
        "wall_ms": 2133.62, "segments_ms_median": {"forward": 687.1}}
ENV = {"FL_ADAMW_FUSED": "0", "FL_FLAGGEMS": ""}
SCRIPT_SHA = "a66f" * 16

def sha(b): return hashlib.sha256(b).hexdigest()

def run(manifest, path):
    r = subprocess.run([sys.executable, "scripts/check_reuse.py",
                        json.dumps(manifest), path, SCRIPT_SHA, json.dumps(ENV)],
                       capture_output=True, text=True)
    return r.returncode, r.stderr

results = []
# 1. valid record
b = json.dumps(BASE).encode(); open("/tmp/cr1.json","wb").write(b)
m = {"exit":0, "script_sha":SCRIPT_SHA, "env":ENV, "wall_ms":2133.62, "result_sha256":sha(b)}
code, err = run(m, "/tmp/cr1.json")
results.append(("valid full match -> OK", code==0, err))
# 2. modify one finite loss, keep wall
d = copy.deepcopy(BASE); d["correctness"]["losses"][5] = 0.70001
b2 = json.dumps(d).encode(); open("/tmp/cr2.json","wb").write(b2)
code, err = run(m, "/tmp/cr2.json")
results.append(("changed loss (wall kept) -> REJECT", code!=0 and "result_sha256" in err, err))
# 3. change config only (env in manifest differs)
m3 = dict(m); m3["env"] = {"FL_ADAMW_FUSED":"1", "FL_FLAGGEMS":""}
code, err = run(m3, "/tmp/cr1.json")
results.append(("changed config -> REJECT", code!=0 and "env mismatch" in err, err))
# 4. wall drift >= 0.5
m4 = dict(m); m4["wall_ms"] = 2134.2
code, err = run(m4, "/tmp/cr1.json")
results.append(("wall drift -> REJECT", code!=0 and "wall mismatch" in err, err))
# 5. exit nonzero
m5 = dict(m); m5["exit"] = 1
code, err = run(m5, "/tmp/cr1.json")
results.append(("exit=1 -> REJECT", code!=0, err))
ok = 0
for name, passed, err in results:
    print(("PASS" if passed else "FAIL"), name)
    ok += passed
print(f"{ok}/{len(results)} pass")
sys.exit(0 if ok==len(results) else 1)
