"""Repeatable wrapper: dump (target flag_gems) -> compare vs committed baseline.

Compares the boundary outputs of the flag_gems found on PYTHONPATH (the fixed
tree under verification) against the committed FlagGems-BASE baseline
(`results/boundary-base-2026-11-11.json`, produced by the same dumper version
on the unpatched FlagGems @4e08b44e9 tree, P800/XPU3).

Exit codes: 0 pass, 1 mismatch/missing fields (old schema counts as failure),
2 = skip (no CUDA), 3 = baseline file missing/old schema.
"""

import json
import os
import subprocess
import sys
import torch  # noqa: F401 -- torch_available check

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, os.pardir, "results", "boundary-base-2026-11-11.json")


def main():
    if not torch_available():
        print("SKIP: no CUDA device; exit 2 (skip is not a pass)", file=sys.stderr)
        return 2

    out_tmp = os.path.join(HERE, os.pardir, "results", "_boundary_target_dump.json")
    dump_src = os.path.join(HERE, "test_gelu_boundary.py")
    rc = subprocess.run([sys.executable, dump_src, "--out", out_tmp],
                        capture_output=True, text=True, timeout=600)
    if rc.returncode != 0:
        print(f"FAIL: dumper rc={rc.returncode}\n{rc.stderr[-800:]}", file=sys.stderr)
        return 1
    if not os.path.isfile(out_tmp):
        print("FAIL: dumper produced no file", file=sys.stderr)
        return 1

    payload = json.load(open(out_tmp))
    if "inputs" not in payload or "bits" not in next(iter(payload["inputs"].values())):
        print("FAIL: dump is in the OLD schema (missing inputs/bits) -- regenerate "
              "with the current test_gelu_boundary.py", file=sys.stderr)
        return 1

    if not os.path.isfile(BASELINE):
        print(f"FAIL: committed baseline missing: {BASELINE}", file=sys.stderr)
        return 3
    baseline = json.load(open(BASELINE))
    if "inputs" not in baseline:
        print("FAIL: baseline file is in the OLD schema -- regenerate the baseline "
              "with the current dumper (FlagGems @4e08b44e9, unpatched)", file=sys.stderr)
        return 3

    sys.path.insert(0, HERE)
    import compare_gelu_boundary as cmp_mod

    failures, report = [], {}
    cmp_mod.compare_payload(baseline, payload, failures, report)

    passed = not failures
    result = {"passed": passed, "failures": failures, "report": report,
              "baseline": "FlagGems @4e08b44e9 unpatched (dumper v3, P800/XPU3)",
              "target_dump": out_tmp}
    with open(out_tmp + ".compare.json", "w") as handle:
        json.dump(result, handle, indent=2)

    print(f"{'COMPARATOR PASS' if passed else 'COMPARATOR FAILED'}: "
          f"{len(failures)} issue(s); tensors compared: {len(report)}")
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    return 0 if passed else 1


def torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
