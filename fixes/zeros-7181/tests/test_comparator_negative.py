# Device-independent negative self-tests for the accuracy comparator
# (tests/test_small_models.py::compare_dumps), added after the round-5
# review: the comparator must FAIL on structural mismatches that a naive
# float64 subtraction would silently accept.
#
# Run: python3 tests/test_comparator_negative.py [--out result.json]
# Exit 0 only if the positive control passes AND every negative fails.
import argparse
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from test_small_models import compare_dumps  # noqa: E402


def make_dump(steps=3, dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    single = {
        "out": torch.randn(4, 5, generator=g).to(dtype),
        "loss": torch.randn((), generator=g).to(dtype),
        "grads": {
            "x": torch.randn(3, 4, generator=g).to(dtype),
            "W": torch.randn(5, 4, generator=g).to(dtype),
        },
        "params_after": {"W": torch.randn(5, 4, generator=g).to(dtype)},
    }
    train = {
        "losses": [float(i) + 0.5 for i in range(steps)],
        "steps": [
            {
                "step": i,
                "grads": {
                    "x": torch.randn(3, 4, generator=g).to(dtype),
                    "W": torch.randn(5, 4, generator=g).to(dtype),
                },
                "params": {"W": torch.randn(5, 4, generator=g).to(dtype)},
            }
            for i in range(steps)
        ],
    }
    return {"single": {"contig": single}, "train": {"contig": train}}


def make_task(dtype=torch.float32):
    g = torch.Generator().manual_seed(1)
    return {
        "X": torch.randn(2, 3, 4, generator=g).to(dtype),
        "params": {"W": torch.randn(5, 4, generator=g).to(dtype)},
        "target": torch.randn(4, 5, generator=g).to(dtype),
    }


THRESHOLDS = {
    "fp32": {"inter": {"atol": 1e-5, "rtol": 1e-4},
             "grad": {"atol": 1e-5, "rtol": 1e-3}},
    "fp64": {"inter": {"atol": 1e-10, "rtol": 1e-8},
             "grad": {"atol": 1e-9, "rtol": 1e-7}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = make_dump()
    task = make_task()
    results = []

    def check(name, verdict, expect_pass):
        ok = verdict["pass"] == expect_pass
        results.append({
            "case": name,
            "expect_pass": expect_pass,
            "got_pass": verdict["pass"],
            "ok": ok,
            "first_fail": verdict.get("first_fail"),
        })
        print(f"{'OK ' if ok else 'BAD'} {name}: pass={verdict['pass']}"
              + (f" first_fail={verdict['first_fail']}"
                 if verdict.get("first_fail") else ""))

    # positive control: identical dumps must pass
    check("control/identical",
          compare_dumps(copy.deepcopy(ref), copy.deepcopy(ref),
                        task, copy.deepcopy(task), THRESHOLDS, "fp32"),
          True)

    # N1: same values but wrong dtype must FAIL
    val = copy.deepcopy(ref)
    for sect in ("out",):
        val["single"]["contig"][sect] = \
            val["single"]["contig"][sect].to(torch.float64)
    check("neg/wrong-dtype-same-values",
          compare_dumps(copy.deepcopy(ref), val, task, copy.deepcopy(task),
                        THRESHOLDS, "fp32"),
          False)

    # N2: broadcastable wrong shape must FAIL (naive subtract would accept)
    val = copy.deepcopy(ref)
    val["single"]["contig"]["out"] = \
        val["single"]["contig"]["out"][0]  # (4,5) -> (5,), broadcastable
    check("neg/broadcastable-wrong-shape",
          compare_dumps(copy.deepcopy(ref), val, task, copy.deepcopy(task),
                        THRESHOLDS, "fp32"),
          False)

    # N3: missing trailing training checkpoint must FAIL
    val = copy.deepcopy(ref)
    val["train"]["contig"]["steps"] = val["train"]["contig"]["steps"][:-1]
    val["train"]["contig"]["losses"] = val["train"]["contig"]["losses"][:-1]
    check("neg/missing-trailing-checkpoint",
          compare_dumps(copy.deepcopy(ref), val, task, copy.deepcopy(task),
                        THRESHOLDS, "fp32"),
          False)

    # N4a: duplicated step ID must FAIL
    val = copy.deepcopy(ref)
    val["train"]["contig"]["steps"][1]["step"] = 0
    check("neg/duplicate-step-id",
          compare_dumps(copy.deepcopy(ref), val, task, copy.deepcopy(task),
                        THRESHOLDS, "fp32"),
          False)

    # N4b: out-of-order step IDs must FAIL
    val = copy.deepcopy(ref)
    val["train"]["contig"]["steps"][0], val["train"]["contig"]["steps"][1] = \
        val["train"]["contig"]["steps"][1], val["train"]["contig"]["steps"][0]
    check("neg/out-of-order-step-ids",
          compare_dumps(copy.deepcopy(ref), val, task, copy.deepcopy(task),
                        THRESHOLDS, "fp32"),
          False)

    # N5: missing grad key in val must FAIL
    val = copy.deepcopy(ref)
    del val["single"]["contig"]["grads"]["x"]
    check("neg/missing-grad-key",
          compare_dumps(copy.deepcopy(ref), val, task, copy.deepcopy(task),
                        THRESHOLDS, "fp32"),
          False)

    # N6: non-finite REFERENCE grad must FAIL (ref experiment invalid,
    # never a silent skip)
    val = copy.deepcopy(ref)
    ref_bad = copy.deepcopy(ref)
    ref_bad["single"]["contig"]["grads"]["x"] = \
        ref_bad["single"]["contig"]["grads"]["x"].clone()
    ref_bad["single"]["contig"]["grads"]["x"][0, 0] = float("inf")
    check("neg/ref-grad-nonfinite",
          compare_dumps(ref_bad, val, task, copy.deepcopy(task),
                        THRESHOLDS, "fp32"),
          False)

    # N7: non-finite FIX grad must FAIL
    val = copy.deepcopy(ref)
    val["single"]["contig"]["grads"]["W"][1, 1] = float("nan")
    check("neg/fix-grad-nonfinite",
          compare_dumps(copy.deepcopy(ref), val, task, copy.deepcopy(task),
                        THRESHOLDS, "fp32"),
          False)

    # N8: value beyond thresholds must FAIL
    val = copy.deepcopy(ref)
    val["single"]["contig"]["out"][0, 0] += 1.0
    check("neg/value-beyond-threshold",
          compare_dumps(copy.deepcopy(ref), val, task, copy.deepcopy(task),
                        THRESHOLDS, "fp32"),
          False)

    all_ok = all(r["ok"] for r in results)
    summary = {"all_ok": all_ok, "n_cases": len(results),
               "cases": results, "comparator": "structural-checks-v2"}
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"COMPARATOR NEGATIVE SELFTEST: "
          f"{'PASS' if all_ok else 'FAIL'} ({len(results)} cases)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
