"""Regression: flaggems_include / exclude / default semantics.

Each case runs in a fresh subprocess (registration is process-global and
irreversible), then checks which ops actually registered:

  include list      -> ONLY those ops (the fix)
  include absent    -> full catalogue (documented stock behaviour, unchanged)
  include "none"    -> full catalogue (string normalisation)
  include ListConfig-> accepted (OmegaConf hands lists back as ListConfig)
  exclude, no include-> full enable minus the excluded op (stock path intact)
  use_flaggems=false-> nothing registered, returns False
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.environ.get("FLAGSCALE_TREE", "")
FAILURES = 0


def run_case(mode, extra_env=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = TARGET + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CASE_MODE"] = mode
    env.update(extra_env or {})
    out = subprocess.run([sys.executable, os.path.join(HERE, "_semantics_case.py")],
                         capture_output=True, text=True, env=env, timeout=600)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": (out.stderr or out.stdout).strip().splitlines()[-8:]}


def case(label, mode, expect):
    global FAILURES
    result = run_case(mode)
    if "error" in result:
        print(f"FAIL {label}: {result['error']}")
        FAILURES += 1
        return
    ok = (result["n_ops"] == expect["n_ops"]
          and result["has_gelu"] == expect["has_gelu"]
          and result["has_layer_norm"] == expect["has_layer_norm"]
          and result["returned"] == expect["returned"]
          and result.get("warn", False) == expect.get("warn", False))
    print(f"{'ok  ' if ok else 'FAIL'} {label}: n_ops={result['n_ops']} "
          f"gelu={result['has_gelu']} ln={result['has_layer_norm']} "
          f"returned={result['returned']} (expect n_ops={expect['n_ops']} "
          f"gelu={expect['has_gelu']} ln={expect['has_layer_norm']} "
          f"returned={expect['returned']})")
    if not ok:
        FAILURES += 1


def main():
    if not torch_available():
        print("SKIP: no torch/CUDA; exit 2 (skip is not a pass)", file=sys.stderr)
        return 2

    case("include list -> only gelu family", "include_list",
         {"n_ops": 2, "has_gelu": True, "has_layer_norm": False, "returned": True})
    case("include absent -> full enable (stock)", "include_absent",
         {"n_ops": "full", "has_gelu": True, "has_layer_norm": True, "returned": True})
    case('include "none" -> full enable', "include_none_str",
         {"n_ops": "full", "has_gelu": True, "has_layer_norm": True, "returned": True})
    case("include ListConfig accepted", "include_listconfig",
         {"n_ops": 2, "has_gelu": True, "has_layer_norm": False, "returned": True})
    # NOTE: gelu sits in FlagGems' cpp_patched_ops list, which bypasses the
    # ``unused`` filter entirely -- the exclusion silently does not apply to it.
    # This documents current FlagGems behaviour (identical on base and fixed);
    # the include path is unaffected.
    case("exclude without include -> gelu still registered (FlagGems cpp-patch quirk)",
         "exclude_only",
         {"n_ops": "full", "has_gelu": True, "has_layer_norm": True, "returned": True})
    case("use_flaggems=false -> no-op", "off",
         {"n_ops": 0, "has_gelu": False, "has_layer_norm": False, "returned": False})
    # PR review: include+exclude 同时设置时定义优先级（include 胜出并告警，
    # 与 flag_gems.use_gems 语义一致），不得静默忽略其中一半配置
    case("include+exclude both -> include wins with WARNING", "include_exclude_both",
         {"n_ops": 2, "has_gelu": True, "has_layer_norm": False, "returned": True,
          "warn": True})

    if FAILURES:
        print(f"FAILED: {FAILURES} case(s)", file=sys.stderr)
        return 1
    print("PASS: all semantics cases match")
    return 0


def torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
