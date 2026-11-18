#!/usr/bin/env python3
"""Run the ORIGINAL examples/retrieval/profile_step_segments.py with the real
FlagScale FlagGems integration applied first.

Why this exists: profile_step_segments.py builds the model/optimizer/loaders
directly and never calls flagscale.train.utils.flaggems.enable_flaggems (that
call lives in the trainer's main()).  For the GELU configs we want the exact
same activation code path the real trainer uses (flag_gems.only_enable via the
patched flagscale/train/utils/flaggems.py), applied at the same lifecycle point
(after device init, before model build), then hand control to the untouched
attribution script.

Usage (mirrors profile_step_segments.py args):
  python3 scripts/segments_with_fixes.py \
      --conf examples/retrieval/conf/train/bge_m3.yaml \
      --steps 3 --skip 2 --device cuda --out results/native_r1.json \
      [--flaggems-include gelu,gelu_backward] [--flagscale-root PATH]
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--flaggems-include", default=None,
                    help="comma list passed to flaggems_include (enables the patched integration)")
    ap.add_argument("--flagscale-root", default="/workspace/FlagScale")
    ap.add_argument("--out", default=None)
    args, rest = ap.parse_known_args()

    sys.path.insert(0, args.flagscale_root)
    os.environ.setdefault("FLAGSCALE_DEVICE", "cuda")
    os.environ.setdefault("GEMS_VENDOR", "kunlunxin")

    import torch  # noqa: E402
    from omegaconf import OmegaConf  # noqa: E402
    from flagscale.train.utils.flaggems import enable_flaggems  # noqa: E402
    from flagscale.train.train_retrieval import init_distributed  # noqa: E402

    hardware = {"use_flaggems": False}
    if args.flaggems_include:
        hardware = {
            "use_flaggems": True,
            "flaggems_include": [s.strip() for s in args.flaggems_include.split(",") if s.strip()],
            "flaggems_required": True,
        }
    _rank, _world, device, platform = init_distributed(OmegaConf.create({"device_type": "cuda"}))
    enabled = enable_flaggems(OmegaConf.create(hardware), device, platform)
    print(f"[segments_with_fixes] enable_flaggems -> {enabled} include={hardware.get('flaggems_include')}", flush=True)

    argv = [sys.argv[0], "--conf", args.conf]
    if args.out:
        argv += ["--out", args.out]
    argv += rest
    sys.argv = argv
    runpy_path = os.path.join(args.flagscale_root, "examples/retrieval/profile_step_segments.py")
    runpy = __import__("runpy")
    runpy.run_path(runpy_path, run_name="__main__")


if __name__ == "__main__":
    main()
