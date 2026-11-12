"""Reproduce the full-enable regression: include list must be honoured.

Baseline FlagScale hands ``flaggems_include`` to nothing: the call goes through
``flag_gems.enable(unused=...)`` and the FULL catalogue registers.  On
KunlunXIN that is a net regression (9 of 10 representative ops slower,
layer_norm 0.04x, sum 0.03x) even when the intent was only the patched gelu
family.  Exit 1 = full enable happened (baseline behaviour); exit 0 = exactly
the included ops registered (fixed behaviour).
"""

import sys

import torch


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device; exit 2 (skip is not a pass)", file=sys.stderr)
        return 2

    from omegaconf import OmegaConf
    from flagscale.train.utils.flaggems import enable_flaggems
    import flag_gems

    hw = OmegaConf.create({"use_flaggems": True,
                           "flaggems_include": ["gelu", "gelu_backward"]})
    ok = enable_flaggems(hw, "cuda")
    ops = flag_gems.all_registered_ops()
    names = {o if isinstance(o, str) else getattr(o, "__name__", str(o)) for o in ops}
    has_gelu = any("gelu" in n for n in names)
    has_ln = any("layer_norm" in n or "layernorm" in n for n in names)

    print(f"enable_flaggems returned : {ok}")
    print(f"registered ops           : {len(names)}")
    print(f"gelu registered          : {has_gelu}")
    print(f"layer_norm registered    : {has_ln}")

    if len(names) > 50 or has_ln:
        print(
            "FAIL: full catalogue registered despite flaggems_include -- "
            "on KunlunXIN this regresses layer_norm/sum/add/max "
            "(problem29 ATTRIBUTION.md); use the include path instead",
            file=sys.stderr,
        )
        return 1
    if not (has_gelu and len(names) <= 5):
        print(f"FAIL: expected only the gelu family, got {sorted(names)[:10]}",
              file=sys.stderr)
        return 1
    print("PASS: selective include honoured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
