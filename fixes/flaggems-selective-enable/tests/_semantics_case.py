"""Single semantic case, spawned fresh by test_flaggems_semantics.py.

Prints one JSON line: {"n_ops": int|"full"|"full_minus_1", "has_gelu": bool,
"has_layer_norm": bool, "returned": bool, "warn": bool}.
"""

import json
import os
import sys

import torch


def main():
    from omegaconf import OmegaConf
    from flagscale.train.utils.flaggems import enable_flaggems
    import flag_gems

    mode = os.environ["CASE_MODE"]

    if mode == "include_exclude_both":
        # PR review: include+exclude 同时设置 —— include 胜出并打印 WARNING
        hw = OmegaConf.create({
            "use_flaggems": True,
            "flaggems_include": ["gelu", "gelu_backward"],
            "flaggems_exclude": ["layer_norm"],
        })
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            returned = enable_flaggems(hw, "cuda")
        warn = "takes precedence" in buf.getvalue()
        ops = flag_gems.all_registered_ops()
        names = {o if isinstance(o, str) else getattr(o, "__name__", str(o)) for o in ops}
        print(json.dumps({
            "n_ops": len(names) if len(names) <= 50 else "full",
            "has_gelu": any("gelu" in n for n in names),
            "has_layer_norm": any("layer_norm" in n or "layernorm" in n for n in names),
            "returned": bool(returned),
            "warn": warn,
        }))
        return

    cases = {
        "include_list": {"use_flaggems": True,
                         "flaggems_include": ["gelu", "gelu_backward"]},
        "include_absent": {"use_flaggems": True},
        "include_none_str": {"use_flaggems": True, "flaggems_include": "none"},
        "include_listconfig": {"use_flaggems": True,
                               "flaggems_include": list(OmegaConf.create(
                                   {"l": ["gelu", "gelu_backward"]}).l)},
        "exclude_only": {"use_flaggems": True, "flaggems_exclude": ["gelu"]},
        "off": {"use_flaggems": False},
    }
    hw = OmegaConf.create(cases[mode])
    returned = enable_flaggems(hw, "cuda")
    if not returned:
        print(json.dumps({"n_ops": 0, "has_gelu": False, "has_layer_norm": False,
                          "returned": False}))
        return
    ops = flag_gems.all_registered_ops()
    names = {o if isinstance(o, str) else getattr(o, "__name__", str(o)) for o in ops}
    if len(names) > 50:
        n_ops = "full_minus_1" if not any("gelu" in nm for nm in names) else "full"
    else:
        n_ops = len(names)
    print(json.dumps({
        "n_ops": n_ops,
        "has_gelu": any("gelu" in n for n in names),
        "has_layer_norm": any("layer_norm" in n or "layernorm" in n for n in names),
        "returned": bool(returned),
    }))


if __name__ == "__main__":
    main()
