"""Dispatch: with the include path active, gelu must actually run FlagGems
Triton kernels -- not the vendor ``gelu_highprecision_nv`` path.

Profiles one gelu fwd/bwd pair after enable_flaggems(include=[gelu,
gelu_backward]) and asserts the captured kernel names contain the FlagGems
``gelu_none_kernel`` and no vendor ``highprecision`` gelu.
"""

import sys

import torch


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device; exit 2 (skip is not a pass)", file=sys.stderr)
        return 2

    from omegaconf import OmegaConf
    from flagscale.train.utils.flaggems import enable_flaggems

    hw = OmegaConf.create({"use_flaggems": True,
                           "flaggems_include": ["gelu", "gelu_backward"]})
    if not enable_flaggems(hw, "cuda"):
        print("FAIL: enable_flaggems returned False", file=sys.stderr)
        return 1

    x = torch.randn(10, 299, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    for _ in range(3):
        x.grad = None
        torch.nn.functional.gelu(x).backward(dy)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    x.grad = None
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.nn.functional.gelu(x).backward(dy)
        torch.cuda.synchronize()

    names = [e.key for e in prof.key_averages() if e.self_device_time_total > 0]
    gems = [n for n in names if "gelu" in n.lower() and "kernel" in n.lower()]
    vendor = [n for n in names if "highprecision" in n.lower()]
    for n in names:
        print(f"  kernel: {n[:100]}")
    if not gems:
        print("FAIL: no FlagGems gelu kernel captured", file=sys.stderr)
        return 1
    if vendor:
        print(f"FAIL: vendor highprecision gelu still dispatched: {vendor}", file=sys.stderr)
        return 1
    print(f"PASS: FlagGems kernels dispatched ({gems})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
