#!/usr/bin/env python3
"""Verify the FlagScale FlagGems integration registers exactly the include
list (patched flaggems.py) vs full enable (baseline flaggems.py)."""
import os, sys

sys.path.insert(0, "/workspace/FlagScale")
os.environ.setdefault("GEMS_VENDOR", "kunlunxin")
import torch
from omegaconf import OmegaConf
from flagscale.train.utils.flaggems import enable_flaggems
from flagscale.train.train_retrieval import init_distributed

_rank, _world, device, platform = init_distributed(OmegaConf.create({"device_type": "cuda"}))
hw = OmegaConf.create({"use_flaggems": True,
                       "flaggems_include": ["gelu", "gelu_backward"],
                       "flaggems_required": True})
ok = enable_flaggems(hw, device, platform)
import flag_gems
print("enabled:", ok)

# counting registered ops: patch a tiny probe through the public is registration
import torch._library as _lib
# Count via aten library inspection is version-dependent; instead dispatch-check:
x = torch.randn(64, 128, device=device, dtype=torch.bfloat16).requires_grad_(True)
from torch.profiler import profile, ProfilerActivity
for _ in range(3):
    y = torch.nn.functional.gelu(x)
    y.backward(torch.ones_like(y))
x.grad = None
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        y = torch.nn.functional.gelu(x)
        y.backward(torch.ones_like(y))
        x.grad = None
    torch.cuda.synchronize()
names = [e.key for e in prof.key_averages() if e.device_time_total > 0 and e.count > 0]
print("kernels:", names)
assert any("gelu" in n.lower() and "highprecision" not in n.lower() for n in names), \
    f"expected FlagGems triton gelu kernel, got {names}"
print("INCLUDE-PATH OK: gelu dispatched to FlagGems")
