"""ZW810E problem29 probe 1: vendor/dispatch/registration entry + profiler hit evidence.

Run BEFORE enable (reference path untouched), then AFTER enable, and prove with
torch.profiler that FlagGems triton kernels actually execute on ZW810E.
"""
import json
import os
import sys

import torch

OUT = {}
OUT["env"] = {
    "PPU_SDK": os.environ.get("PPU_SDK"),
    "PPU_VERSION": os.environ.get("PPU_VERSION"),
    "GEMS_VENDOR": os.environ.get("GEMS_VENDOR"),
    "FLAGGEMS_VENDOR": os.environ.get("FLAGGEMS_VENDOR"),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
OUT["torch"] = torch.__version__
OUT["triton"] = __import__("triton").__version__
OUT["device"] = torch.cuda.get_device_name(0)
OUT["cc"] = list(torch.cuda.get_device_capability(0))

# --- vendor resolution as flag_gems sees it ---
from flag_gems.runtime.backend import device_finder

vendor = None
try:
    res = device_finder.device_vendor()
    vendor = res
except AttributeError:
    try:
        res = device_finder.get_vendor_from_env()
        vendor = res
    except AttributeError:
        vendor = "unknown"

# fallback: replicate decision path manually
OUT["vendor"] = str(vendor)

import flag_gems

# --- reference path runs BEFORE enable: native aten only ---
x = torch.randn(4096, 1280, device="cuda", dtype=torch.float32)
y_ref = torch.nn.functional.gelu(x)
torch.cuda.synchronize()
OUT["native_gelu_finite"] = bool(torch.isfinite(y_ref).all())

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
    y_ref2 = torch.nn.functional.gelu(x)
    torch.cuda.synchronize()
native_events = [e.key for e in prof.key_averages()]
OUT["native_gelu_kernel_names"] = [n for n in native_events if "gelu" in n.lower() or "erf" in n.lower()][:8]

# --- enable and dispatch through FlagGems ---
flag_gems.enable()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
    y_gems = torch.nn.functional.gelu(x)
    torch.cuda.synchronize()
gems_events = [e.key for e in prof.key_averages()]
OUT["gems_gelu_kernel_names"] = [n for n in gems_events if "gelu" in n.lower()][:8]

diff = (y_ref - y_gems).abs().max().item()
OUT["gelu_fp32_max_abs_diff_native_vs_gems"] = diff
OUT["gelu_bitwise_equal_native_vs_gems"] = bool(torch.equal(y_ref, y_gems))

# gelu tanh + backward + gelu_and_mul hits
z = torch.randn(4096, 1280, device="cuda", dtype=torch.float32, requires_grad=True)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
    loss = torch.nn.functional.gelu(z, approximate="tanh").sum()
    loss.backward()
    torch.cuda.synchronize()
names = [e.key for e in prof.key_averages()]
OUT["gems_tanh_bwd_kernel_names"] = sorted(set(n for n in names if "gelu" in n.lower()))[:8]

a = torch.randn(4096, 2560, device="cuda", dtype=torch.float32)
b = torch.randn(4096, 2560, device="cuda", dtype=torch.float32)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
    flag_gems.gelu_and_mul(a, b)
    torch.cuda.synchronize()
names = [e.key for e in prof.key_averages()]
OUT["gems_gelu_and_mul_kernel_names"] = [n for n in names if "gelu" in n.lower()][:8]

# --- dispatch key sanity: same torch.cuda path, vendor override presence ---
import flag_gems.runtime.backend as backend_pkg
OUT["_thead_exists"] = os.path.isdir(os.path.join(os.path.dirname(backend_pkg.__file__), "_thead"))

pass  # enable is registered at import time in this version
print(json.dumps(OUT, indent=2, ensure_ascii=False))
