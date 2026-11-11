# Round-6 diagnostic: FULL profiler event capture for the zeros family
# hit-verification, with explicit kernel-name recognition (review
# 6784122575 P2: descriptions must bind to the actual probe config, keep
# complete event lists, and mark unknown/native kernel names instead of
# inferring hits from expected names).
#
# Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<tree>/src python3 benchmarks/zeros_hitprobe.py --out <path>
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

import torch  # noqa: E402
import small_models as sm  # noqa: E402

try:
    import flag_gems  # noqa: F401
    HAVE_GEMS = True
except ImportError:
    HAVE_GEMS = False


def classify(name):
    n = name.lower()
    if "memcpy" in n or "memset" in n:
        return "memcpy/memset"
    if "at::native" in name or "aten::" in name:
        return "native-aten"
    if "kernel" in n and "at::native" not in name:
        # triton kernels appear as their function names (e.g.
        # sum_dim_kernel_inner); recognized only by convention, so we keep
        # the raw name verbatim and tag it triton-candidate
        return "triton-candidate"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default="f_target")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no device")
        sys.exit(2)
    rec = sm.set_determinism(0, "fp32")
    if HAVE_GEMS:
        # tensors built BEFORE registration (native creation paths)
        x8 = torch.randn(8, 8, dtype=torch.float32,
                         device="cuda").to(torch.complex64)      # probe
        xbig = torch.randn(1024, 1024, dtype=torch.float32,
                           device="cuda").to(torch.complex64)    # 2^20
        flag_gems.only_enable(include=["zeros", "zero_", "zeros_like"])
        keys = sorted(set(flag_gems.all_registered_keys()))
    else:
        x8 = torch.randn(8, 8, dtype=torch.float32,
                         device="cuda").to(torch.complex64)
        xbig = torch.randn(1024, 1024, dtype=torch.float32,
                           device="cuda").to(torch.complex64)
        keys = []
    prov = {
        "group": args.group + ("" if HAVE_GEMS else " (native, no flag_gems)"),
        "flag_gems_available": HAVE_GEMS,
        "registered_keys": keys,
        "registered_target_keys": {k: (k in keys)
                                   for k in ("zeros", "zero_", "zeros_like")},
        "flag_gems_path": (flag_gems.__file__ if HAVE_GEMS else None),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "determinism": rec,
        "sync": "torch.cuda.synchronize() before/after each profiled call",
        "probes": [],
    }
    from torch.profiler import profile, ProfilerActivity
    probes = [
        ("zeros/c64/8x8", lambda: torch.zeros((8, 8),
                                              dtype=torch.complex64,
                                              device="cuda")),
        ("zeros/c64/1024x1024", lambda: torch.zeros((1024, 1024),
                                                    dtype=torch.complex64,
                                                    device="cuda")),
        ("zero_/c64/8x8", lambda: x8.zero_()),
        ("zero_/c64/1024x1024", lambda: xbig.zero_()),
        ("copy_scalar/c64/1024x1024", lambda: xbig.copy_(0)),
    ]
    for name, fn in probes:
        fn()  # warm outside the profiled window
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        events = []
        for ev in prof.key_averages():
            dev_us = getattr(ev, "self_device_time_total", None)
            if dev_us is None:
                dev_us = getattr(ev, "self_cuda_time_total", 0)
            if dev_us and dev_us > 0:
                events.append({
                    "name": ev.key,
                    "class": classify(ev.key),
                    "count": ev.count,
                    "self_device_us_total": dev_us,
                })
        entry = {
            "fn": name,
            "probe_dtype": "complex64",
            "probe_entry": "torch.zeros / Tensor.zero_ / Tensor.copy_(0)",
            "events": events,
            "note": "full event list, no truncation; classes are "
                    "advisory, raw names are authoritative",
        }
        prov["probes"].append(entry)
        print(name, "->", [(e["class"], e["name"][:60]) for e in events],
              flush=True)
    sm.write_json(args.out, prov)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
