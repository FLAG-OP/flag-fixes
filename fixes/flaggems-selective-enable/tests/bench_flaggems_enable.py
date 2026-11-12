"""Benchmark gelu fwd/bwd through the integration include path vs vendor.

Times (median of 30) one gelu forward+backward pair at the real model shape
(10, 299, 4096 bf16) in three modes, each in this single process in order:
  vendor            - no FlagGems registration
  include           - enable_flaggems(include=[gelu, gelu_backward])
Writes JSON via --out.  Informational; exit 0 unless CUDA missing (exit 2).
"""

import argparse
import json
import sys
import time

import torch


def bench(fn, n=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device; exit 2 (skip is not a pass)", file=sys.stderr)
        return 2

    from omegaconf import OmegaConf
    from flagscale.train.utils.flaggems import enable_flaggems
    import flag_gems

    x = torch.randn(10, 299, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    traffic = x.numel() * 3 * 2  # x + dy + dx bf16

    def pair(vendor):
        def fn():
            x.grad = None
            with torch.autocast("cuda", enabled=False):
                y = torch.nn.functional.gelu(x)
                y.backward(dy)
        return fn

    rows = {}
    rows["vendor"] = {"pair_ms": round(bench(pair(True)), 4)}

    hw = OmegaConf.create({"use_flaggems": True,
                           "flaggems_include": ["gelu", "gelu_backward"]})
    rows["include_enabled"] = {"enabled": enable_flaggems(hw, "cuda")}
    rows["include_enabled"]["pair_ms"] = round(bench(pair(True)), 4)

    for label, row in rows.items():
        ms = row["pair_ms"]
        row["gbps"] = round(traffic / (ms * 1e-3) / 1e9, 1)
        print(f"{label:16s} pair {ms:8.4f} ms  {row['gbps']:7.1f} GB/s")

    if a.out:
        with open(a.out, "w") as handle:
            json.dump({"shape": [10, 299, 4096], "dtype": "bf16",
                       "iters": a.iters, "rows": rows}, handle, indent=2)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    main()
