#!/usr/bin/env python3
"""GEMM sweep at the REAL BGE-M3 training shapes on KunlunXIN P800 (issue #9 D).

Per encoder pass the model runs 6 linears per layer over 2,990 tokens
(batch 10 x padded 299, bf16):
  qkv/attn-out : (2990,1024) @ (1024,1024)^T   x3
  ffn up       : (2990,1024) @ (1024,4096)^T
  ffn down     : (2990,4096) @ (4096,1024)^T
Per optimizer step: 6 x 24 layers x 20 passes (5 text sets x accum 4).

Each env config: per-shape CUDA-event timing (warmup 20, 100 iters, best of 5),
TFLOPS, and bitwise output comparison against the no-env baseline.
Exit 0; results JSON to --out.
"""
import argparse, json, os
import torch

M = 2990
SHAPES = [
    ("qkv/attnout_1024x1024", 1024, 1024),
    ("ffn_up_1024x4096", 1024, 4096),
    ("ffn_down_4096x1024", 4096, 1024),
]


def bench(fn, warmup=20, iters=100, rounds=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gemm_sweep.json")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    results = {}
    base_refs = {}
    name = "proc"
    entry = {}
    for label, K, N in SHAPES:
        x = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1)
        w = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05)
        b = torch.randn(N, device=dev, dtype=torch.bfloat16) * 0.05
        ms = bench(lambda: torch.nn.functional.linear(x, w, b))
        tflops = 2 * M * K * N / (ms * 1e-3) / 1e12
        y = torch.nn.functional.linear(x, w, b)
        key_fs = label.replace("/", "_")
        ref_path = f"/tmp/opencode/gemm_base_{key_fs}.pt"
        if os.path.exists(ref_path):
            base = torch.load(ref_path)
            same = torch.equal(y, base)
            diff = (y.float() - base.float()).abs().max().item()
            entry[label] = {"ms": round(ms, 4), "tflops": round(tflops, 1),
                            "bitwise_vs_base": bool(same), "max_abs_diff": diff}
        else:
            torch.save(y.detach().clone(), ref_path)
            entry[label] = {"ms": round(ms, 4), "tflops": round(tflops, 1), "bitwise_vs_base": "base"}
    env_snapshot = {k: os.environ.get(k) for k in
                    ("XMLIR_MATMUL_FAST_MODE", "XMLIR_MATMUL_FAST_TUNE", "KLX_USE_AUTOTUNE",
                     "XMLIR_ENABLE_XBLAS_ADDMM", "XMLIR_FC_AUTOTUNE_BOOL", "XMLIR_BMM_DISPATCH_VALUE")}
    results[name] = {"env": env_snapshot, "shapes": entry}
    json.dump({"M": M, "dtype": "bf16", "results": results}, open(a.out, "w"), indent=1)
    print(json.dumps(entry), flush=True)
    print("saved", a.out)


if __name__ == "__main__":
    main()
