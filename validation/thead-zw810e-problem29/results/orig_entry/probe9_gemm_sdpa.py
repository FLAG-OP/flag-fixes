"""Issue #7 round-8: PPU real-shape GEMM efficiency + SDPA backend probe.

problem29 原始归因（PPU）: GEMM 44.3% of gap — 2990x1024x1024 = 44 TFLOP/s.
attention 20.4% (SDPA -> math backend). Real BGE-M3/XLM-R-large step shapes.
"""
import json, statistics
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

OUT = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
torch.manual_seed(0)

def timed(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for s, e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return round(statistics.median(s.elapsed_time(e) for s, e in ev), 4)

def tflops(M, N, K, ms):
    return round(2 * M * N * K / (ms * 1e-3) / 1e12, 2)

M, H = 2990, 1024
gemms = {
    "qkv_1024x3072": (M, 3072, H),
    "attn_out_1024x1024": (M, H, H),
    "ffn_up_1024x4096": (M, 4096, H),
    "ffn_down_4096x1024": (M, H, 4096),
    "lm_head_1024x250002": (M, 250002, H),
}
res = {}
for name, (m, n, k) in gemms.items():
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    wk = w.t().contiguous()
    r = {}
    r["bf16_xw_ms"] = timed(lambda: x @ w.t())
    r["bf16_tflops"] = tflops(m, n, k, r["bf16_xw_ms"])
    r["bf16_xwkn_ms"] = timed(lambda: x @ wk)
    xf, wf = x.float(), w.float()
    r["fp32_ms"] = timed(lambda: xf @ wf.t())
    r["fp32_tflops"] = tflops(m, n, k, r["fp32_ms"])
    res[name] = r
OUT["gemm_bf16_fp32"] = res

def sdpa_run(q, backend):
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, q, q)

B, S, Hh, D = 8, 512, 16, 64
q = torch.randn(B, Hh, S, D, device="cuda", dtype=torch.bfloat16)
sd = {}
for bn, be in {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}.items():
    try:
        with sdpa_kernel(be):
            F.scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
        sd[bn] = {"ok": True, "ms": timed(lambda be=be: sdpa_run(q, be))}
    except Exception as e:
        sd[bn] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:160]}
OUT["sdpa_bf16_b8_s512_h16_d64"] = sd
OUT["sdpa_bf16_default"] = {"ms": timed(lambda: F.scaled_dot_product_attention(q, q, q))}
q32 = q.float()
sd32 = {}
for bn, be in {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}.items():
    try:
        with sdpa_kernel(be):
            F.scaled_dot_product_attention(q32, q32, q32)
            torch.cuda.synchronize()
        sd32[bn] = {"ok": True, "ms": timed(lambda be=be: sdpa_run(q32, be))}
    except Exception as e:
        sd32[bn] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:160]}
OUT["sdpa_fp32_same_shape"] = sd32

with open("results/probe9_gemm_sdpa.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1))
