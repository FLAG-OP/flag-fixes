"""Issue #7 round-17: masked vs unmasked SDPA, ALL warmup + timing inside
sdpa_kernel(be) context (review round-17 fix: previous version timed outside
the context making backend labels meaningless)."""
import json, statistics
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

B,S,Hh,D = 8,512,16,64
torch.manual_seed(0)
q = torch.randn(B,Hh,S,D, device="cuda", dtype=torch.bfloat16)
padding_mask = torch.zeros(B,1,S,S, dtype=torch.bool, device="cuda")
padding_mask[:,:,:, -100:] = True
padding_mask[4:, :, -100:, :] = True
attn_mask = ~padding_mask  # True = participate (non-padded)

BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}

def timed_in_ctx(q, attn_mask, be, iters=30, warmup=5):
    """ALL warmup + timing inside sdpa_kernel(be)."""
    with sdpa_kernel([be]):
        for _ in range(warmup):
            if attn_mask is not None:
                F.scaled_dot_product_attention(q, q, q, attn_mask=attn_mask)
            else:
                F.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
        ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
        for s, e in ev:
            s.record()
            if attn_mask is not None:
                F.scaled_dot_product_attention(q, q, q, attn_mask=attn_mask)
            else:
                F.scaled_dot_product_attention(q, q, q)
            e.record()
        torch.cuda.synchronize()
    return [round(s.elapsed_time(e), 4) for s, e in ev]

res = {"shape": f"B{B} H{Hh} S{S} D{D} bf16",
       "mask": "bool inverted padding mask (True=participate); ~100 padded cols",
       "iters": 30, "warmup": 5, "device": torch.cuda.get_device_name(0),
       "torch": torch.__version__,
       "note": "ALL warmup+timing inside sdpa_kernel(be) context (round-17 fix)"}

for mask_desc, m in (("masked", attn_mask), ("unmasked", None)):
    for bn_name, be_val in BACKENDS.items():
        key = f"{mask_desc}_{bn_name}"
        try:
            ms = timed_in_ctx(q, m, be_val)
            res[key] = {"ok": True, "p50_ms": round(statistics.median(sorted(ms)), 4),
                        "samples_ms": ms, "backend_enforced": True}
        except Exception as e:
            res[key] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:150]}
    # default (no backend pin)
    dm_key = f"{mask_desc}_default"
    try:
        with sdpa_kernel([SDPBackend.MATH]):
            pass  # verify context works
        ms = timed(lambda q=q, m=m: F.scaled_dot_product_attention(q, q, q, attn_mask=m) if m is not None else F.scaled_dot_product_attention(q, q, q))
        res[dm_key] = {"ok": True, "p50_ms": ms}
    except Exception as e:
        res[dm_key] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:150]}

def timed(fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for s, e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return round(statistics.median(s.elapsed_time(e) for s, e in ev), 4)

json.dump(res, open("/tmp/opencode/zw810e-problem29/results/probe11_sdpa_ctx_r13.json", "w"), indent=1)
print(json.dumps(res, indent=1))
