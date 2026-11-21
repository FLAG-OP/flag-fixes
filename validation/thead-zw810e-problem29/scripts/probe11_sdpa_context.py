"""Issue #7 round-13: masked attention backend timing — ALL warmup + timing
INSIDE sdpa_kernel(be) context (review round-12: previous version had timed()
outside context, making backend labels meaningless)."""
import json, statistics
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

B,S,Hh,D = 8,512,16,64
torch.manual_seed(0)
q = torch.randn(B,Hh,S,D, device="cuda", dtype=torch.bfloat16)
mask = torch.zeros(B,1,S,S, dtype=torch.bool, device="cuda")
mask[:,:,:, -100:] = True
mask[4:, :, -100:, :] = True

def timed(fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return round(statistics.median(s.elapsed_time(e) for s,e in ev),4)

def timed_in_context(q, k, v, be, iters=30, warmup=5):
    with sdpa_kernel(be):
        for _ in range(warmup): F.scaled_dot_product_attention(q,k,q)
        torch.cuda.synchronize()
        ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
        for s,e in ev:
            s.record(); r=F.scaled_dot_product_attention(q,k,v); e.record()
        torch.cuda.synchronize()
    return [round(s.elapsed_time(e),4) for s,e in ev]

res = {"shape": f"B{B} H{Hh} S{S} D{D} bf16", "mask": "bool padding",
       "note": "ALL warmup + timing inside sdpa_kernel(be) context", "device": torch.cuda.get_device_name(0)}

for mask_desc, m in (("masked", ~mask), ("unmasked", None)):
    for bn, be in {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}.items():
        key = f"{mask_desc}_{bn}"
        try:
            ms = timed_in_context(q, q, q, be)
            res[key] = {"ok": True, "p50_ms": round(statistics.median(sorted(ms)),4),
                        "samples_ms": ms}
        except Exception as e:
            res[key] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:120]}
    res[f"{mask_desc}_default_ms"] = timed(lambda: F.scaled_dot_product_attention(q,q,q))

json.dump(res, open("results/probe11_sdpa_context.json","w"), indent=1)
print(json.dumps(res, indent=1))
