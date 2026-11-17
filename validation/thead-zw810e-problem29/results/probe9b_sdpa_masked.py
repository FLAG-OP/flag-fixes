"""SDPA with padding mask — does flash still serve, or fall to math/efficient?
Real config: padded batches -> attn_mask (B,1,S,S) bool. bf16."""
import json, statistics
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

OUT = {"device": torch.cuda.get_device_name(0)}
B, S, Hh, D = 8, 512, 16, 64
torch.manual_seed(0)
q = torch.randn(B, Hh, S, D, device="cuda", dtype=torch.bfloat16)
# realistic padding mask: last 20% of tokens padded for half the batch
mask = torch.zeros(B, 1, S, S, dtype=torch.bool, device="cuda")
mask[:, :, :, -100:] = True
mask[4:, :, -100:, :] = True

def timed(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return round(statistics.median(s.elapsed_time(e) for s,e in ev),4)

def sdpa_run_masked(q, backend, m):
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q,q,q,attn_mask=m)


res={}
for name, mk in (("masked_bool", lambda: F.scaled_dot_product_attention(q,q,q,attn_mask=~mask)),
                 ("masked_float", lambda: F.scaled_dot_product_attention(q,q,q,attn_mask=(~mask).float().masked_fill(~mask, -1e9)))):
    try:
        res[name] = {"default_ms": timed(mk)}
    except Exception as e:
        res[name] = {"error": f"{type(e).__name__}: {e}"[:150]}
for bn, be in {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}.items():
    try:
        with sdpa_kernel(be):
            F.scaled_dot_product_attention(q,q,q,attn_mask=~mask); torch.cuda.synchronize()
        res[f"masked_bool/{bn}"] = {"ok": True, "ms": timed(lambda be=be: sdpa_run_masked(q, be, ~mask))}
    except Exception as e:
        res[f"masked_bool/{bn}"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:150]}

def sdpa_run_masked(q, backend, m):
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q,q,q,attn_mask=m)

OUT = {"device": OUT["device"], "shape": f"B{B} H{Hh} S{S} D{D} bf16 + padding mask", "results": res}
with open("results/probe9b_sdpa_masked.json","w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1))
