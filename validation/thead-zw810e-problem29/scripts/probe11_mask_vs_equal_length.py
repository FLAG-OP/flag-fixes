"""Issue #7 round-13: masked attention vs equal-length no-mask flash.
ALL warmup + timing INSIDE the correct backend context.
Mask IS passed into timed calls for masked variant."""
import json, statistics
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B,S,Hh,D = 8,512,16,64
torch.manual_seed(0)
q = torch.randn(B,Hh,S,D, device="cuda", dtype=torch.bfloat16)
padding_mask = torch.zeros(B,1,S,S, dtype=torch.bool, device="cuda")
padding_mask[:,:,:, -100:] = True
padding_mask[4:, :, -100:, :] = True

def sdpa_with_ctx(q, k, v, be_val, attn_mask=None):
    """Call SDPA inside the specified backend context."""
    if be_val is not None:
        with sdpa_kernel([be_val]):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    else:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

def timed_ms(q_in, be_val, attn_mask=None, iters=30, warmup=5):
    """All warmup + timing inside the correct backend context."""
    for _ in range(warmup):
        sdpa_with_ctx(q_in, q_in, q_in, be_val, attn_mask)
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record()
        sdpa_with_ctx(q_in, q_in, q_in, be_val, attn_mask)
        e.record()
    torch.cuda.synchronize()
    return round(statistics.median(s.elapsed_time(e) for s,e in ev),4)

res = {"shape": f"B{B} H{Hh} S{S} D{D} bf16",
       "mask_desc": "bool padding mask (B,1,S,S); ~100 padded cols; ~half rows with extra padding",
       "iters": 30, "warmup": 5, "device": torch.cuda.get_device_name(0),
       "torch": torch.__version__,
       "note": "ALL warmup + timing inside the correct backend context (round-13 fix)"}

for mask_label, m in (("masked", padding_mask), ("unmasked", None)):
    for bn_name, be_val in [("flash", SDPBackend.FLASH_ATTENTION),
                            ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                            ("math", SDPBackend.MATH)]:
        key = f"{mask_label}_{bn_name}"
        try:
            ms = timed_ms(q, be_val, m)
            res[key] = {"ok": True, "p50_ms": ms}
        except Exception as e:
            res[key] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:150]}

res["masked_default_ms"] = timed_ms(q, None, padding_mask)
res["unmasked_default_ms"] = timed_ms(q, None, None)

json.dump(res, open("/tmp/opencode/zw810e-problem29/results/probe11_masked_v2.json","w"), indent=1)
print(json.dumps(res, indent=1))
