"""Issue #7 round-10: masked attention vs equal-length no-mask flash.
Real shape B8 H16 S512 D64 bf16 + padding bool mask. Raw samples archived."""
import json, statistics
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

B,S,Hh,D = 8,512,16,64
torch.manual_seed(0)
q = torch.randn(B,Hh,S,D, device="cuda", dtype=torch.bfloat16)
mask = torch.zeros(B,1,S,S, dtype=torch.bool, device="cuda")
mask[:,:,:, -100:] = True          # last 100 tokens padded for all rows
mask[4:, :, -100:, :] = True       # half the rows: 200 padded (variable padding)

def timed(fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return [round(s.elapsed_time(e),4) for s,e in ev]

res={"shape":f"B{B} H{Hh} S{S} D{D} bf16", "mask":"bool (B,1,S,S); 100 padded cols; rows 4-7 also 200 padded rows",
     "iters":30,"warmup":5,"device":torch.cuda.get_device_name(0),"torch":torch.__version__}
res["masked_default_samples_ms"]=timed(lambda: F.scaled_dot_product_attention(q,q,q,attn_mask=~mask))
res["unmasked_flash_samples_ms"]=timed(lambda: F.scaled_dot_product_attention(q,q,q))
for bn,be in {"flash":SDPBackend.FLASH_ATTENTION,"efficient":SDPBackend.EFFICIENT_ATTENTION,"math":SDPBackend.MATH}.items():
    try:
        with sdpa_kernel(be):
            F.scaled_dot_product_attention(q,q,q,attn_mask=~mask); torch.cuda.synchronize()
        res[f"masked_{bn}_samples_ms"]=timed(lambda be=be: F.scaled_dot_product_attention(q,q,q,attn_mask=~mask))
        res[f"masked_{bn}_available"]=True
    except Exception as e:
        res[f"masked_{bn}_available"]=False
        res[f"masked_{bn}_error"]=f"{type(e).__name__}: {e}"[:120]
json.dump(res,open("/tmp/opencode/zw810e-problem29/results/probe11_mask_vs_equal_length.json","w"),indent=1)
print(json.dumps(res,indent=1))
