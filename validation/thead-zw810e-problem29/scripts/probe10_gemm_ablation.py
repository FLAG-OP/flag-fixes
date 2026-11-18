"""GEMM ablations at the real BGE-M3 step shape (review round-9 step 2):
bf16 reduced-precision reduction on/off, layout, contiguity. M=2990."""
import json, statistics
import torch
OUT={"device": torch.cuda.get_device_name(0), "shape":"M2990 K1024 N1024 bf16 (attn_out class, 2880 calls/step class)"}
def timed(fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return round(statistics.median(s.elapsed_time(e) for s,e in ev),4)
M,K,N=2990,1024,1024
x=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)
w=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)
res={}
for red in (True, False):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = red
    ms=timed(lambda: x @ w.t())
    res[f"reduced_reduction={red}"]={"ms":ms,"tflops":round(2*M*N*K/(ms*1e-3)/1e12,2)}
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
# non-contiguous input (simulating stride patterns from fused ops)
wk=w.t().contiguous()
xnc=x.t().contiguous().t()
res["noncontig_input_ms"]=timed(lambda: xnc @ wk.t())
res["contig_input_ms"]=timed(lambda: x.contiguous() @ wk.t())
# fp32 accumulate reference via float() matmul
xf32=x.float(); wf32=w.t().float()  # pre-convert outside timed region
res["fp32_matmul_ms"]=timed(lambda: xf32 @ wf32.t())
OUT["results"]=res
json.dump(OUT,open("results/probe10_gemm_ablation.json","w"),indent=1)
print(json.dumps(OUT,indent=1))
