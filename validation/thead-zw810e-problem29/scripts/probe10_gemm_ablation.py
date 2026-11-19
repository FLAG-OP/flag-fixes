"""GEMM ablations at the real BGE-M3 step shape (review round-19 fix):
- fp32 reference now uses SAME operand/layout as bf16 baseline (w.float(),
  not w.t().float()) — previous version computed x@w for fp32 vs x@w.t()
  for bf16 (silent mismatch on square K=N).
- adds contiguous-weight layout control (w.t().contiguous().t()).
- explicit --out (no relative-path default)."""
import argparse, json, statistics
import torch

def timed(fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return round(statistics.median(s.elapsed_time(e) for s,e in ev),4)

ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); a=ap.parse_args()
M,K,N=2990,1024,1024
x=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)
w=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)
OUT={"device":torch.cuda.get_device_name(0),"shape":f"M{M} K{K} N{N} bf16 (attn_out class, 2880 calls/step class)",
     "note":"fp32 ref uses same operand (w.float()) and op (x@w.t()) as bf16 baseline"}
res={}
for red in (True,False):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=red
    ms=timed(lambda: x @ w.t())
    res[f"reduced_reduction={red}"]={"ms":ms,"tflops":round(2*M*N*K/(ms*1e-3)/1e12,2)}
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
# layout controls: transposed-view weight vs pre-contiguous weight
res["weight_transposed_view_ms"]=timed(lambda: x @ w.t())
wc=w.t().contiguous()
res["weight_precontig_ms"]=timed(lambda: x @ wc.t())
xnc=x.t().contiguous().t()
res["noncontig_input_ms"]=timed(lambda: xnc @ wc.t())
res["contig_input_ms"]=timed(lambda: x.contiguous() @ wc.t())
# fp32 reference: SAME operand semantics (x@w.t()), pre-converted outside timing
xf32=x.float(); wf32=w.float()
ref32=(xf32 @ wf32.t())
res["fp32_matmul_ms"]=timed(lambda: xf32 @ wf32.t())
# correctness cross-check: fp32 vs bf16 on identical operands
res["fp32_vs_bf16_max_absdiff"]=round((ref32.float()-(x.float()@wf32.t().float())).abs().max().item(),8)
res["bf16_vs_fp32_relerr"]=round(((x@w.t()).float()-ref32).abs().max().item()/ref32.abs().max().item(),8)
OUT["results"]=res
json.dump(OUT,open(a.out,"w"),indent=1)
print(json.dumps(OUT,indent=1))
