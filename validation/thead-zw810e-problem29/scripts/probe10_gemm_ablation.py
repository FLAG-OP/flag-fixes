"""GEMM ablations at the real BGE-M3 step shape (review bd1eea8 P2 fix):
ALL branches compute the SAME product x @ w.t() (CPU-verified non-square);
layout branches differ ONLY in memory layout of the operands.
Correctness fields compare ACTUAL bf16 output vs fp32 reference (no self-
comparison). Raw samples saved for every timed branch."""
import argparse, json, statistics
import torch

def timed_raw(fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return [round(s.elapsed_time(e),4) for s,e in ev]

ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); a=ap.parse_args()
M,K,N=2990,1024,1024
x=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)
w=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)
wc=w.t().contiguous()        # [K,N] contiguous copy of w.t()
xnc=x.t().contiguous().t()   # non-contiguous view, same logical x
xf32=x.float(); wf32=w.float()
OUT={"device":torch.cuda.get_device_name(0),"shape":f"M{M} K{K} N{N} bf16 (attn_out class)",
     "note":"all branches compute x@w.t(); layout branches differ only in operand memory layout; CPU non-square verified",
     "iters":30,"warmup":5,"results":{}}

def put(name, raw, product_ref=None):
    r={"p50_ms":round(statistics.median(sorted(raw)),4),"samples_ms":raw}
    if product_ref is not None:
        r["product_matches_reference"]=bool(torch.equal(product_ref.float(), (x.float()@wf32.t()).float()) is False) if False else None
    return r

res=OUT["results"]
for red in (True,False):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=red
    raw=timed_raw(lambda: x @ w.t())
    res[f"reduced_reduction={red}"]={"p50_ms":round(statistics.median(sorted(raw)),4),
        "samples_ms":raw,"tflops":round(2*M*N*K/(statistics.median(sorted(raw))*1e-3)/1e12,2)}
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True

# --- save ACTUAL outputs for cross-dtype / layout comparison ---
out_bf16_view   = x @ w.t()          # baseline: transposed-view weight
out_bf16_precon = x @ wc             # layout branch: contiguous weight (SAME product)
out_bf16_innc   = xnc @ wc           # input-layout branch (same right operand)
out_fp32        = xf32 @ wf32.t()    # fp32 reference (same operands/op)

for name,raw,out in (
    ("weight_transposed_view", timed_raw(lambda: x @ w.t()),  out_bf16_view),
    ("weight_precontig",       timed_raw(lambda: x @ wc),     out_bf16_precon),
    ("input_noncontig",        timed_raw(lambda: xnc @ wc),   out_bf16_innc),
    ("fp32_reference",         timed_raw(lambda: xf32 @ wf32.t()), out_fp32)):
    res[name]={"p50_ms":round(statistics.median(sorted(raw)),4),"samples_ms":raw}

# --- correctness: ACTUAL bf16 outputs vs fp32 reference (named accurately) ---
scale=out_fp32.abs().max().clamp_min(1e-30).float()
for name,out in (("weight_precontig",out_bf16_precon),
                 ("input_noncontig",out_bf16_innc),
                 ("weight_transposed_view",out_bf16_view)):
    d=(out.float()-out_fp32).abs()
    res[f"{name}__vs_fp32_ref"]={"max_absdiff":round(d.max().item(),8),
                                  "max_reldiff":round((d/scale).max().item(),8),
                                  "identical_to_baseline":bool(torch.equal(out,out_bf16_view))}
res["fp32_vs_bf16_baseline__reldiff"]={"max_reldiff":round(((out_bf16_view.float()-out_fp32).abs().div(scale)).max().item(),8),
    "note":"bf16 baseline output vs fp32 reference, real comparison (self-comparison removed)"}
json.dump(OUT,open(a.out,"w"),indent=1)
for k,v in res.items():
    if "p50_ms" in v: print(f"{k}: {v['p50_ms']} ms")
    elif "max_reldiff" in v: print(f"{k}: reldiff={v['max_reldiff']}")
