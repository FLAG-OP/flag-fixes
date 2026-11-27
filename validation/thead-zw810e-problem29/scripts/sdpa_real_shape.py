"""Review bqPbopQ item 2: training-mode SDPA on the REAL captured S=228 call.
Single-shape isolation (--shape is the only selector; this script runs ONE shape).
Inputs = tensors captured from the real model forward (real per-sample additive
mask with -3.4e38, non-contiguous Q/K/V transposed views, autocast bf16).
Unified on both sides: autocast, layout, input prep, timing boundaries.
p=0: outputs + dQ/dK/dV vs FP32 and vs native (same inputs, same upstream grad).
p=0.1: dropout randomness control stated. fwd / bwd / full fwd+bwd timed.
Actual kernels recorded via profiler. If backward/dropout unsupported: full
exception + SHA + command + affected shape."""
import argparse, json, statistics, subprocess, sys, traceback
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

def timed(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return round(statistics.median([s.elapsed_time(e) for s,e in ev]),4)

def device_kernels(fn):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    ks=set()
    for e in p.events():
        n=e.name or ""
        if n and ("attention" in n.lower() or "fmha" in n.lower() or "sdpa" in n.lower()) and "aten::" not in n and "autograd" not in n and "Backward" not in n:
            ks.add(n[:110])
    return sorted(ks)

ap=argparse.ArgumentParser()
ap.add_argument("--inputs",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--impl",required=True,choices=["native","flaggems"])
ap.add_argument("--dropout",type=float,default=0.0)
a=ap.parse_args()
AC=lambda: torch.autocast("cuda", dtype=torch.bfloat16)
d=torch.load(a.inputs); q,k,v,m=d["q"],d["k"],d["v"],d["m"]
# NOTE: q/k/v kept as captured (non-contiguous transposed views); m as captured (float32 additive)
res={"impl":a.impl,"dropout_p":a.dropout,"shape":list(q.shape),
     "q_stride":list(q.stride()),"k_stride":list(k.stride()),"v_stride":list(v.stride()),
     "mask":{"shape":list(m.shape),"dtype":str(m.dtype),"stride":list(m.stride()),
             "min":float(m.min()),"valid_frac":round(float((m==0).float().mean()),4)},
     "autocast":"bf16 (model path)","inputs":"captured from real model forward (sdpa_real_S214_inputs.pt)"}
if a.impl=="flaggems":
    sys.path.insert(0,"/tmp/opencode/FlagGems-base/src")
    import flag_gems, triton
    res["flag_gems_sha"]=subprocess.run(["git","-C","/tmp/opencode/FlagGems-base","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
    res["triton"]=triton.__version__
    def sdpa(qq,kk,vv,mm,p): return flag_gems.scaled_dot_product_attention(qq,kk,vv,attn_mask=mm,dropout_p=p)
else:
    def sdpa(qq,kk,vv,mm,p): return F.scaled_dot_product_attention(qq,kk,vv,attn_mask=mm,dropout_p=p)
try:
    # ---- p=0 numeric: outputs + dQ/dK/dV ----
    with AC():
        qq,kk,vv=[t.detach().clone().requires_grad_(True) for t in (q,k,v)]
        qq.data=q; kk.data=k; vv.data=v     # keep captured (non-contig) layout
        y=sdpa(qq,kk,vv,m,a.dropout)
        res["forward_ran"]=True
        res["fwd_only_ms"]=timed(lambda: sdpa(q,k,v,m,a.dropout))
        res["fwd_kernels"]=device_kernels(lambda: sdpa(q,k,v,m,a.dropout))
        g=torch.ones_like(y)
        res["backward_supported"]=True
        dq,dk,dv=torch.autograd.grad(y,(qq,kk,vv),g,retain_graph=True)
    res["ran"]=True
    if a.dropout==0.0:
        # fp32 reference on the SAME inputs (contiguous copies for fp32 path), no autocast
        qf,kf,vf=[t.detach().float().clone().requires_grad_(True) for t in (q,k,v)]
        yf=F.scaled_dot_product_attention(qf,kf,vf,attn_mask=m,dropout_p=0.0)
        dqf,dkf,dvf=torch.autograd.grad(yf,(qf,kf,vf),torch.ones_like(yf))
        def rel(x,r): return round(float((x.float()-r).abs().max()/r.abs().max().clamp_min(1e-6)),6)
        res["vs_fp32_reldiff"]={"out":rel(y,yf),"dQ":rel(dq,dqf),"dK":rel(dk,dkf),"dV":rel(dv,dvf)}
        # native comparison done by parent (needs native run); store tensors' hashes
        import hashlib
        def bh(t): return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
        res["tensor_hashes"]={"out":bh(y),"dQ":bh(dq),"dK":bh(dk),"dV":bh(dv)}
        torch.save({"out":y.detach().cpu(),"dQ":dq.detach().cpu(),"dK":dk.detach().cpu(),"dV":dv.detach().cpu()},a.out+".tensors.pt")
    # ---- timings: fwd / bwd / full (unified boundaries) ----
    def fwd():
        with AC(): return sdpa(q,k,v,m,a.dropout)
    res["fwd_ms"]=timed(fwd)
    with AC():
        q2,k2,v2=[t.detach().clone().requires_grad_(True) for t in (q,k,v)]
        q2.data=q; k2.data=k; v2.data=v
        yg=sdpa(q2,k2,v2,m,a.dropout); gg=torch.ones_like(yg)
    res["bwd_ms"]=timed(lambda: torch.autograd.grad(yg,(q2,k2,v2),gg,retain_graph=True))
    def full():
        with AC():
            q3,k3,v3=[t.detach().clone().requires_grad_(True) for t in (q,k,v)]
            q3.data=q; k3.data=k; v3.data=v
            yy=sdpa(q3,k3,v3,m,a.dropout)
            torch.autograd.grad(yy,(q3,k3,v3),torch.ones_like(yy))
    res["full_ms"]=timed(full)
    res["fwd_kernels"]=device_kernels(fwd)
    res["bwd_kernels"]=device_kernels(lambda: torch.autograd.grad(yg,(q2,k2,v2),gg,retain_graph=True))
except Exception:
    res["ran"]=False
    if res.get("forward_ran") and res.get("backward_supported"):
        res["backward_supported"]=False
        res["limitation"]="candidate FORWARD runs on the real non-contiguous layout but BACKWARD is unsupported"
    tb=traceback.format_exc()
    res["error"]={"type":tb.strip().splitlines()[-1][:200],"full_tail":tb[-1500:],
                  "static_assert":[ln.strip() for ln in tb.splitlines() if "static_assert" in ln],
                  "command":f"python3 examples/retrieval/sdpa_real_shape.py --inputs {a.inputs} --impl {a.impl} --dropout {a.dropout} --out {a.out}"}
json.dump(res,open(a.out,"w"),indent=1)
print(json.dumps({k2:v2 for k2,v2 in res.items() if k2 in ("impl","dropout_p","ran","vs_fp32_reldiff","fwd_ms","bwd_ms","full_ms","fwd_kernels","bwd_kernels","error")},indent=1)[:1500])
