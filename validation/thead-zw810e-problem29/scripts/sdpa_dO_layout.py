"""Review bqQqCnw item 1b: forward comparison FIRST, then independent backward
with the REAL captured dO, changing ONLY the dO layout.
Inputs = captured from the real model (S=214 target batch: Q/K/V/mask/dO all
non-contiguous transposed views). For S=228 the same test is run on its
captured inputs (dO re-captured via hook).
p=0. Saves: forward error vs FP32 for native-BF16 and candidate-BF16; then
backward with original dO vs dO.contiguous(): output/dO strides, dQ/dK/dV
errors, contiguization copy time, full fwd+bwd incl. copy. First failing
location per path if it still fails."""
import argparse, json, statistics, sys, time, traceback, hashlib
import torch
import torch.nn.functional as F

def timed(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return round(statistics.median([s.elapsed_time(e) for s,e in ev]),4)

ap=argparse.ArgumentParser()
ap.add_argument("--inputs",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--label",required=True); ap.add_argument("--impl",required=True,choices=["native","flaggems"])
a=ap.parse_args()
AC=lambda: torch.autocast("cuda", dtype=torch.bfloat16)
d=torch.load(a.inputs); q,k,v,m,dO=d["q"],d["k"],d["v"],d["m"],d["dO"]
res={"label":a.label,"impl":a.impl,"S":q.shape[2],"dropout_p":0.0,
     "input_strides":{"q":list(q.stride()),"k":list(k.stride()),"v":list(v.stride()),"dO":list(dO.stride())},
     "input_contig":{"q":q.is_contiguous(),"k":k.is_contiguous(),"v":v.is_contiguous(),"dO":dO.is_contiguous()}}
if a.impl=="flaggems":
    sys.path.insert(0,"/tmp/opencode/FlagGems-base/src")
    import flag_gems
    def sdpa(qq,kk,vv,mm,p): return flag_gems.scaled_dot_product_attention(qq,kk,vv,attn_mask=mm,dropout_p=p)
else:
    def sdpa(qq,kk,vv,mm,p): return F.scaled_dot_product_attention(qq,kk,vv,attn_mask=mm,dropout_p=p)

# ---------- FP32 reference (same inputs, same dO values) ----------
qf,kf,vf=[t.detach().float().clone().requires_grad_(True) for t in (q,k,v)]
yf=F.scaled_dot_product_attention(qf,kf,vf,attn_mask=m,dropout_p=0.0)
dqf,dkf,dvf=torch.autograd.grad(yf,(qf,kf,vf),dO.detach().float().clone())
def rel(x,r): return float((x.float()-r).abs().max()/r.abs().max().clamp_min(1e-9))
res["fp32_ref"]={"out_absmax":float(yf.abs().max()),"dQ_absmax":float(dqf.abs().max()),"dO_absmax":float(dO.float().abs().max())}

# ---------- FORWARD FIRST (always recorded, even if backward fails) ----------
def fwd_fn():
    with AC(): return sdpa(q,k,v,m,0.0)
try:
    y=fwd_fn()
    res["forward"]={"ran":True,"out_stride":list(y.stride()),"out_contig":y.is_contiguous(),
                    "vs_fp32_max_abs":float((y.float()-yf).abs().max()),"vs_fp32_relmax":rel(y,yf),
                    "fwd_ms":timed(fwd_fn)}
except Exception:
    res["forward"]={"ran":False,"error_tail":traceback.format_exc()[-800:]}

# ---------- BACKWARD: only dO layout changes ----------
def bwd(dO_in):
    with AC():
        qq,kk,vv=[t.detach().clone().requires_grad_(True) for t in (q,k,v)]
        qq.data=q; kk.data=k; vv.data=v          # keep captured non-contiguous layout
        yy=sdpa(qq,kk,vv,m,0.0)
        g=torch.autograd.grad(yy,(qq,kk,vv),dO_in,retain_graph=True)
    return g,yy
for tag,dO_in in (("dO_original_layout",dO),("dO_contiguous",dO.contiguous())):
    try:
        g,yy=bwd(dO_in)
        dq,dk,dv=g
        res[f"backward_{tag}"]={"ran":True,"dO_stride_used":list(dO_in.stride()),"dO_contig":dO_in.is_contiguous(),
            "dQ_vs_fp32_relmax":rel(dq,dqf),"dK_vs_fp32_relmax":rel(dk,dkf),"dV_vs_fp32_relmax":rel(dv,dvf),
            "dQ_nonfinite":int((~torch.isfinite(dq.float())).sum()),"dK_nonfinite":int((~torch.isfinite(dk.float())).sum()),"dV_nonfinite":int((~torch.isfinite(dv.float())).sum()),
            "bwd_ms":timed(lambda: bwd(dO_in))}
    except Exception:
        tb=traceback.format_exc()
        res[f"backward_{tag}"]={"ran":False,"dO_stride_used":list(dO_in.stride()),
            "first_fail":next((ln.strip() for ln in tb.splitlines() if "File" in ln and "flag_gems" in ln),None) or tb.strip().splitlines()[-2][:160],
            "assert_or_error":tb.strip().splitlines()[-1][:200],"tail":tb[-900:]}
# third path: contiguous Q/K/V too (localize whether the candidate needs ALL inputs contiguous)
def bwd_allcontig():
    qc,kc,vc=[t.detach().contiguous().clone().requires_grad_(True) for t in (q,k,v)]
    with AC():
        yy=sdpa(qc,kc,vc,m,0.0)
        g=torch.autograd.grad(yy,(qc,kc,vc),dO.contiguous(),retain_graph=True)
    return g,yy
try:
    g,yy=bwd_allcontig()
    dq,dk,dv=g
    res["backward_all_contiguous"]={"ran":True,"dQ_vs_fp32_relmax":rel(dq,dqf),"dK_vs_fp32_relmax":rel(dk,dkf),"dV_vs_fp32_relmax":rel(dv,dvf),
        "dQ_nonfinite":int((~torch.isfinite(dq.float())).sum()),"fwd_out_vs_fp32_relmax":rel(yy,yf),
        "bwd_ms_incl_contiguize":timed(bwd_allcontig)}
except Exception:
    tb=traceback.format_exc()
    res["backward_all_contiguous"]={"ran":False,"assert_or_error":tb.strip().splitlines()[-1][:200],
        "first_fail":next((ln.strip() for ln in tb.splitlines() if "File" in ln and "flag_gems" in ln),None),"tail":tb[-700:]}
# copy cost + full fwd+bwd incl. copy (for the candidate)
res["dO_contiguize_copy_ms"]=timed(lambda: dO.contiguous())
def full_with_copy():
    g=dO.contiguous()
    with AC():
        qq,kk,vv=[t.detach().clone().requires_grad_(True) for t in (q,k,v)]
        qq.data=q; kk.data=k; vv.data=v
        yy=sdpa(qq,kk,vv,m,0.0)
        torch.autograd.grad(yy,(qq,kk,vv),g)
try:
    res["full_fwd_bwd_with_copy_ms"]=timed(full_with_copy)
except Exception:
    res["full_fwd_bwd_with_copy_ms"]=None
json.dump(res,open(a.out,"w"),indent=1)
print(f"[{a.label}/{a.impl}] S{q.shape[2]} | fwd ran={res['forward']['ran']} rel={res['forward'].get('vs_fp32_relmax')} {res['forward'].get('fwd_ms')}ms")
for tag in ("dO_original_layout","dO_contiguous"):
    b=res[f"backward_{tag}"]
    if b["ran"]: print(f"   bwd[{tag}]: dO_contig={b['dO_contig']} relQ/K/V {b['dQ_vs_fp32_relmax']:.2e}/{b['dK_vs_fp32_relmax']:.2e}/{b['dV_vs_fp32_relmax']:.2e} bwd {b['bwd_ms']}ms")
    else: print(f"   bwd[{tag}]: FAIL {b['assert_or_error'][:90]} | first_fail: {b['first_fail']}")
b=res["backward_all_contiguous"]
if b["ran"]: print(f"   bwd[all_contiguous]: relQ/K/V {b['dQ_vs_fp32_relmax']:.2e}/{b['dK_vs_fp32_relmax']:.2e}/{b['dV_vs_fp32_relmax']:.2e} | fwd_out_rel {b['fwd_out_vs_fp32_relmax']:.2e} | {b['bwd_ms_incl_contiguize']}ms")
else: print(f"   bwd[all_contiguous]: FAIL {b['assert_or_error'][:90]} | {b['first_fail']}")
print(f"   copy_ms {res['dO_contiguize_copy_ms']} | full+copy {res['full_fwd_bwd_with_copy_ms']}ms")
