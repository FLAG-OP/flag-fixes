"""Review df851e0 item 5: SDPA with the model's ACTUAL call signature.
Captured from the real model (sdpa_actual_call_r43.json): dropout_p=0.1,
attn_mask = float32 ADDITIVE [B,1,S,S] with -inf at padded keys, q/k/v layout
[B,H,S,D] with TRANSPOSED (non-contiguous) strides, is_causal=False.
Order per shape: sample native kernels FIRST, then only_enable + sample again.
S=512 and S=214 tested separately. Direct FlagGems call tested per shape."""
import argparse, json, statistics, subprocess, sys, traceback
import torch
import torch.nn.functional as F
AC = lambda: torch.autocast("cuda", dtype=torch.bfloat16)   # model runs under autocast
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
        dt=str(getattr(e,"device_type",""))
        if n and ("attention" in n.lower() or "fmha" in n.lower() or "sdpa" in n.lower()) and "aten::" not in n and "autograd" not in n and "Backward" not in n:
            ks.add(n[:110])
    return sorted(ks)

ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True)
ap.add_argument("--shape",default="", help="S512_pad32 | S214_pad16 | '' = run both via subprocesses")
a=ap.parse_args()
SHAPE_MAP={"S512_pad32":(512,32),"S214_pad16":(214,16)}
def run_one(shape_key, out):
    global a
    import sys as _s
    a=argparse.Namespace(out=out, shape=shape_key)
    B,H,D = 10,16,64
    torch.manual_seed(0)
    res={"actual_model_call":{"dropout_p":0.1,"mask":"float32 additive [B,1,S,S], -inf at padded keys",
         "layout":"[B,H,S,D] transposed view strides [B*H*S? -> H*S, D, H*D, 1]","is_causal":False,
         "source":"sdpa_actual_call_r43.json (captured from real model forward)"},
         "shapes":{}}

    def make_case(S_val, pad_last):
        # replicate the model's layout: hidden [B,S,H,D] -> transpose(1,2) -> [B,H,S,D] view
        hidden=torch.randn(B,S_val,H,D,device="cuda",dtype=torch.bfloat16)*0.3
        q=hidden.transpose(1,2)                     # non-contiguous, like the model
        k=torch.randn(B,S_val,H,D,device="cuda",dtype=torch.bfloat16).transpose(1,2)*0.3
        v=torch.randn(B,S_val,H,D,device="cuda",dtype=torch.bfloat16).transpose(1,2)*0.3
        # float32 additive mask: 0 for valid keys, -inf for padded (model semantics)
        m=torch.zeros(B,1,S_val,S_val,device="cuda",dtype=torch.float32)
        m[:,:,:,-pad_last:]=float("-inf")
        return q.contiguous().clone().transpose(1,2).transpose(1,2) if False else q, k, v, m

    for S_val,pad in ((512,32),(214,16)):
        q,k,v,m=make_case(S_val,pad)
        e={"q_stride":list(q.stride()),"mask_dtype":str(m.dtype)}
        def fwd(p):
            def _f():
                with AC(): return F.scaled_dot_product_attention(q,k,v,attn_mask=m,dropout_p=p)
            return _f
        # S-axis perturbation of V at MASKED key positions + no-mask control
        v2=v.clone(); v2[:,:,-pad:,:]+=100.0
        with torch.no_grad():
            with AC():
                y_m=F.scaled_dot_product_attention(q,k,v,attn_mask=m,dropout_p=0.0)
                y_m2=F.scaled_dot_product_attention(q,k,v2,attn_mask=m,dropout_p=0.0)
                y_n=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0); y_n2=F.scaled_dot_product_attention(q,k,v2,dropout_p=0.0)
        e["independence"]={"with_mask_valid_query_allD_max_diff":round(float((y_m2-y_m)[:,:,:S_val-pad,:].abs().max()),8),
                           "no_mask_control_max_diff":round(float((y_n2-y_n).abs().max()),6)}
        # timings: fwd / bwd / full at p=0 and p=0.1 (unified boundaries)
        for p,label in ((0.0,"p0"),(0.1,"p01")):
            e[f"fwd_{label}_ms"]=timed(fwd(p))
            qq,kk,vv=[t.detach().clone().requires_grad_(True) for t in (q,k,v)]
            with AC(): y=F.scaled_dot_product_attention(qq,kk,vv,attn_mask=m,dropout_p=p)
            g=torch.ones_like(y)
            e[f"bwd_{label}_ms"]=timed(lambda: torch.autograd.grad(y,(qq,kk,vv),g,retain_graph=True))
            def full():
                q2,k2,v2_=[t.detach().clone().requires_grad_(True) for t in (q,k,v)]
                with AC(): yy=F.scaled_dot_product_attention(q2,k2,v2_,attn_mask=m,dropout_p=p)
                torch.autograd.grad(yy,(q2,k2,v2_),torch.ones_like(yy))
            e[f"full_{label}_ms"]=timed(full)
        # numeric vs FP32 (p=0, same upstream grad)
        qq,kk,vv=[t.detach().clone().requires_grad_(True) for t in (q,k,v)]
        with AC(): y=F.scaled_dot_product_attention(qq,kk,vv,attn_mask=m,dropout_p=0.0)
        dq,dk,dv=torch.autograd.grad(y,(qq,kk,vv),torch.ones_like(y),retain_graph=True)
        qf,kf,vf=[t.detach().float().clone().requires_grad_(True) for t in (q,k,v)]
        yf=F.scaled_dot_product_attention(qf,kf,vf,attn_mask=m,dropout_p=0.0)   # fp32 ref, no autocast
        dqf,dkf,dvf=torch.autograd.grad(yf,(qf,kf,vf),torch.ones_like(yf))
        def rel(x,r): return round(float((x.float()-r).abs().max()/r.abs().max().clamp_min(1e-6)),6)
        e["native_vs_fp32_reldiff"]={"out":rel(y,yf),"dQ":rel(dq,dqf),"dK":rel(dk,dkf),"dV":rel(dv,dvf)}
        # STEP 1: native kernels at SAME dropout (p=0.1 and p=0) BEFORE any registration
        e["native_kernels_p01"]=device_kernels(fwd(0.1))
        e["native_kernels_p0"]=device_kernels(fwd(0.0))
        res["shapes"][f"S{S_val}_pad{pad}"]=e

    # STEP 2: register candidate, then sample again (per shape) + direct call per shape
    sys.path.insert(0,"/tmp/opencode/FlagGems-base/src")
    import triton, flag_gems, io, contextlib, warnings
    fg_sha=subprocess.run(["git","-C","/tmp/opencode/FlagGems-base","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip() or "29e14822"
    buf=io.StringIO()
    with contextlib.redirect_stderr(buf), warnings.catch_warnings():
        warnings.simplefilter("always")
        flag_gems.only_enable(include=["scaled_dot_product_attention"])
    only_msg=buf.getvalue().strip()
    res["only_enable_message"]=only_msg[:200]
    for S_val,pad in ((512,32),(214,16)):
        q,k,v,m=make_case(S_val,pad)
        e=res["shapes"][f"S{S_val}_pad{pad}"]
        after=device_kernels(lambda: (AC().__enter__(), F.scaled_dot_product_attention(q,k,v,attn_mask=m,dropout_p=0.0))[1])
        e["after_only_enable_kernels_p0"]=after
        e["entry_kernel_set_differs_same_dropout"]=set(after)!=set(e["native_kernels_p0"])
        e["entry_status"]=("NO-OP PASSTHROUGH (identical kernel set at same dropout; only_enable: '"+("No op to register" if "No op to register" in only_msg else "registered")+"')"
                           if (not e["entry_kernel_set_differs_same_dropout"]) else "CANDIDATE KERNEL ENGAGED")
        def _fa():
            with AC(): return F.scaled_dot_product_attention(q,k,v,attn_mask=m,dropout_p=0.0)
        e["fwd_p0_ms_after_only_enable"]=timed(_fa)
        # direct call per shape
        try:
            with AC(): flag_gems.scaled_dot_product_attention(q,k,v,attn_mask=m,dropout_p=0.0)
            e["direct_call"]={"status":"ran","fwd_ms":timed(lambda: flag_gems.scaled_dot_product_attention(q,k,v,attn_mask=m,dropout_p=0.0))}
        except Exception:
            tb=traceback.format_exc()
            tiles=[ln.strip() for ln in tb.splitlines() if ("BLOCK" in ln or "HEAD_DIM" in ln or "GROUP_HEAD" in ln)]
            e["direct_call"]={"status":"COMPILE FAILED - CANDIDATE BLOCKED","flag_gems_sha":fg_sha,
                              "triton":triton.__version__,"torch":torch.__version__,
                              "tile_params_from_exception":tiles[:8],
                              "static_assert_lines":[t for t in tiles if "static_assert" in t],
                              "blocked_shape":f"B{B} H{H} S{S_val} D{D} bf16 + float32 additive mask pad{pad}",
                              "minimal_repro":f"flag_gems.scaled_dot_product_attention(q[B,{H},{S_val},{D}] non-contig, k, v, attn_mask=float32[B,1,{S_val},{S_val}] -inf padded, dropout_p=0.0)",
                              "full_exception_tail":tb[-1200:]}
    json.dump(res,open(a.out,"w"),indent=1)
    for sk,e in res["shapes"].items():
        print(f"{sk}: indep {e['independence']} | fwd/bwd/full p0 {e['fwd_p0_ms']}/{e['bwd_p0_ms']}/{e['full_p0_ms']} p01 {e['fwd_p01_ms']}/{e['bwd_p01_ms']}/{e['full_p01_ms']}")
        print(f"   native p0 kernels: {[k[:60] for k in e['native_kernels_p0']]}")
        print(f"   native p01 kernels: {[k[:60] for k in e['native_kernels_p01']]}")
        print(f"   entry: {e['entry_status']} | direct: {e['direct_call']['status']} {e['direct_call'].get('static_assert_lines','')}")
    print("only_enable:", only_msg[:120])


if a.shape == "":
    import subprocess, os
    merged={"actual_model_call":None,"shapes":{},"only_enable_message":None,"per_shape_subprocess":True}
    for sk in SHAPE_MAP:
        tmp=a.out+"."+sk+".json"
        env=dict(os.environ); env["PYTHONPATH"]=env.get("PYTHONPATH","")+":/tmp/opencode/FlagGems-base/src"
        r=subprocess.run([sys.executable,__file__,"--out",tmp,"--shape",sk],env=env,capture_output=True,text=True,timeout=1200)
        print(f"[{sk}] exit={r.returncode} | {r.stdout.strip()[-300:]}", flush=True)
        if r.returncode!=0: print(f"[{sk}] STDERR tail: {r.stderr.strip()[-500:]}", flush=True)
        if os.path.exists(tmp):
            d=json.load(open(tmp)); merged["shapes"].update(d.get("shapes",{}))
            merged["actual_model_call"]=merged["actual_model_call"] or d.get("actual_model_call")
            merged["only_enable_message"]=merged["only_enable_message"] or d.get("only_enable_message")
    json.dump(merged,open(a.out,"w"),indent=1)
    for sk,e in merged["shapes"].items():
        print(f"{sk}: entry={e.get('entry_status')} | direct={e.get('direct_call',{}).get('status')}")
else:
    run_one(a.shape, a.out)
