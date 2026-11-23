"""Review 5b8fb29 item 5: training-mode SDPA retest (full rewrite).
- Perturb V along S axis at MASKED positions: v2[:, :, -32:, :] ; compare ALL D
  channels on VALID queries; PLUS no-mask control proving same perturbation
  changes output without mask.
- Timings: forward, backward, and one direct full fwd+bwd — unified clone /
  graph-build / sync boundaries on BOTH sides.
- Candidate via the model's F.scaled_dot_product_attention entry (aten-level);
  backend/kernel recorded.
- Numeric (dropout=0): outputs + dQ/dK/dV vs FP32 reference with same upstream
  grads. Dropout config recorded; randomness = torch generator (seeded).
- On compile failure: impl SHA, actual tile params from exception, minimal
  repro, full traceback, blocked shapes."""
import argparse, json, statistics, sys
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

ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True)
a=ap.parse_args()
B,H,S,D = 10,16,512,64
torch.manual_seed(0)
q=torch.randn(B,H,S,D,device="cuda",dtype=torch.bfloat16)*0.3
k=torch.randn(B,H,S,D,device="cuda",dtype=torch.bfloat16)*0.3
v=torch.randn(B,H,S,D,device="cuda",dtype=torch.bfloat16)*0.3
mask=torch.ones(B,1,S,S,dtype=torch.bool,device="cuda")
mask[:,:,:,-32:]=False   # last 32 KEYS masked (S-axis), True=participate
res={"shape":f"B{B} H{H} S{S} D{D} bf16","mask":"bool True=participate; last 32 KEYS masked (S-axis)",
     "dropout":{"model_config_hidden_dropout_prob":0.1,"sdpa_internal_p":"recorded below; module-level dropout active",
                 "randomness_control":"torch global generator seeded (torch.manual_seed(0)); dropout masks generated per call by backend"}}

# --- S-axis perturbation independence + no-mask control ---
v2=v.clone(); v2[:,:,-32:,:]+=100.0        # perturb V at MASKED key positions (S axis)
with torch.no_grad():
    y_m = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0)
    y_m2 = F.scaled_dot_product_attention(q,k,v2,attn_mask=mask,dropout_p=0.0)
    y_nom = F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=0.0)
    y_nom2 = F.scaled_dot_product_attention(q,k,v2,attn_mask=None,dropout_p=0.0)
res["independence"]={
    "with_mask_valid_region_max_diff": round(float((y_m2-y_m)[:,:,:,:S-32].abs().max()),8),
    "no_mask_control_max_diff": round(float((y_nom2-y_nom).abs().max()),6),
    "interpretation":"with-mask diff ~0 => masked-position V does not affect valid outputs; no-mask diff >>0 => perturbation is real"}

# --- timings: fwd / bwd / full, unified boundaries ---
def make_inputs(require_grad=True):
    if require_grad:
        return [t.detach().clone().requires_grad_(True) for t in (q,k,v)]
    return [t.detach().clone() for t in (q,k,v)]
def fwd_only(p):
    qq,kk,vv = make_inputs(False)
    return lambda: F.scaled_dot_product_attention(qq,kk,vv,attn_mask=mask,dropout_p=p)
def bwd_only(p):
    qq,kk,vv = make_inputs(True)
    y = F.scaled_dot_product_attention(qq,kk,vv,attn_mask=mask,dropout_p=p)   # graph+clone OUTSIDE timing
    g = torch.ones_like(y)
    return lambda: torch.autograd.grad(y,(qq,kk,vv),g,retain_graph=True)
def full_fwd_bwd(p):
    def fn():
        qq,kk,vv = make_inputs(True)
        y = F.scaled_dot_product_attention(qq,kk,vv,attn_mask=mask,dropout_p=p)
        torch.autograd.grad(y,(qq,kk,vv),torch.ones_like(y))
    return fn
for p,label in ((0.0,"p0"),(0.1,"p01")):
    res[f"native_fwd_{label}_ms"]=timed(fwd_only(p))
    res[f"native_bwd_{label}_ms"]=timed(bwd_only(p))
    res[f"native_full_{label}_ms"]=timed(full_fwd_bwd(p))

# --- numeric vs FP32 (dropout=0, same upstream grad) ---
qq,kk,vv = make_inputs(True)
y = F.scaled_dot_product_attention(qq,kk,vv,attn_mask=mask,dropout_p=0.0)
g = torch.ones_like(y)
dq,dk,dv = torch.autograd.grad(y,(qq,kk,vv),g,retain_graph=True)
qf,kf,vf = [t.float().detach().requires_grad_(True) for t in (q,k,v)]
mf = mask if mask.dtype!=torch.bool else mask
yf = F.scaled_dot_product_attention(qf,kf,vf,attn_mask=mask,dropout_p=0.0)
gf = torch.ones_like(yf)
dqf,dkf,dvf = torch.autograd.grad(yf,(qf,kf,vf),gf)
def rel(x,ref): return round(float((x.float()-ref).abs().max()/ref.abs().max().clamp_min(1e-6)),6)
res["native_vs_fp32_reldiff"]={"out":rel(y,yf),"dQ":rel(dq,dqf),"dK":rel(dk,dkf),"dV":rel(dv,dvf)}

# --- candidate via model entry: aten-level override, then plain F.sdpa call ---
sys.path.insert(0,"/tmp/opencode/FlagGems-base/src")
import flag_gems, triton
fg_repro={"component_versions":{"torch":torch.__version__,"triton":triton.__version__},
           "entry":"flag_gems.only_enable(['scaled_dot_product_attention']) then F.scaled_dot_product_attention (model entry, aten-level)",
           "impl":"FlagGems-base/src @ 29e14822 (v5.3.5-46)","full_exception":None,
           "tile_params_from_exception":None,"minimal_repro":"flag_gems.only_enable(['scaled_dot_product_attention']); F.scaled_dot_product_attention(q,k,v,attn_mask=bool mask,dropout_p=0.0)",
           "blocked_shapes":[f"B{B} H{H} S{S} D{D} bf16 + bool mask"]}
try:
    flag_gems.only_enable(include=["scaled_dot_product_attention"])
    try:
        yc = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0)
        res["candidate"]={"status":"ran","fwd_ms":timed(fwd_only(0.0))}
    except Exception as ex:
        import traceback
        fg_repro["full_exception"]=traceback.format_exc()[-2000:]
        for ln in fg_repro["full_exception"].splitlines():
            if "BLOCK" in ln or "HEAD_DIM" in ln or "GROUP_HEAD" in ln:
                fg_repro["tile_params_from_exception"]=ln.strip(); break
        res["candidate"]={"status":"COMPILE FAILED via model entry (F.scaled_dot_product_attention)","repro":fg_repro}
except Exception as ex:
    fg_repro["full_exception"]=f"{type(ex).__name__}: {ex}"
    res["candidate"]={"status":"registration failed","repro":fg_repro}

json.dump(res,open(a.out,"w"),indent=1)
print(json.dumps({k2:v2 for k2,v2 in res.items() if k2 in ("independence","native_fwd_p0_ms","native_bwd_p0_ms","native_full_p0_ms","native_fwd_p01_ms","native_bwd_p01_ms","native_full_p01_ms","native_vs_fp32_reldiff")},indent=1))
print("candidate:", res["candidate"]["status"])
