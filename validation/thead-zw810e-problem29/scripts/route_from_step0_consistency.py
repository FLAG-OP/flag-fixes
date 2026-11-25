"""Decomposition of a discrepancy found in round 57: is the dW Python route numerically
identical to the entry when it is ACTIVE FROM STEP 0 (i.e. during the warmup steps too)?

Why this test exists
  * same-snapshot single-step test (train_state_consistency.py): dw_route vs native is
    bitwise identical for all 386 dense gradient tensors.
  * but with the route registered BEFORE the warmup, the state after 2 warmup steps differed
    from the native warmup (323/388 dense parameter tensors, max_abs 6.1e-05), and two
    independent processes gave different hit-step losses (0.945996 vs 0.950732).
  * five native processes reproduced the native warmup state (dense sha256 b133cb9f...).

--mode native_from_start / route_from_start (fresh process, one arm): 2 warmup steps, then the hit step on
batches 8-11; run several processes to see whether the state after the warmup is reproducible.

Design (decompose) - ONE process, five arms, every arm starts from the same initial snapshot S0 (model,
optimizer state, CPU+CUDA RNG) and consumes the same batches:
  A native            entry path
  B native again      determinism band of the entry path itself
  C fallback only     aten::mm override active, whitelisted shapes also go to mm.out (no flag_gems)
  D route             whitelisted dW shapes -> flag_gems.mm, everything else -> mm.out
  E route again       repeatability of D inside one process
Per warmup step the accumulated dense gradients (before clip) are compared, and after the two
steps the dense parameters and optimizer state. The three embedding tables are excluded
(index-accumulated gradient, not reproducible on this device: reported separately).
Registration is permanent in a process, so A and B run first and D/E are switched on by a flag.
"""
import argparse, copy, hashlib, json, os, sys
import torch
ap=argparse.ArgumentParser()
ap.add_argument("--conf",required=True); ap.add_argument("--out",required=True); ap.add_argument("--steps",type=int,default=2)
ap.add_argument("--mode",default="decompose",choices=["decompose","native_from_start","route_from_start"],
                help="native_from_start / route_from_start: fresh process, 2 warmup steps, then the hit step (batches 8-11) - run several processes and compare")
a=ap.parse_args()
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE","cuda"); os.environ["FL_ADAMW_FUSED"]="0"
sys.path.insert(0,"/tmp/opencode/FlagGems-base/src"); import flag_gems
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model
MM_OUT=torch.ops.aten.mm.out; FG=flag_gems.mm
N_MK=[(4096,5120,1024),(1024,5120,4096),(4096,4430,1024),(1024,4430,4096),(4096,4310,1024),(1024,4310,4096),
      (4096,3740,1024),(1024,3740,4096),(4096,3290,1024),(1024,3290,4096),(4096,2990,1024),(1024,2990,4096)]
DW={((N,M),(M,K)) for (N,M,K) in N_MK}
USE_FG=False; HITS={}
def route(self,other):
    k=(tuple(self.shape),tuple(other.shape))
    if k in DW:
        HITS[str(k)]=HITS.get(str(k),0)+1
        if USE_FG: return FG(self,other)
    return MM_OUT(self,other,out=torch.empty(self.shape[0],other.shape[1],device=self.device,dtype=self.dtype))
cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
TEMP=float(cfg.temperature); GN=float(cfg.optimizer.get("max_grad_norm",1.0))
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":"cuda"}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev); model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
EMB=set(n for n,_ in model.named_parameters() if n.endswith(("word_embeddings.weight","position_embeddings.weight","token_type_embeddings.weight")))
def micro_step(b):
    with autocast_context(dev,"bf16"):
        raw=loss_for_batch(model,processor,str(cfg.task),b,dev,TEMP); loss=raw/accum
    if not torch.isfinite(raw): raise FloatingPointError("Non-finite training loss")
    loss.backward(); return float(raw.detach())
def snap(): return {"model":copy.deepcopy(model.state_dict()),"opt":copy.deepcopy(optimizer.state_dict()),
                    "rng_cpu":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state()}
def restore(s):
    model.load_state_dict(copy.deepcopy(s["model"])); optimizer.load_state_dict(copy.deepcopy(s["opt"]))
    torch.set_rng_state(s["rng_cpu"]); torch.cuda.set_rng_state(s["rng_cuda"])
def to_cpu(t): return t.detach().to("cpu")
def arm(nsteps):
    restored_param_sha=digest({n:to_cpu(p) for n,p in model.named_parameters() if n not in EMB})
    stream=iter(loaders[0]); grads=[]; losses=[]; emb_bad=0
    for _ in range(nsteps):
        optimizer.zero_grad(set_to_none=True)
        batches=[next(stream) for _ in range(accum)]; losses.append([micro_step(b) for b in batches])
        g={n:to_cpu(p.grad) for n,p in model.named_parameters() if p.grad is not None and n not in EMB}
        emb_bad+=sum(int((~p.grad.detach().float().isfinite()).sum()) for n,p in model.named_parameters() if p.grad is not None and n in EMB)
        grads.append(g); torch.nn.utils.clip_grad_norm_(trainable,GN); optimizer.step()
    params={n:to_cpu(p) for n,p in model.named_parameters() if n not in EMB}
    state={n+"."+k:(to_cpu(v) if torch.is_tensor(v) else v) for n,p in model.named_parameters() if n not in EMB for k,v in optimizer.state.get(p,{}).items()}
    return {"grads":grads,"losses":losses,"params":params,"state":state,"emb_grad_nonfinite":emb_bad,"restored_param_sha":restored_param_sha,
            "first_batch_sha":hashlib.sha256("|".join("|".join(batches[0][k]) for k in sorted(batches[0])).encode()).hexdigest()[:16]}
def cmp(Da,Db):
    ma=mr=0.0; nf=0; n=diff=0; nt=0; names=[]
    for k in sorted(Da):
        x,y=Da[k],Db[k]
        if torch.is_tensor(x) and torch.is_tensor(y):
            n+=1
            if not (x.dtype==y.dtype and torch.equal(x,y)):
                diff+=1; e=(x.float()-y.float()).abs(); fl=0.01*float(y.float().abs().max())
                r=float((e/torch.clamp(y.float().abs(),min=fl)).max()) if y.numel() else 0.0
                ma=max(ma,float(e.max())); mr=max(mr,r); names.append((round(r,4),k))
            nf+=int((~torch.isfinite(x.float())).sum())+int((~torch.isfinite(y.float())).sum())
        elif x!=y: nt+=1
    names.sort(reverse=True)
    return {"tensors":n,"bitwise_identical":n-diff,"differing":diff,"max_abs":ma,"max_rel_floored":mr,"over_2pct":sum(r>0.02 for r,_ in names),
            "nonfinite":nf,"nontensor_mismatch":nt,"top_differing":[k for _,k in names[:5]]}
def digest(D):
    h=hashlib.sha256()
    for k in sorted(D):
        v=D[k]; h.update(k.encode()); h.update(v.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() if torch.is_tensor(v) else str(v).encode())
    return h.hexdigest()
def single(mode):
    if mode=="route_from_start":
        USE_FG=True; lib=torch.library.Library("aten","IMPL"); lib.impl("mm",route,"CUDA")
    wl=arm(a.steps); P=wl["params"]; S=wl["state"]
    stream=iter(loaders[0]); [next(stream) for _ in range(a.steps*accum)]; optimizer.zero_grad(set_to_none=True)
    hl=[micro_step(next(stream)) for _ in range(accum)]; gn=float(torch.nn.utils.clip_grad_norm_(trainable,GN))
    g={n:to_cpu(p.grad) for n,p in model.named_parameters() if p.grad is not None and n not in EMB}
    out={"field_version":"route_from_step0_v2","mode":mode,"warmup_steps":a.steps,"config":"bge_m3_ppu.yaml, seed 42, FL_ADAMW_FUSED=0, fresh process",
         "warmup_losses":wl["losses"],"params_dense_sha256_after_warmup":digest(P),"optimizer_state_dense_sha256_after_warmup":digest(S),
         "hit_step_micro_losses":hl,"hit_step_grad_norm":gn,"route_hits":dict(HITS),"hit_step_dense_grad_sha256":digest(g)}
    json.dump(out,open(a.out,"w"),indent=1)
    print(mode,"-> warmup sha",out["params_dense_sha256_after_warmup"][:12],"| hit step loss0 %.6f"%hl[0],"| grad norm",gn,"| hits",sum(HITS.values()))
if a.mode!="decompose": single(a.mode); sys.exit(0)
S0=snap(); R={}; HITS_PER_ARM={}
def run_arm(name):
    if R: restore(S0)
    HITS.clear(); R[name]=arm(a.steps); HITS_PER_ARM[name]=dict(HITS)
run_arm("A_native"); run_arm("B_native_again")
_LIB=torch.library.Library("aten","IMPL"); _LIB.impl("mm",route,"CUDA"); USE_FG=False
run_arm("C_fallback_only")
USE_FG=True
run_arm("D_route"); run_arm("E_route_again")
for name in R: assert R[name]["restored_param_sha"]==R["A_native"]["restored_param_sha"], "restore failed"
def pair(X,Y):
    x,y=R[X],R[Y]
    return {"losses_equal_per_step":[lx==ly for lx,ly in zip(x["losses"],y["losses"])],
            "loss_max_abs_per_step":[max(abs(p-q) for p,q in zip(lx,ly)) for lx,ly in zip(x["losses"],y["losses"])],
            "grads_dense_per_step":[cmp(x["grads"][i],y["grads"][i]) for i in range(a.steps)],
            "params_dense":cmp(x["params"],y["params"]),"optimizer_state_dense":cmp(x["state"],y["state"]),
            "params_dense_sha_equal":digest(x["params"])==digest(y["params"]),"optimizer_state_sha_equal":digest(x["state"])==digest(y["state"])}
res={"field_version":"route_from_step0_v2","design":__doc__,"warmup_steps":a.steps,"config":"bge_m3_ppu.yaml, seed 42, FL_ADAMW_FUSED=0",
     "hits_per_arm":HITS_PER_ARM,
     "arms":{k:{"losses":v["losses"],"first_batch_sha":v["first_batch_sha"],"restored_param_sha":v["restored_param_sha"],"params_dense_sha256":digest(v["params"]),"optimizer_state_dense_sha256":digest(v["state"]),"emb_grad_nonfinite":v["emb_grad_nonfinite"]} for k,v in R.items()},
     "pairs":{p:pair(*p.split("_vs_")) for p in ("A_native_vs_B_native_again","A_native_vs_C_fallback_only","A_native_vs_D_route","A_native_vs_E_route_again","D_route_vs_E_route_again","C_fallback_only_vs_D_route")},
     "flag_gems_source":os.path.abspath(flag_gems.__file__),"mm_source":getattr(FG,"__code__",None) and FG.__code__.co_filename,"mm_first_line":getattr(FG,"__code__",None) and FG.__code__.co_firstlineno}
json.dump(res,open(a.out,"w"),indent=1)
for k,v in res["arms"].items(): print(f"ARM {k:<18} first batch {v['first_batch_sha']} restored sha {v['restored_param_sha'][:12]} loss(step0,micro0)={v['losses'][0][0]:.6f} loss(step1,micro0)={v['losses'][1][0]:.6f} params_sha={v['params_dense_sha256'][:12]} state_sha={v['optimizer_state_dense_sha256'][:12]}")
for k,p in res["pairs"].items():
    print(f"{k:<32} step0 grads: identical {p['grads_dense_per_step'][0]['bitwise_identical']}/{p['grads_dense_per_step'][0]['tensors']} max_abs {p['grads_dense_per_step'][0]['max_abs']:.3g} | step1 grads: identical {p['grads_dense_per_step'][1]['bitwise_identical']}/{p['grads_dense_per_step'][1]['tensors']} max_abs {p['grads_dense_per_step'][1]['max_abs']:.3g} | params identical {p['params_dense']['bitwise_identical']}/{p['params_dense']['tensors']} max_abs {p['params_dense']['max_abs']:.3g} | state max_abs {p['optimizer_state_dense']['max_abs']:.3g} | losses equal {p['losses_equal_per_step']}")
