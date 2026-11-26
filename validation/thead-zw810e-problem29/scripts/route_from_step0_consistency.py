"""Is the dW Python route numerically identical to the entry when it is ACTIVE FROM STEP 0, and is
it reproducible across processes? (review 7182886976 item 2)

Round-57 bug fixed here: in the previous version the "route_from_start" mode set the enable flag
as a FUNCTION-LOCAL variable, so the override kept reading the module-level flag (False) and every
whitelisted call fell through to the native mm.out. The three route_from_start processes of
round 57 therefore never executed flag_gems.mm and their numbers say nothing about FlagGems.
In this version the switch is a module-level global that is only ever assigned at module level or
through an explicit `global` statement, three counters are kept separately
(whitelist matches / actual flag_gems.mm calls / native fallbacks), and a run that requests the
route but never reaches flag_gems.mm exits with code 3 and writes nothing.

Modes (one fresh process each; run several processes per mode and compare with --aggregate):
  native_from_start    no override                       (entry path)
  fallback_from_start  aten::mm override, whitelisted shapes ALSO go to mm.out (no flag_gems)
  route_from_start     whitelisted dW shapes -> flag_gems.mm, everything else -> mm.out
Every process: same seed, 2 warmup steps on batches 0-7 (fresh data iterator), then the hit step
on batches 8-11 (new iterator, same order). Per step the batch hashes, the micro losses and the
three call counters are recorded; after the warmup the dense parameter / optimizer-state hashes.

--copy-grads host|none : "host" copies the accumulated dense gradients to the host before clip
(hashes them, optionally saves them); "none" does not touch the gradients before clip. The two
differ only in host synchronization around the step and are used to test whether that alone
changes run-to-run reproducibility.

--aggregate <glob> : loads the per-process JSON files (and the saved gradients when present),
compares every pair of processes per step (dense gradients) and summarizes by pair class
(native vs native = the self-comparison band of the entry path, route vs native, fallback vs
native, ...), plus the per-group distributions of hit-step loss and grad norm.
"""
import argparse, collections, copy, glob, hashlib, json, math, os, statistics, sys, time
import torch
ap=argparse.ArgumentParser()
ap.add_argument("--conf"); ap.add_argument("--out"); ap.add_argument("--steps",type=int,default=2)
ap.add_argument("--mode",default="native_from_start",choices=["native_from_start","fallback_from_start","route_from_start"])
ap.add_argument("--copy-grads",default="host",choices=["host","none"]); ap.add_argument("--save-grads",default=None)
ap.add_argument("--aggregate",default=None,help="glob of per-process json files -> cross-process comparison json (no device needed)")
a=ap.parse_args()
ORDER=["native_from_start","fallback_from_start","route_from_start"]
def cmpD(Da,Db):
    n=diff=nf=over=0; ma=mr=0.0; top=[]
    for k in sorted(Da):
        x,y=Da[k],Db[k]; n+=1
        if x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y): continue
        diff+=1; p=x.float(); q=y.float(); e=(p-q).abs(); fl=0.01*float(q.abs().max())
        r=float((e/torch.clamp(q.abs(),min=fl)).max()) if q.numel() else 0.0; m=float(e.max())
        ma=max(ma,m); mr=max(mr,r); over+=(r>0.02); nf+=int((~torch.isfinite(p)).sum())+int((~torch.isfinite(q)).sum()); top.append((m,r,k))
    top.sort(reverse=True)
    return {"tensors":n,"bitwise_identical":n-diff,"n_differing":diff,"max_abs":ma,"max_rel":mr,"over_2pct":over,"nonfinite":nf,"top_differing":[k for _,_,k in top[:3]]}
def d3(v): v=sorted(v); return {"min":round(v[0],6),"median":round(statistics.median(v),6),"max":round(v[-1],6)} if v else None
def aggregate(pattern,out):
    """cross-process comparison. Pair classes are summarized PER STEP: step 0 starts from the identical
    initial model / batches / RNG in every process (forward identical), later steps also carry the
    divergence accumulated in the preceding steps."""
    files=sorted(glob.glob(pattern)); R=[json.load(open(f)) for f in files]
    def saved(i,s): return os.path.exists(os.path.join(os.path.dirname(files[i]),f"grads_step{s}.pt"))
    hit=a.steps; steps=sorted({s for i in range(len(R)) for s in list(range(a.steps))+[hit] if saved(i,s)})
    short=lambda m:m.replace("_from_start","")
    acc=collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list))); detail=collections.defaultdict(lambda: collections.defaultdict(list))
    for s in steps:
        G={i:torch.load(os.path.join(os.path.dirname(files[i]),f"grads_step{s}.pt")) for i in range(len(R)) if saved(i,s)}
        keys=sorted(G)
        for ii in range(len(keys)):
            for jj in range(ii+1,len(keys)):
                i,j=keys[ii],keys[jj]; mi,mj=R[i]["mode"],R[j]["mode"]
                if ORDER.index(mi)>ORDER.index(mj): i,j,mi,mj=j,i,mj,mi
                c=cmpD(G[i],G[j]); cls=f"{short(mi)}_vs_{short(mj)}"
                for k in ("n_differing","max_abs","max_rel","over_2pct"): acc[cls][str(s)][k].append(c[k])
                detail[cls][str(s)].append({"a":os.path.basename(files[i]),"b":os.path.basename(files[j]),"n_differing":c["n_differing"],"max_abs":round(c["max_abs"],8),"max_rel":round(c["max_rel"],5),"over_2pct":c["over_2pct"]})
        del G
    summ={cls:{st:({k:d3(v) for k,v in d.items()}|{"n_pairs":len(d["n_differing"])}) for st,d in steps_d.items()} for cls,steps_d in acc.items()}
    grp={}
    for m in ORDER:
        rs=[r for r in R if r["mode"]==m]
        if not rs: continue
        gn=[r["hit_step"]["grad_norm"] for r in rs]
        grp[m]={"n_processes":len(rs),"distinct_initial_param_sha256":len({r["initial_param_sha256_dense"] for r in rs}),
                "step0_micro0_losses":sorted({round(r["steps"][0]["losses"][0],9) for r in rs}),
                "step0_losses_identical_across_processes":len({json.dumps(r["steps"][0]["losses"]) for r in rs})==1,
                "step_batch_hashes_first_process":[x for s_ in rs[0]["steps"] for x in s_["batch_hashes"]]+rs[0]["hit_step"]["batch_hashes"],
                "distinct_warmup_param_sha256":len({r["params_dense_sha256_after_warmup"] for r in rs}),"distinct_warmup_state_sha256":len({r["state_dense_sha256_after_warmup"] for r in rs}),
                "hit_step_micro0_loss":sorted(round(r["hit_step"]["losses"][0],6) for r in rs),"hit_step_grad_norm":sorted(gn),
                "hit_step_grad_norm_range":[min(gn),max(gn)],
                "call_counts_total":{k:sum(s_["calls"][k] for r in rs for s_ in r["steps"])+sum(r["hit_step"]["calls"][k] for r in rs) for k in ("whitelist_match","flag_gems_mm","native_fallback")},
                "per_step_calls_first_process":[s_["calls"] for s_ in rs[0]["steps"]]+[rs[0]["hit_step"]["calls"]]}
    res={"field_version":"route_from_step0_v3","n_processes":len(R),"copy_grads":sorted({r["copy_grads"] for r in R}),"files":[os.path.basename(f) for f in files],
         "steps_compared":steps,"step_labels":{str(s):("hit step" if s==hit else f"warmup step {s}") for s in steps},"groups":grp,"pair_classes_by_step":summ,"pair_detail":detail}
    if summ:
        parts=[]
        for s in steps:
            row=lambda cls: summ.get(cls,{}).get(str(s))
            nv,fb,rv=row("native_vs_native"),row("native_vs_fallback"),row("native_vs_route")
            fmt=lambda X:(f"tensors differing {X['n_differing']['min']}-{X['n_differing']['max']} (median {X['n_differing']['median']}/386), max_abs {X['max_abs']['min']:.2g}-{X['max_abs']['max']:.2g}" if X else "n/a")
            parts.append(f"{res['step_labels'][str(s)]}: native-vs-native [{fmt(nv)}] ; fallback-vs-native [{fmt(fb)}] ; route-vs-native [{fmt(rv)}]")
            if s==0 and nv and rv:
                res["step0_native_vs_route_within_native_range"]=bool(rv["n_differing"]["min"]>=nv["n_differing"]["min"]*0.95 and rv["n_differing"]["max"]<=nv["n_differing"]["max"]*1.05 and rv["max_abs"]["max"]<=max(nv["max_abs"]["max"]*1.5,1e-6))
        res["reading"]=("dense accumulated gradients of two processes, per step. Step 0 starts from the identical initial model / batches / RNG in every process (forward identical). "+ " | ".join(parts))
        def ranks(a,b):   # Mann-Whitney one-sided P(all a < all b) when the ranges do not overlap
            lo_a,hi_a=min(a),max(a); lo_b,hi_b=min(b),max(b)
            return {"a_range":[lo_a,hi_a],"b_range":[lo_b,hi_b],"ranges_overlap":not (hi_a<lo_b or hi_b<lo_a)}
        if "native_from_start" in grp and "route_from_start" in grp:
            res["hit_step_grad_norm_ranges"]={"route":ranks(grp["route_from_start"]["hit_step_grad_norm"],grp["native_from_start"]["hit_step_grad_norm"]),
                                              "fallback":ranks(grp["fallback_from_start"]["hit_step_grad_norm"],grp["native_from_start"]["hit_step_grad_norm"]) if "fallback_from_start" in grp else None}
    json.dump(res,open(out,"w"),indent=1)
    for m,g in grp.items(): print(f"{m:<22} n {g['n_processes']} | distinct init sha {g['distinct_initial_param_sha256']} | step0 losses identical {g['step0_losses_identical_across_processes']} | distinct warmup param sha {g['distinct_warmup_param_sha256']} | hit gn {g['hit_step_grad_norm']} | calls {g['call_counts_total']}")
    for cls,st in summ.items():
        for s,X in st.items(): print(f"  {cls:<22} {res['step_labels'][s]:<13} pairs {X['n_pairs']:>3} | n_differing {X['n_differing']['min']}-{X['n_differing']['max']} med {X['n_differing']['median']} | max_abs {X['max_abs']['min']:.2g}-{X['max_abs']['max']:.2g} | over2% {X['over_2pct']['min']}-{X['over_2pct']['max']}")
    if "reading" in res: print(res["reading"][:1500]); print("step0 route-vs-native within native range:",res.get("step0_route_vs_native_within_native_range"))
if a.aggregate: aggregate(a.aggregate,a.out); sys.exit(0)
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
MODE="native"                     # module-level ONLY: "native" | "fallback" | "route"
CNT={"whitelist_match":0,"flag_gems_mm":0,"native_fallback":0}
def route(self,other):
    k=(tuple(self.shape),tuple(other.shape))
    if k in DW:
        CNT["whitelist_match"]+=1
        if MODE=="route":
            CNT["flag_gems_mm"]+=1; return FG(self,other)
    CNT["native_fallback"]+=1
    return MM_OUT(self,other,out=torch.empty(self.shape[0],other.shape[1],device=self.device,dtype=self.dtype))
MODE={"native_from_start":"native","fallback_from_start":"fallback","route_from_start":"route"}[a.mode]
_R=None
if MODE!="native":
    _R=torch.library.Library("aten","IMPL"); _R.impl("mm",route,"CUDA")       # object kept alive: a garbage-collected Library silently drops the registration
cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
TEMP=float(cfg.temperature); GN=float(cfg.optimizer.get("max_grad_norm",1.0))
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":"cuda"}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev); model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
EMB=set(n for n,_ in model.named_parameters() if n.endswith(("word_embeddings.weight","position_embeddings.weight","token_type_embeddings.weight")))
def to_cpu(t): return t.detach().to("cpu")
def digest(D):
    h=hashlib.sha256()
    for k in sorted(D):
        v=D[k]; h.update(k.encode()); h.update(v.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() if torch.is_tensor(v) else str(v).encode())
    return h.hexdigest()
def bhash(b): return hashlib.sha256("|".join("|".join(b[k]) for k in sorted(b)).encode()).hexdigest()[:16]
def micro_step(b):
    with autocast_context(dev,"bf16"):
        raw=loss_for_batch(model,processor,str(cfg.task),b,dev,TEMP); loss=raw/accum
    if not torch.isfinite(raw): raise FloatingPointError("Non-finite training loss")
    loss.backward(); return float(raw.detach())
def reset_cnt(): CNT.update(whitelist_match=0,flag_gems_mm=0,native_fallback=0)
def dense_grads(): return {n:to_cpu(p.grad) for n,p in model.named_parameters() if p.grad is not None and n not in EMB}
def run_step(batches,step_tag):
    reset_cnt(); optimizer.zero_grad(set_to_none=True)
    losses=[micro_step(b) for b in batches]
    rec={"losses":losses,"batch_hashes":[bhash(b) for b in batches]}
    if a.copy_grads=="host":
        g=dense_grads(); rec["dense_grad_sha256"]=digest(g)
        if a.save_grads: torch.save(g,os.path.join(a.save_grads,f"grads_{step_tag}.pt"))
        del g
    rec["calls"]=dict(CNT)
    gn=float(torch.nn.utils.clip_grad_norm_(trainable,GN)); optimizer.step(); rec["grad_norm"]=gn
    rec["after_step_params_sha256_all"]=digest({n:to_cpu(p) for n,p in model.named_parameters()})
    rec["after_step_params_sha256_dense"]=digest({n:to_cpu(p) for n,p in model.named_parameters() if n not in EMB})
    rec["after_step_rng_cuda_sha256"]=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()[:16]
    rec["after_step_rng_cpu_sha256"]=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16]
    return rec
started=time.time()
initial=digest({n:to_cpu(p) for n,p in model.named_parameters() if n not in EMB})
steps=[]; stream=iter(loaders[0])
for s in range(a.steps):
    b=[next(stream) for _ in range(accum)]; steps.append(run_step(b,f"step{s}"))
P={n:to_cpu(p) for n,p in model.named_parameters() if n not in EMB}
S={n+"."+k:to_cpu(v) for n,p in model.named_parameters() if n not in EMB for k,v in optimizer.state.get(p,{}).items() if torch.is_tensor(v)}
pw,sw=digest(P),digest(S); del P,S
stream=iter(loaders[0]); [next(stream) for _ in range(a.steps*accum)]
hit=run_step([next(stream) for _ in range(accum)],f"step{a.steps}")
tot={k:sum(s["calls"][k] for s in steps)+hit["calls"][k] for k in CNT}
res={"field_version":"route_from_step0_v3","mode":a.mode,"route_enabled":MODE,"copy_grads":a.copy_grads,"pid":os.getpid(),"started_unix":started,"finished_unix":time.time(),
     "config":"bge_m3_ppu.yaml, seed 42, FL_ADAMW_FUSED=0, fresh process; warmup %d steps on a fresh iterator, hit step on batches %d-%d of a new iterator"%(a.steps,a.steps*accum,a.steps*accum+accum-1),
     "initial_param_sha256_dense":initial,"steps":steps,"params_dense_sha256_after_warmup":pw,"state_dense_sha256_after_warmup":sw,
     "hit_step":{k:hit[k] for k in ("losses","batch_hashes","calls","grad_norm","after_step_params_sha256_all","after_step_params_sha256_dense","after_step_rng_cuda_sha256","after_step_rng_cpu_sha256") if k in hit}|({"dense_grad_sha256":hit["dense_grad_sha256"]} if "dense_grad_sha256" in hit else {}),"call_counts_total":tot,
     "grads_saved":bool(a.save_grads),"flag_gems_mm_source":FG.__code__.co_filename,"flag_gems_mm_first_line":FG.__code__.co_firstlineno}
if MODE=="route" and tot["flag_gems_mm"]==0:
    print("ROUTE REQUESTED BUT flag_gems.mm WAS NEVER CALLED",tot); sys.exit(3)
if MODE!="route": assert tot["flag_gems_mm"]==0,tot
if MODE=="route": assert tot["flag_gems_mm"]==tot["whitelist_match"],tot
if MODE=="fallback": assert tot["flag_gems_mm"]==0 and tot["whitelist_match"]>0,tot
json.dump(res,open(a.out,"w"),indent=1)
print(a.mode,"-> calls",tot,"| warmup param sha",pw[:12],"state sha",sw[:12],"| hit loss0 %.6f"%hit["losses"][0],"| grad norm",hit["grad_norm"],"| per-step calls",[s["calls"] for s in steps+[hit]])
