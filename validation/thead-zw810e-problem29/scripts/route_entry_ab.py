"""dW Python route on the ORIGINAL entry loop, paired WITHIN one process.

Round-48 (route_isolated.py: -5.6 ms median, 5/6 faster) used one process per mode
and the r36 capture loop (no lookahead, no loss sync). The between-process step-time
noise measured in round-57 (tokenize_overlap_native_r57: h2d_nb sd 14 ms per pair)
is larger than that effect, so this script pairs differently:

  phase 1 : native path; fresh data iterator (seed 42): 2 warmup + 3 timed steps
            (steps 2-4, the r36 window)
  [register the aten::mm dW route if --second py_route]
  phase 2 : fresh data iterator with the SAME seed (identical batches, verified by
            the text hashes): 2 warmup + 3 timed steps

--second native is the control: its phase2 - phase1 difference is the pure
order effect (warmer process, memory state, clocks). The route effect is the
difference of differences: mean(py_route diff) - mean(native diff).

Step loop = the ORIGINAL entry (train_retrieval.py:646-673): one-batch lookahead,
loss under autocast, the entry's isfinite(loss) host sync, backward,
clip_grad_norm_ + optimizer.step. Profiler OFF, no experiment sync inside the loss
path; the step-end sync only closes the timed window. Tokens are counted after it.
"""
import argparse, collections, hashlib, json, math, os, sys, time
T_CRIT={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262,11:2.228,12:2.201,13:2.179,14:2.160,15:2.145,
        16:2.131,17:2.120,18:2.110,19:2.101,20:2.093,21:2.086,22:2.080,23:2.074,24:2.069,25:2.064,30:2.045,40:2.021,60:2.000}
def fisher_two_sided(a,b,c,d):
    n=a+b+c+d; r1=a+b; c1=a+c
    def q(x): return math.comb(r1,x)*math.comb(n-r1,c1-x)/math.comb(n,c1)
    p0=q(a); return sum(q(x) for x in range(max(0,c1-(n-r1)),min(r1,c1)+1) if q(x)<=p0*(1+1e-9))
def stats(xs):
    n=len(xs); s=sorted(xs); mean=sum(s)/n; med=s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
    sd=math.sqrt(sum((x-mean)**2 for x in s)/(n-1)) if n>1 else 0.0
    half=(T_CRIT.get(n,1.96) if n>1 else float("nan"))*sd/math.sqrt(n) if n>1 else float("nan")
    return {"n":n,"mean_ms":round(mean,2),"median_ms":round(med,2),"min_ms":s[0],"max_ms":s[-1],"sd_ms":round(sd,2),
            "mean_ci95_ms":[round(mean-half,2),round(mean+half,2)],"n_slower":sum(x>0 for x in s),"n_faster":sum(x<0 for x in s)}
def t_crit(df):
    """two-sided 95% critical value of Student's t for (possibly fractional) df: linear interpolation in 1/df between table entries"""
    ks=sorted(T_CRIT)
    if df>=ks[-1]: return T_CRIT[ks[-1]] if df<ks[-1]*1.5 else 1.96
    if df<=ks[0]: return T_CRIT[ks[0]]
    lo=max(k for k in ks if k<=df); hi=min(k for k in ks if k>=df); w=(1/df-1/lo)/(1/hi-1/lo)
    return T_CRIT[lo]*(1-w)+T_CRIT[hi]*w
def welch(A,B):
    """Welch 95% CI of mean(B)-mean(A); A = native control, B = py_route. The unit is whatever the caller passes
    (paired steps, or one mean per process)."""
    nA,nB=len(A),len(B); mA=sum(A)/nA; mB=sum(B)/nB
    vA=sum((x-mA)**2 for x in A)/(nA-1); vB=sum((x-mB)**2 for x in B)/(nB-1)
    se=math.sqrt(vA/nA+vB/nB); df=(vA/nA+vB/nB)**2/((vA/nA)**2/(nA-1)+(vB/nB)**2/(nB-1)); tc=t_crit(df)
    lo,hi=mB-mA-tc*se,mB-mA+tc*se
    return {"unit":None,"n_native":nA,"n_py_route":nB,"native_mean_ms":round(mA,2),"py_route_mean_ms":round(mB,2),"diff_ms":round(mB-mA,2),
            "native_sd_ms":round(math.sqrt(vA),2),"py_route_sd_ms":round(math.sqrt(vB),2),"se_ms":round(se,2),"welch_df":round(df,1),"t_crit":round(tc,3),
            "ci95_ms":[round(lo,2),round(hi,2)],"ci_excludes_zero":bool(lo>0 or hi<0),
            "process_means_negative":{"py_route":sum(x<0 for x in B),"native":sum(x<0 for x in A)} if nA==len(set(A)) or True else None}
def aggregate(files,out):
    runs=[json.load(open(f)) for f in files]
    by=collections.defaultdict(list); per=collections.defaultdict(list); pm=collections.defaultdict(dict); mismatch=0
    for r in runs:
        batch=r.get("batch","batch2")
        for a,b in zip(r["phase1"]["rounds"],r["phase2"]["rounds"]):
            if "n/a" not in a["input_hash"] and a["input_hash"]!=b["input_hash"]: mismatch+=1
            by[(batch,r["second"])].append(round(b["step_ms"]-a["step_ms"],2)); per[(batch,r["second"],r["run"])].append(round(b["step_ms"]-a["step_ms"],2))
        pm[(batch,r["second"])][r["run"]]=round(sum(per[(batch,r["second"],r["run"])])/len(per[(batch,r["second"],r["run"])]),2)
    def std(xs):
        n=len(xs); m=sum(xs)/n
        return {"n":n,"mean_ms":round(m,2),"median_ms":round(sorted(xs)[n//2] if n%2 else (sorted(xs)[n//2-1]+sorted(xs)[n//2])/2,2),"n_negative":sum(x<0 for x in xs),
                "min_ms":min(xs),"max_ms":max(xs)}
    def sel(batches): return [b for b in ("batch2","batch1_console") if b in batches]
    PROV={"batch2":"16 processes (8 py_route + 8 native control), per-run JSON files committed (results/route_entry_ab_r57/*.json)",
          "batch1_console":"12 processes (6 + 6); per-run JSON files were lost to a shell file-naming bug, step times transcribed from the console and committed as results/route_entry_ab_r57_batch1_console/*.json - NOT independent raw records"}
    levels={}
    for name,batches in (("all",("batch2","batch1_console")),("batch2_only",("batch2",)),("batch1_console_only",("batch1_console",))):
        if not all((b,"py_route") in pm for b in batches): continue
        sA=[x for b in batches for x in by[(b,"native")]]; sB=[x for b in batches for x in by[(b,"py_route")]]
        pA=[x for b in batches for x in pm[(b,"native")].values()]; pB=[x for b in batches for x in pm[(b,"py_route")].values()]
        levels[name]={"batches":list(batches),"provenance":{b:PROV[b] for b in batches},
            "step_level":{"native":std(sA),"py_route":std(sB),"welch":dict(welch(sA,sB),unit="paired step (phase2 - phase1)")},
            "process_level":{"native":std(pA),"py_route":std(pB),"welch":dict(welch(pA,pB),unit="process (mean of its 3 paired steps)")}}
    A=pm[("batch2","native")]; B=pm[("batch2","py_route")]
    for b in ("batch1_console",):
        if (b,"native") in pm: A={**A,**{k+"@b1":v for k,v in pm[(b,"native")].items()}}; B={**B,**{k+"@b1":v for k,v in pm[(b,"py_route")].items()}}
    neg={"py_route":sum(x<0 for x in B.values()),"n_py_route":len(B),"native":sum(x<0 for x in A.values()),"n_native":len(A)}
    sign={"process_mean_negative":neg,"fisher_two_sided_p":round(fisher_two_sided(neg["py_route"],neg["n_py_route"]-neg["py_route"],neg["native"],neg["n_native"]-neg["native"]),4)}
    pl=levels["all"]["process_level"]["welch"]; lo,hi=pl["ci95_ms"]
    cls="REGRESSION (slower)" if lo>0 else ("GAIN (faster)" if hi<0 else "CI SPANS ZERO")
    res={"field_version":"route_entry_ab_v3","time_unit":"ms",
         "estimate":"phase2 - phase1 per process (py_route registered between the phases; --second native = control without registration); effect = py_route minus native control. Reported at two levels: paired STEPS (treats the 3 steps of a process as independent) and PROCESSES (one mean per process) - the three steps of a process share the run environment, so the process level is the conservative one.",
         "unit_note":"the process level is the primary result; the step level is kept for comparison with round-57 (which pooled steps)",
         "levels":levels,"process_level_sign_test":sign,"process_mean_diffs_ms":{f"{k[0]}|{k[1]}":dict(sorted(v.items())) for k,v in sorted(pm.items())},
         "batch_hash_mismatch":mismatch,"raw_files":files,
         "valid_tokens_per_s":{k:{p:round(sum(x for r in runs if r["second"]==k and r.get("batch","batch2")=="batch2" for x in r["tokens_per_s_"+p])/max(1,sum(len(r["tokens_per_s_"+p]) for r in runs if r["second"]==k and r.get("batch","batch2")=="batch2")),1) for p in ("p1","p2")} for k in ("native","py_route")}}
    res["verdict"]=(f"PROCESS LEVEL (primary), all 28 processes: diff {pl['diff_ms']:+.2f} ms, Welch 95% CI [{lo:+.2f}, {hi:+.2f}] -> {cls}; "
                    f"batch 2 only (complete raw records, 16 processes): diff {levels['batch2_only']['process_level']['welch']['diff_ms']:+.2f} ms, CI {levels['batch2_only']['process_level']['welch']['ci95_ms']}; "
                    f"batch 1 only (console transcription, 12 processes): diff {levels['batch1_console_only']['process_level']['welch']['diff_ms']:+.2f} ms, CI {levels['batch1_console_only']['process_level']['welch']['ci95_ms']}; "
                    f"process means negative in {neg['py_route']}/{neg['n_py_route']} py_route vs {neg['native']}/{neg['n_native']} native processes. "
                    + ("The interval spans zero: there is a small indication of a gain, not enough to deploy." if cls=="CI SPANS ZERO" else ""))
    json.dump(res,open(out,"w"),indent=1); print(res["verdict"]); return
ap=argparse.ArgumentParser()
ap.add_argument("--conf"); ap.add_argument("--second",choices=["native","py_route"]); ap.add_argument("--run",default="run1")
ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--warmup",type=int,default=2)
ap.add_argument("--out"); ap.add_argument("--aggregate"); a=ap.parse_args()
if a.aggregate: aggregate(a.aggregate.split(","),a.out); sys.exit(0)
import random
import torch
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE","cuda"); os.environ["FL_ADAMW_FUSED"]="0"
sys.path.insert(0,"/tmp/opencode/FlagGems-base/src"); import flag_gems
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model, unwrap_model
MM_OUT=torch.ops.aten.mm.out; FG=flag_gems.mm
N_MK=[(4096,5120,1024),(1024,5120,4096),(4096,4430,1024),(1024,4430,4096),(4096,4310,1024),(1024,4310,4096),
      (4096,3740,1024),(1024,3740,4096),(4096,3290,1024),(1024,3290,4096),(4096,2990,1024),(1024,2990,4096)]
DW={((N,M),(M,K)) for (N,M,K) in N_MK}; HITS=collections.Counter()
def route(self,other):
    out=torch.empty(self.shape[0],other.shape[1],device=self.device,dtype=self.dtype)
    if (tuple(self.shape),tuple(other.shape)) in DW: HITS["x".join(map(str,self.shape))+"_"+str(other.shape[1])]+=1; return FG(self,other)
    return MM_OUT(self,other,out=out)
cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
TEMP=float(cfg.temperature); GN=float(cfg.optimizer.get("max_grad_norm",1.0))
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":"cuda"}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev); model.train()
base=unwrap_model(model)
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
def step(stream,pending):
    t0=time.perf_counter(); optimizer.zero_grad(set_to_none=True); losses=[]; kept=[]
    for _ in range(accum):
        batch=pending if pending is not None else next(stream)
        pending=next(stream,None); kept.append(batch)
        with autocast_context(dev,"bf16"):
            raw=loss_for_batch(model,processor,str(cfg.task),batch,dev,TEMP); loss=raw/accum
        if not torch.isfinite(raw): raise FloatingPointError("Non-finite training loss")
        loss.backward(); losses.append(float(raw.detach()))
    torch.nn.utils.clip_grad_norm_(trainable,GN); optimizer.step(); torch.cuda.synchronize()
    ms=(time.perf_counter()-t0)*1000; valid=0; h=hashlib.sha256()
    for batch in kept:
        for k in sorted(batch):
            h.update("|".join(batch[k]).encode()); valid+=int(base.prepare_embedding_inputs(batch[k])["attention_mask"].sum())
    return {"step_ms":round(ms,2),"valid_tokens":valid,"tokens_per_s":round(valid/(ms/1000),1),"input_hash":h.hexdigest()[:16],
            "micro_losses":[round(x,6) for x in losses]},pending
def phase():
    stream=iter(loaders[0]); pending=None
    for _ in range(a.warmup): _,pending=step(stream,pending)
    rounds=[]
    for i in range(a.rounds):
        r,pending=step(stream,pending); rounds.append(r)
        print(f"[{a.run} {a.second}] step {i}: {r['step_ms']} ms  {r['tokens_per_s']} tok/s",flush=True)
    return {"rounds":rounds}
p1=phase(); HITS.clear()
if a.second=="py_route":
    _LIB=torch.library.Library("aten","IMPL"); _LIB.impl("mm",route,"CUDA")
p2=phase()
same=all(x["input_hash"]==y["input_hash"] for x,y in zip(p1["rounds"],p2["rounds"]))
tps=lambda P:[r["tokens_per_s"] for r in P["rounds"]]
json.dump({"run":a.run,"second":a.second,"loop":"ORIGINAL entry loop (lookahead, autocast loss, isfinite host sync)","profiler":False,
           "phase_batches_identical":same,"per_shape_hits":dict(HITS),"n_shapes_hit":len(HITS),"total_hits":sum(HITS.values()),
           "tokens_per_s_p1":tps(p1),"tokens_per_s_p2":tps(p2),"phase1":p1,"phase2":p2},open(a.out,"w"),indent=1)
print(f"[{a.run} second={a.second}] phase1 {[r['step_ms'] for r in p1['rounds']]} phase2 {[r['step_ms'] for r in p2['rounds']]} | batches identical {same} | hits {sum(HITS.values())} over {len(HITS)}/12 shapes")
