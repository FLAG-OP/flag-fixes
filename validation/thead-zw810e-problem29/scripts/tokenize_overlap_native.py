"""Review 7166338004 item 2: is the prefetch gain an artefact of the measurement path?

Round-56 (tokenize_overlap.py) called torch.cuda.synchronize() inside compute() to
observe device intervals; that sync is EXPERIMENT-ONLY. This script never adds a
sync inside the loss path and records only step wall time (profiler OFF).

Step loop = the ORIGINAL training entry (flagscale/train/train_retrieval.py:646-673):
one-batch lookahead, loss under autocast, the entry's own isfinite(loss) host sync,
backward, clip_grad_norm_ + optimizer.step. The only added sync is the step-end
boundary that closes the measured window (identical in both modes).

Modes (the untouched loss_for_batch in every mode unless stated)
  inline                 original path
  prefetch_faithful      only embedding_batch's prepare_embedding_inputs + move_features run
                         on a worker thread (by-group futures, keyed by the group's text list)
  prefetch_loss_reimpl   the round-56 / run-1 implementation (model call under autocast;
                         normalize / logits matmul / cross_entropy re-implemented OUTSIDE autocast)
  h2d_nb                 move_features copies each CPU tensor through a pinned buffer with
                         non_blocking=True (host_stage_timeline_r57: the pageable .to(device)
                         blocks the host until all previously enqueued kernels finish).
                         Same bytes, same kernels, same order; pinned buffers are held until
                         the step-end sync.

Tokens are counted and inputs hashed AFTER the timed window. --aggregate pairs the
per-run files (same rep + step, batch hash must match) and reports mean / median /
range / 95% CI of the paired differences and the effective tokens/s per mode.
"""
import argparse, hashlib, json, math, os, sys, time, concurrent.futures
from collections import defaultdict
from collections.abc import Mapping
T_CRIT={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262,11:2.228,12:2.201,13:2.179,14:2.160,15:2.145,
        16:2.131,17:2.120,18:2.110,19:2.101,20:2.093,21:2.086,22:2.080,23:2.074,24:2.069,25:2.064,30:2.045,40:2.021}
def stats(xs):
    n=len(xs); s=sorted(xs); mean=sum(s)/n; med=s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
    sd=math.sqrt(sum((x-mean)**2 for x in s)/(n-1)) if n>1 else 0.0
    half=(T_CRIT.get(n,1.96) if n>1 else float("nan"))*sd/math.sqrt(n) if n>1 else float("nan")
    return {"n":n,"mean_ms":round(mean,2),"median_ms":round(med,2),"min_ms":s[0],"max_ms":s[-1],"sd_ms":round(sd,2),
            "mean_ci95_ms":[round(mean-half,2),round(mean+half,2)],"n_slower":sum(x>0 for x in s),"n_faster":sum(x<0 for x in s)}
def dist(xs):
    s=sorted(xs); n=len(s)
    return {"n":n,"mean":round(sum(s)/n,1),"median":round(s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2,1),"min":s[0],"max":s[-1]}
def aggregate(a):
    runs={}
    for f in a.aggregate.split(","):
        d=json.load(open(f)); runs[d["run"]]=d
    pairs=[]
    for rep in sorted({r["rep"] for r in runs.values()}):
        A=runs.get(rep+"|inline")
        if A is None: continue
        for k in sorted(runs):
            if k==rep+"|inline" or not k.startswith(rep+"|"): continue
            B=runs[k]; order="inline->"+B["mode"] if A["started_unix"]<B["started_unix"] else B["mode"]+"->inline"
            for i in range(min(len(A["rounds"]),len(B["rounds"]))):
                ra,rb=A["rounds"][i],B["rounds"][i]
                pairs.append({"run":"run2","rep":rep,"order":order,"mode":B["mode"],"step":i,
                              "batch_hash_equal":ra["input_hash"]==rb["input_hash"],"inline_ms":ra["step_ms"],"candidate_ms":rb["step_ms"],
                              "diff_ms":round(rb["step_ms"]-ra["step_ms"],2),
                              "inline_valid_tokens_per_s":ra["valid_tokens_per_s"],"candidate_valid_tokens_per_s":rb["valid_tokens_per_s"]})
    if a.prior:
        p=json.load(open(a.prior))
        for rep in sorted(p["runs"]):
            r=p["runs"][rep]
            for i,d in enumerate(r["paired_prefetch_minus_inline"]):
                pairs.append({"run":"run1","rep":rep,"order":r["order"],"mode":"prefetch_loss_reimpl","step":i,"batch_hash_equal":None,
                              "inline_ms":r["inline_ms"][i],"candidate_ms":r["prefetch_ms"][i],"diff_ms":d,
                              "inline_valid_tokens_per_s":None,"candidate_valid_tokens_per_s":None})
    by=defaultdict(list)
    for p in pairs: by[(p["run"],p["mode"])].append(p["diff_ms"])
    per=defaultdict(list)
    for p in pairs: per[(p["run"],p["mode"],p["rep"])].append(p["diff_ms"])
    r2=[p for p in pairs if p["run"]=="run2"]
    res={"field_version":"prefetch_native_v3","time_unit":"ms","diff_definition":"candidate_ms - inline_ms (positive = candidate slower)",
         "loop":"ORIGINAL entry loop: one-batch lookahead, autocast loss, isfinite(loss) host sync, backward, clip+step; profiler OFF; no experiment sync inside the loss path",
         "protocol":"per rep: one inline process and one prefetch process (seed 42, same batch order), 2 warmup steps + 3 timed steps each; A->B / B->A alternated across reps",
         "modes":{"prefetch_faithful":"same loss_for_batch, only prepare_embedding_inputs + move_features on a worker thread",
                  "prefetch_loss_reimpl":"round-56 / run-1 implementation: logits matmul + cross_entropy re-implemented outside autocast",
                  "h2d_nb":"move_features via pinned buffers + non_blocking=True (same bytes, same kernels)"},
         "batch_hash_mismatch_run2":sum(1 for p in r2 if p["batch_hash_equal"] is False),
         "stats_by_run_variant":{f"{k[0]}|{k[1]}":stats(v) for k,v in sorted(by.items())},
         "stats_all_pairs":stats([p["diff_ms"] for p in pairs]),
         "per_rep_median_ms":{f"{k[0]}|{k[1]}|{k[2]}":round(sorted(v)[len(v)//2],2) for k,v in sorted(per.items())},
         "valid_tokens_per_s_run2":{"inline":dist([p["inline_valid_tokens_per_s"] for p in r2]),**{m:dist([p["candidate_valid_tokens_per_s"] for p in r2 if p["mode"]==m]) for m in sorted({p["mode"] for p in r2})}},
         "pairs":pairs,"raw_files":a.aggregate.split(",")}
    parts=[]
    for k,raw in sorted(by.items()):
        v=stats(raw); lo,hi=v["mean_ci95_ms"]
        cls="REGRESSION (slower)" if lo>0 else ("GAIN (faster)" if hi<0 else "NO SIGNIFICANT DIFFERENCE")
        parts.append(f"{k[0]}|{k[1]}: {cls} - slower in {v['n_slower']}/{v['n']}, median {v['median_ms']:+.1f} ms, mean {v['mean_ms']:+.1f} ms, 95% CI of the mean [{lo:+.1f}, {hi:+.1f}], range [{v['min_ms']:+.1f}, {v['max_ms']:+.1f}]")
    lo,hi=res["stats_all_pairs"]["mean_ci95_ms"]
    res["verdict"]="; ".join(parts)+f" | all {res['stats_all_pairs']['n']} pairs: slower in {res['stats_all_pairs']['n_slower']}/{res['stats_all_pairs']['n']}, median {res['stats_all_pairs']['median_ms']:+.1f} ms, mean {res['stats_all_pairs']['mean_ms']:+.1f} ms, 95% CI [{lo:+.1f}, {hi:+.1f}]"
    json.dump(res,open(a.out,"w"),indent=1); print(res["verdict"]); return
ap=argparse.ArgumentParser()
ap.add_argument("--conf"); ap.add_argument("--mode",default="inline",choices=["inline","prefetch_faithful","prefetch_loss_reimpl","h2d_nb"])
ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--warmup",type=int,default=2); ap.add_argument("--rep",default="rep1")
ap.add_argument("--out"); ap.add_argument("--aggregate",help="comma-separated per-run json files -> paired statistics"); ap.add_argument("--prior",help="run-1 json (round-57 first run) to include")
a=ap.parse_args()
if a.aggregate: aggregate(a); sys.exit(0)
import random
import torch
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE","cuda"); os.environ["FL_ADAMW_FUSED"]="0"
from omegaconf import OmegaConf
import torch.nn.functional as F
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses import retrieval as Lmod
from torch import Tensor
from flagscale.train.losses.retrieval import loss_for_batch, move_features
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model, unwrap_model
cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
TEMP=float(cfg.temperature); GN=float(cfg.optimizer.get("max_grad_norm",1.0))
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":"cuda"}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev); model.train()
base=unwrap_model(model)
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
stream=iter(loaders[0]); pending=None
_ex=concurrent.futures.ThreadPoolExecutor(max_workers=1); _FUT={}; _HOLD=[]
def _prep(texts): return move_features(base.prepare_embedding_inputs(texts),dev)
def _move_nb(features,device):
    if isinstance(features,Mapping): return {k:_move_nb(v,device) for k,v in features.items()}
    if isinstance(features,(list,tuple)): return type(features)(_move_nb(v,device) for v in features)
    if isinstance(features,Tensor) and features.device.type!=device.type:
        buf=torch.empty_like(features,pin_memory=True); buf.copy_(features); _HOLD.append(buf)
        return buf.to(device,non_blocking=True)
    return move_features(features,device)
_ORIG_EMB=Lmod.embedding_batch
def _emb_prefetch(m,inputs,device):
    f=_FUT.pop(id(inputs),None)
    feats=f.result() if f is not None else move_features(base.prepare_embedding_inputs(inputs),device)
    out=m(feats); e=out.get("sentence_embedding")
    if e is None: e=out.get("pooler_output")
    if e is None: raise RuntimeError(f"Retrieval model output has no embedding field: {list(out)}")
    return F.normalize(e.float(),p=2,dim=-1)
def loss_prefetch_faithful(b):
    keys=["anchor","positive"]+sorted(k for k in b if k.startswith("negative"))
    _FUT.clear(); Lmod.embedding_batch=_emb_prefetch
    for k in keys: _FUT[id(b[k])]=_ex.submit(_prep,b[k])
    try: return loss_for_batch(model,processor,str(cfg.task),b,dev,TEMP)
    finally: Lmod.embedding_batch=_ORIG_EMB
def loss_prefetch_reimpl(b):
    neg=sorted(k for k in b if k.startswith("negative")); keys=["anchor","positive"]+neg; fut={}; outs={}
    for k in keys[:2]: fut[k]=_ex.submit(_prep,b[k])
    for i,k in enumerate(keys):
        if i+2<len(keys) and keys[i+2] not in fut: fut[keys[i+2]]=_ex.submit(_prep,b[keys[i+2]])
        feats=fut[k].result()
        with autocast_context(dev,"bf16"): e=model(feats).get("sentence_embedding")
        outs[k]=F.normalize(e.float(),p=2,dim=-1)
    anchors,positives=outs["anchor"],outs["positive"]
    grouped=torch.stack([positives]+[outs[k] for k in neg],dim=1).reshape(anchors.shape[0]*(1+len(neg)),-1)
    return F.cross_entropy(anchors@grouped.T/TEMP,torch.arange(anchors.shape[0],device=dev)*(1+len(neg)))
def micro_loss(b):
    if a.mode=="inline": return loss_for_batch(model,processor,str(cfg.task),b,dev,TEMP)
    if a.mode=="prefetch_faithful": return loss_prefetch_faithful(b)
    if a.mode=="prefetch_loss_reimpl": return loss_prefetch_reimpl(b)
    Lmod.move_features=_move_nb
    try: return loss_for_batch(model,processor,str(cfg.task),b,dev,TEMP)
    finally: Lmod.move_features=move_features
def step():
    global pending
    t0=time.perf_counter(); optimizer.zero_grad(set_to_none=True); losses=[]; kept=[]
    for _ in range(accum):
        batch=pending if pending is not None else next(stream)
        pending=next(stream,None)
        kept.append(batch)
        with autocast_context(dev,"bf16"):
            raw=micro_loss(batch); loss=raw/accum
        if not torch.isfinite(raw): raise FloatingPointError("Non-finite training loss")
        loss.backward(); losses.append(float(raw.detach()))
    torch.nn.utils.clip_grad_norm_(trainable,GN); optimizer.step(); torch.cuda.synchronize(); _HOLD.clear()
    return (time.perf_counter()-t0)*1000,losses,kept
def tokens_and_hash(kept):
    valid=0; h=hashlib.sha256()
    for batch in kept:
        for k in sorted(batch):
            h.update("|".join(batch[k]).encode()); valid+=int(base.prepare_embedding_inputs(batch[k])["attention_mask"].sum())
    return valid,h.hexdigest()[:16]
started=time.time()
for _ in range(a.warmup): step()
rounds=[]
for i in range(a.rounds):
    ms,losses,kept=step(); valid,h=tokens_and_hash(kept)
    rounds.append({"step":i,"step_ms":round(ms,2),"valid_tokens":valid,"valid_tokens_per_s":round(valid/(ms/1000),1),
                   "input_hash":h,"micro_losses":[round(x,6) for x in losses]})
    print(f"[{a.rep} {a.mode}] step {i}: {ms:.1f} ms  {valid} tokens  {valid/(ms/1000):.0f} tok/s",flush=True)
json.dump({"run":f"{a.rep}|{a.mode}","rep":a.rep,"mode":a.mode,
           "started_unix":started,"experiment_sync_in_loss_path":False,"profiler":False,"warmup_steps":a.warmup,
           "rounds":rounds},open(a.out,"w"),indent=1)
