"""Review 7166338004 item 3, profiler-free complement: where does the HOST spend
its time in the ORIGINAL training entry loop?

No profiler and no added synchronization: perf_counter() is placed around calls
the loop already makes. Device idle is NOT observed here (that needs a tracer);
what is observed is how long the host blocks inside each call.

--loop entry   : flagscale/train/train_retrieval.py:646-673 - one-batch lookahead,
                 loss under autocast, the entry's isfinite(loss) host sync, backward,
                 clip_grad_norm_ + optimizer.step
--loop capture : the loop used by training_trace_r36 / route_isolated / wholestep_mm
                 (no lookahead, no loss sync); measured only to relate older numbers
                 to the entry loop

The only extra synchronization is the step-end boundary that closes the timed window.
Per group (anchor / positive / negative_1..3): prepare_embedding_inputs (tokenizer +
tensorize), H2D (move_features), forward launch (model call returns).
Per micro: DataLoader fetch, loss sync (entry loop), backward launch.
Per step: clip launch, optimizer.step launch, end wait.

The final section is a micro-test of whether a pageable H2D copy blocks the host
until previously enqueued kernels finish (stream-ordered) on this device.
"""
import argparse, gc, json, os, sys, time
from collections import defaultdict
from collections.abc import Mapping
import torch
ap=argparse.ArgumentParser()
ap.add_argument("--conf",required=True); ap.add_argument("--loop",default="entry",choices=["entry","capture"])
ap.add_argument("--h2d",default="blocking",choices=["blocking","nonblocking"],help="nonblocking = move_features via pinned buffers + non_blocking=True")
ap.add_argument("--gc",default="default",choices=["default","track","disabled"],help="track: count Python GC pauses via gc.callbacks (no sync); disabled: gc.disable()")
ap.add_argument("--warmup",type=int,default=2); ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--out",required=True)
a=ap.parse_args()
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE","cuda"); os.environ["FL_ADAMW_FUSED"]="0"
import torch.nn.functional as F
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses import retrieval as Lmod
from flagscale.train.losses.retrieval import loss_for_batch
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
stream=iter(loaders[0])
pc=time.perf_counter; GROUPS=[]
GC={"n":0,"ms":0.0,"max":0.0,"t":None}
def _gc_cb(phase,info):
    if phase=="start": GC["t"]=pc()
    elif GC["t"] is not None:
        d=(pc()-GC["t"])*1e3; GC["n"]+=1; GC["ms"]+=d; GC["max"]=max(GC["max"],d); GC["t"]=None
if a.gc=="track": gc.callbacks.append(_gc_cb)
elif a.gc=="disabled": gc.disable()
_ORIG_MOVE=Lmod.move_features; _HOLD=[]
def _move_nb(features,device):
    if isinstance(features,Mapping): return {k:_move_nb(v,device) for k,v in features.items()}
    if isinstance(features,(list,tuple)): return type(features)(_move_nb(v,device) for v in features)
    if torch.is_tensor(features) and features.device.type!=device.type:
        buf=torch.empty_like(features,pin_memory=True); buf.copy_(features); _HOLD.append(buf)
        return buf.to(device,non_blocking=True)
    return _ORIG_MOVE(features,device)
MOVE=_move_nb if a.h2d=="nonblocking" else _ORIG_MOVE
def embedding_batch_timed(m,inputs,device):
    t0=pc(); feats0=base.prepare_embedding_inputs(inputs); t1=pc()
    feats=MOVE(feats0,device); t2=pc()
    out=m(feats); t3=pc()
    e=out.get("sentence_embedding")
    if e is None: e=out.get("pooler_output")
    if e is None: raise RuntimeError(f"Retrieval model output has no embedding field: {list(out)}")
    GROUPS.append({"prepare_embedding_inputs_ms":round((t1-t0)*1e3,3),"h2d_ms":round((t2-t1)*1e3,3),"forward_launch_ms":round((t3-t2)*1e3,3)})
    return F.normalize(e.float(),p=2,dim=-1)
Lmod.embedding_batch=embedding_batch_timed
pending=None
def step(i):
    global pending,GROUPS
    t0=pc(); rec=defaultdict(float); GROUPS=[]; w0=time.time()
    GC.update(n=0,ms=0.0,max=0.0,t=None)
    optimizer.zero_grad(set_to_none=True)
    for micro in range(accum):
        if a.loop=="entry":
            batch=pending if pending is not None else next(stream)
            t=pc(); pending=next(stream,None); rec["next_batch_ms"]+=(pc()-t)*1e3
            t=pc()
            with autocast_context(dev,"bf16"):
                raw=loss_for_batch(model,processor,str(cfg.task),batch,dev,TEMP); loss=raw/accum
            rec["forward_call_ms"]+=(pc()-t)*1e3
            t=pc(); finite=bool(torch.isfinite(raw)); rec["loss_sync_ms"]+=(pc()-t)*1e3
            if not finite: raise FloatingPointError("Non-finite training loss")
            t=pc(); loss.backward(); rec["backward_launch_ms"]+=(pc()-t)*1e3
            float(raw.detach())
        else:
            t=pc(); batch=next(stream); rec["next_batch_ms"]+=(pc()-t)*1e3
            t=pc()
            with autocast_context(dev,"bf16"): loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,TEMP)
            rec["forward_call_ms"]+=(pc()-t)*1e3
            t=pc(); (loss/accum).backward(); rec["backward_launch_ms"]+=(pc()-t)*1e3
    t=pc(); torch.nn.utils.clip_grad_norm_(trainable,GN); rec["clip_launch_ms"]+=(pc()-t)*1e3
    t=pc(); optimizer.step(); rec["optimizer_step_launch_ms"]+=(pc()-t)*1e3
    t=pc(); torch.cuda.synchronize(); rec["end_wait_ms"]+=(pc()-t)*1e3; _HOLD.clear()
    g=defaultdict(float)
    for x in GROUPS:
        for k,v in x.items(): g[k]+=v
    return {"step":i,"step_ms":round((pc()-t0)*1e3,2),"wall_start":w0,"wall_end":time.time(),"gc_collections":GC["n"],"gc_total_ms":round(GC["ms"],2),"gc_max_ms":round(GC["max"],3),**{k:round(v,2) for k,v in rec.items()},
            "host_accounted_ms":round(sum(rec.values()),2),**{k:round(v,2) for k,v in g.items()},"n_groups":len(GROUPS),"groups":GROUPS}
def h2d_probe():
    x=torch.randn(4096,4096,device=dev,dtype=torch.bfloat16)
    for _ in range(3): y=x@x
    torch.cuda.synchronize(); t=pc()
    for _ in range(20): y=x@x
    torch.cuda.synchronize(); per=(pc()-t)/20*1e3
    n=max(1,int(round(120.0/per)))
    ids=torch.randint(0,100000,(40,48)); pin=torch.empty(40,48,dtype=torch.int64).pin_memory(); pin.copy_(ids)
    cases=[("pageable_to_device",lambda: ids.to(dev)),
           ("pageable_to_device_non_blocking",lambda: ids.to(dev,non_blocking=True)),
           ("pinned_non_blocking_to_device",lambda: pin.to(dev,non_blocking=True)),
           ("device_scalar_item_D2H",lambda: float(x.sum()))]
    out={"mm_ms":round(per,3),"device_work_enqueued_ms":round(n*per,1),"shape":"[40,48] int64"}
    for name,fn in cases:
        hs=[]
        for _ in range(3):
            torch.cuda.synchronize()
            for _ in range(n): y=x@x
            t0=pc(); fn(); hs.append((pc()-t0)*1e3); torch.cuda.synchronize()
        out[name]={"host_block_ms_median":round(sorted(hs)[1],3),"host_block_ms_all":[round(v,3) for v in hs]}
    return out
for i in range(a.warmup): step(i)
R=[step(a.warmup+i) for i in range(a.rounds)]
keys=[k for k in R[0] if k.endswith("_ms") and k not in ("step_ms","host_accounted_ms")]
res={"field_version":"host_stage_timeline_v1","time_unit":"ms","loop":a.loop,"h2d":a.h2d,"gc":a.gc,"profiler":False,"added_sync":"step-end boundary only",
     "stage_definitions":{"prepare_embedding_inputs_ms":"host time in the tokenizer + tensorization (20 groups per step)",
                          "h2d_ms":"host time blocked in move_features (.to(device), pageable, non_blocking=False)",
                          "forward_launch_ms":"host time for the model call to return (kernel issue)",
                          "next_batch_ms":"DataLoader fetch","loss_sync_ms":"the entry's isfinite(loss) host sync",
                          "backward_launch_ms":"host time for .backward() to return","clip_launch_ms":"clip_grad_norm_ call",
                          "optimizer_step_launch_ms":"optimizer.step() call","end_wait_ms":"step-end sync (measurement boundary)"},
     "timed_window_unix":[R[0]["wall_start"],R[-1]["wall_end"]],"steps":R,"mean_per_step":{k:round(sum(r[k] for r in R)/len(R),2) for k in ["step_ms",*keys]},
     "gc_per_step_mean":{"collections":round(sum(r["gc_collections"] for r in R)/len(R),1),"total_ms":round(sum(r["gc_total_ms"] for r in R)/len(R),2),"max_pause_ms":max(r["gc_max_ms"] for r in R)},
     "tokenize_total_per_step_ms":round(sum(r["prepare_embedding_inputs_ms"] for r in R)/len(R),2),
     "tokenize_pct_of_step":round(100*sum(r["prepare_embedding_inputs_ms"] for r in R)/sum(r["step_ms"] for r in R),2),
     "h2d_probe":h2d_probe()}
json.dump(res,open(a.out,"w"),indent=1)
print(f"loop={a.loop} h2d={a.h2d} gc={a.gc} step_ms={[r['step_ms'] for r in R]} | gc per step (n, ms, max): {[(r['gc_collections'],r['gc_total_ms'],r['gc_max_ms']) for r in R]}")
for k in keys: print(f"  {k:<32} mean {res['mean_per_step'][k]:>9.2f}  per step {[r[k] for r in R]}")
print("  groups (last step):",[(g["prepare_embedding_inputs_ms"],g["h2d_ms"],g["forward_launch_ms"]) for g in R[-1]["groups"]])
print("  h2d_probe:",{k:(v["host_block_ms_median"] if isinstance(v,dict) else v) for k,v in res["h2d_probe"].items()})
