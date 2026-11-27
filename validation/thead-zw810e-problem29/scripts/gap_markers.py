"""Review 7166338004 item 3: explicit host markers for the device gaps.

Captures ONE profiler trace (2 warmup steps, then 3 profiled steps = steps 2-4, the
window of the r36 natural trace) and adds explicit record_function markers for the
host stages that emit no aten op:

  MK_tokenizer                  the tokenizer call itself
  MK_prepare_embedding_inputs   model.prepare_embedding_inputs (contains MK_tokenizer)
  MK_h2d                        losses.retrieval.move_features
  MK_next_batch                 DataLoader fetch (entry loop)
  MK_loss_sync                  the entry's isfinite(loss) host sync (entry loop)
  MK_python_gc                  Python garbage-collection pauses (gc.callbacks), on the thread
                                that runs the collector; --gc disabled switches the collector off

plus the CUDA runtime calls of the same trace:
  launch_submit     cudaLaunchKernel* / cuLaunchKernel* call duration (host submitting)
  blocking_runtime  *Synchronize / cudaMemcpy / cudaMalloc / cudaFree ... call duration
  other_runtime     remaining runtime calls
  cpu_op            aten op dispatch intervals

--loop entry   : the ORIGINAL entry loop (train_retrieval.py:646-673): one-batch
                 lookahead + isfinite(loss) host sync
--loop capture : the r36 capture loop (training_trace_r36 / gap_attribution_r53):
                 no lookahead, no loss sync

For every device gap >= --min-gap-ms in every profiled step each category is CLIPPED
to the gap and then unioned. Categories are reported as an exclusive partition in
the priority order above, so they add up to the covered time; the remainder is
UNKNOWN. The first device event after the gap ("next kernel") is reported together
with the host time its launch call was issued. Nothing is attributed by
neighbourhood: a gap is only covered by a category whose own interval is inside it.
Time unit of every *_ms field: ms. The profiler inflates absolute step time and
launch overhead; only the per-gap partition is interpreted, and the profiler-free
host timings of host_stage_timeline_r57 are the reference for the real entry.
"""
import argparse, bisect, gc, gzip, json, math, os, sys, time
from collections import defaultdict
import torch
from torch.profiler import profile, ProfilerActivity, record_function
ap=argparse.ArgumentParser()
ap.add_argument("--conf",required=True); ap.add_argument("--loop",default="entry",choices=["entry","capture"])
ap.add_argument("--warmup",type=int,default=2); ap.add_argument("--steps",type=int,default=3)
ap.add_argument("--min-gap-ms",type=float,default=2.0); ap.add_argument("--out",required=True)
ap.add_argument("--gc",default="default",choices=["default","disabled"],help="disabled: gc.disable() (collector off, explicit marker not emitted)")
a=ap.parse_args()
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE","cuda"); os.environ["FL_ADAMW_FUSED"]="0"
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
_ORIG_PREP=base.prepare_embedding_inputs; _ORIG_MOVE=Lmod.move_features
def _prep(inputs):
    with record_function("MK_prepare_embedding_inputs"): return _ORIG_PREP(inputs)
base.prepare_embedding_inputs=_prep
def _move(features,device):
    with record_function("MK_h2d"): return _ORIG_MOVE(features,device)
Lmod.move_features=_move
_tokcls=type(base.tokenizer); _ORIG_TOK=_tokcls.__call__
def _tok(self,*args,**kwargs):
    with record_function("MK_tokenizer"): return _ORIG_TOK(self,*args,**kwargs)
_tokcls.__call__=_tok
_GC_RF={}
def _gc_cb(phase,info):
    if phase=="start": _GC_RF["gc"]=record_function("MK_python_gc"); _GC_RF["gc"].__enter__()
    elif "gc" in _GC_RF: _GC_RF.pop("gc").__exit__(None,None,None)
if a.gc=="disabled": gc.disable()
else: gc.callbacks.append(_gc_cb)
pending=None
def step(i):
    global pending
    t0=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
    for micro in range(accum):
        if a.loop=="entry":
            batch=pending if pending is not None else next(stream)
            with record_function("MK_next_batch"): pending=next(stream,None)
            with record_function(f"rf_s{i}_forward_m{micro}"):
                with autocast_context(dev,"bf16"):
                    raw=loss_for_batch(model,processor,str(cfg.task),batch,dev,TEMP); loss=raw/accum
                with record_function("MK_loss_sync"): finite=bool(torch.isfinite(raw))
            if not finite: raise FloatingPointError("Non-finite training loss")
            float(raw.detach())
        else:
            batch=next(stream)
            with record_function(f"rf_s{i}_forward_m{micro}"):
                with autocast_context(dev,"bf16"): loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,TEMP)
        with record_function(f"rf_s{i}_backward_m{micro}"): loss.backward()
    with record_function(f"rf_s{i}_clip_optimizer"):
        torch.nn.utils.clip_grad_norm_(trainable,GN); optimizer.step()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)*1e3
for i in range(a.warmup): step(i)
prof=profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]); prof.__enter__()
step_ms={a.warmup+i:step(a.warmup+i) for i in range(a.steps)}
prof.__exit__(None,None,None)
tmp=a.out+".trace.json"; prof.export_chrome_trace(tmp)
trace=json.load(open(tmp)); os.remove(tmp)
with gzip.open(tmp+".gz","wt") as f: json.dump(trace,f)
E=trace["traceEvents"]
dev_e=[e for e in E if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
rt=[e for e in E if e.get("cat") in ("cuda_runtime","cuda_driver")]
cpu=[e for e in E if e.get("cat")=="cpu_op"]
ann=[e for e in E if e.get("cat")=="user_annotation"]
rf=[e for e in ann if e.get("name","").startswith("rf_s")]
def end(e): return e["ts"]+e.get("dur",0)
def merge(ivs):
    out=[]
    for s,e in sorted(tuple(iv) for iv in ivs):
        if e<=s: continue
        if out and s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out
def length(iv): return sum(e-s for s,e in iv)
def sub(iv,cut):
    out=[]
    for s,e in iv:
        cur=s
        for cs,ce in cut:
            if ce<=cur or cs>=e: continue
            if cs>cur: out.append((cur,cs))
            cur=max(cur,ce)
            if cur>=e: break
        if cur<e: out.append((cur,e))
    return out
def stepidx(name): return int(name.split("_")[1][1:])
host_win={}
for e in rf:
    s=stepidx(e["name"]); w=host_win.setdefault(s,[e["ts"],end(e)])
    w[0]=min(w[0],e["ts"]); w[1]=max(w[1],end(e))
steps=sorted(host_win)
corr2rt={}
for e in rt:
    c=e.get("args",{}).get("correlation")
    if c is not None: corr2rt[c]=e
def host_step(t):
    for s in steps:
        if host_win[s][0]<=t<=host_win[s][1]: return s
    return min(steps,key=lambda s:min(abs(t-host_win[s][0]),abs(t-host_win[s][1])))
dev_by_step={s:[] for s in steps}; dwin={s:[math.inf,-math.inf] for s in steps}
def put(e,s):
    dev_by_step[s].append(e); w=dwin[s]; w[0]=min(w[0],e["ts"]); w[1]=max(w[1],end(e))
placed=set()
for e in dev_e:
    r=corr2rt.get(e.get("args",{}).get("correlation"))
    if r is not None: put(e,host_step(r["ts"])); placed.add(id(e))
for e in dev_e:
    if id(e) in placed: continue
    t=e["ts"]
    put(e,min(steps,key=lambda k: 0 if dwin[k][0]<=t<=dwin[k][1] else min(abs(t-dwin[k][0]),abs(t-dwin[k][1]))))
def is_launch(e): return "LaunchKernel" in e.get("name","")
SYNC_NAMES={"cudaMemcpy","cudaMalloc","cudaFree","cudaHostAlloc","cudaMemset","cudaStreamQuery","cudaEventQuery","cudaStreamWaitEvent"}
def is_sync(e): n=e.get("name",""); return ("Synchronize" in n or n in SYNC_NAMES) and not is_launch(e)
def named(n): return [e for e in ann if e.get("name")==n]
launch=[e for e in rt if is_launch(e)]; sync=[e for e in rt if is_sync(e)]
other_rt=[e for e in rt if not is_launch(e) and not is_sync(e)]
CATS=[("tokenizer",named("MK_tokenizer")),("prepare_embedding_inputs_other",named("MK_prepare_embedding_inputs")),
      ("h2d",named("MK_h2d")),("next_batch",named("MK_next_batch")),("loss_sync",named("MK_loss_sync")),
      ("python_gc",named("MK_python_gc")),("launch_submit",launch),("blocking_runtime",sync),("other_runtime",other_rt),("cpu_op",cpu)]
GRID=5000
def binned(items):
    B=defaultdict(list)
    for s,t,p in items:
        for b in range(int(s)//GRID,int(t)//GRID+1): B[b].append((s,t,p))
    return B
BINS=[(n,binned([(e["ts"],end(e),None) for e in ev])) for n,ev in CATS]
RT_BIN=binned([(e["ts"],end(e),e.get("name","")) for e in rt])
def gather(B,ga,gb):
    out=[]
    for b in range(int(ga)//GRID,int(gb)//GRID+1): out+=B.get(b,())
    return [(max(s,ga),min(t,gb),p) for s,t,p in out if max(s,ga)<min(t,gb)]
def partition(ga,gb):
    used=[]; part={}
    for n,B in BINS:
        x=sub(merge([(s,t) for s,t,_ in gather(B,ga,gb)]),used); part[n]=length(x); used=merge(used+x)
    part["unknown"]=(gb-ga)-sum(part.values())
    return part
MIN_US=a.min_gap_ms*1000
out_steps={}; tot=defaultdict(float); tot_ge=defaultdict(float); n_dev_without_corr=len(dev_e)-len(placed)
for s in steps:
    Ds=sorted(dev_by_step[s],key=lambda e:e["ts"]); starts=[e["ts"] for e in Ds]
    D=merge([(e["ts"],end(e)) for e in Ds])
    big=[]; allp=defaultdict(float); ge=defaultdict(float); n_ge=0; n_g=0; idle_ge=0.0
    rf_s=[e for e in rf if stepidx(e["name"])==s]
    for i in range(len(D)-1):
        ga,gb=D[i][1],D[i+1][0]
        if gb<=ga: continue
        n_g+=1; part=partition(ga,gb)
        for k,v in part.items(): allp[k]+=v
        if (gb-ga)<MIN_US: continue
        n_ge+=1; idle_ge+=gb-ga
        for k,v in part.items(): ge[k]+=v; tot_ge[k]+=v
        j=bisect.bisect_left(starts,gb-0.5); k=bisect.bisect_right(starts,ga+0.5)-1
        ne=Ds[j] if j<len(Ds) else None; pe=Ds[k] if k>=0 else None
        r=corr2rt.get(ne.get("args",{}).get("correlation")) if ne else None
        enc=sorted(((min(gb,end(e))-max(ga,e["ts"]),e["name"]) for e in rf_s if min(gb,end(e))>max(ga,e["ts"])),reverse=True)
        byname=defaultdict(list)
        for st,en,nm in gather(RT_BIN,ga,gb): byname[nm].append((st,en))
        top=sorted(((length(merge(v)),n) for n,v in byname.items()),reverse=True)[:4]
        big.append({"gap_start_rel_step_ms":round((ga-D[0][0])/1000,3),"gap_ms":round((gb-ga)/1000,3),
                    **{k:round(v/1000,3) for k,v in part.items()},
                    "enclosing_host_markers":[{"name":n,"overlap_ms":round(o/1000,3)} for o,n in enc[:2]],
                    "top_runtime_calls":[{"name":n,"clipped_union_ms":round(v/1000,3)} for v,n in top],
                    "prev_kernel":pe["name"][:120] if pe else None,
                    "next_kernel":{"name":ne["name"][:120],"dur_us":round(ne.get("dur",0),1),"launch_call":r["name"] if r else None,
                                   "launch_rel_gap_start_ms":round((r["ts"]-ga)/1000,3) if r else None} if ne else None})
    window=D[-1][1]-D[0][0]; busy=length(D)
    out_steps[f"s{s}"]={"step_ms_profiled":round(step_ms[s],1),"device_window_ms":round(window/1000,2),
        "device_busy_union_ms":round(busy/1000,2),"device_idle_ms":round((window-busy)/1000,2),"busy_pct":round(100*busy/window,2),
        "n_device_events":len(Ds),"n_gaps":n_g,"n_gaps_ge_min":n_ge,"idle_in_gaps_ge_min_ms":round(idle_ge/1000,2),
        "all_gaps_partition_ms":{k:round(v/1000,2) for k,v in allp.items()},
        "gaps_ge_min_partition_ms":{k:round(v/1000,2) for k,v in ge.items()},"gaps":big}
    tot["device_window_ms"]+=window; tot["device_busy_union_ms"]+=busy; tot["n_gaps"]+=n_g; tot["n_gaps_ge_min"]+=n_ge
    tot["idle_in_gaps_ge_min_ms"]+=idle_ge
    for k,v in allp.items(): tot["all_gaps|"+k]+=v
CAT_ORDER=[n for n,_ in CATS]+["unknown"]
res={"field_version":"gap_markers_v2","time_unit":"ms","loop":a.loop,"gc":a.gc,"min_gap_ms":a.min_gap_ms,"profiled_steps":steps,
     "method":"device gaps = union of kernel/gpu_memcpy/gpu_memset intervals per step; per gap every category interval is CLIPPED to the gap, then unioned, then made exclusive in priority order (tokenizer > prepare_embedding_inputs > h2d > next_batch > loss_sync > python_gc > launch_submit > blocking_runtime > other_runtime > cpu_op); UNKNOWN = gap - covered. next_kernel = first device event after the gap, with the host time of its launch call relative to the gap start. No neighbourhood attribution.",
     "categories":{"tokenizer":"MK_tokenizer: the tokenizer call","prepare_embedding_inputs_other":"MK_prepare_embedding_inputs minus tokenizer",
                   "h2d":"MK_h2d: move_features","next_batch":"MK_next_batch: DataLoader fetch (entry loop)",
                   "loss_sync":"MK_loss_sync: the entry's isfinite(loss) host sync (entry loop)",
                   "python_gc":"MK_python_gc: Python garbage-collection pauses (gc.callbacks)",
                   "launch_submit":"cudaLaunchKernel*/cuLaunchKernel* call duration",
                   "blocking_runtime":"*Synchronize / cudaMemcpy / cudaMalloc / cudaFree / cudaStreamQuery ... call duration",
                   "other_runtime":"other CUDA runtime call durations","cpu_op":"aten op dispatch intervals not inside the above"},
     "marker_names_found":sorted({e["name"] for e in ann if e.get("name","").startswith("MK_")}),
     "device_events_without_correlation":n_dev_without_corr,"device_events_total":len(dev_e),
     "steps":out_steps,
     "totals":{"device_window_ms":round(tot["device_window_ms"]/1000,2),"device_busy_union_ms":round(tot["device_busy_union_ms"]/1000,2),
               "busy_pct":round(100*tot["device_busy_union_ms"]/tot["device_window_ms"],2),"n_gaps":int(tot["n_gaps"]),
               "n_gaps_ge_min":int(tot["n_gaps_ge_min"]),"idle_in_gaps_ge_min_ms":round(tot["idle_in_gaps_ge_min_ms"]/1000,2)},
     "totals_gaps_ge_min_partition_ms":{k:round(tot_ge[k]/1000,2) for k in CAT_ORDER},
     "totals_all_gaps_partition_ms":{k:round(tot["all_gaps|"+k]/1000,2) for k in CAT_ORDER}}
json.dump(res,open(a.out,"w"),indent=1)
print(f"loop={a.loop} gc={a.gc} markers={res['marker_names_found']} | dev events {len(dev_e)} (without correlation {n_dev_without_corr})")
for s,rec in out_steps.items():
    print(f" step {s}: profiled {rec['step_ms_profiled']} ms | device busy {rec['device_busy_union_ms']} / {rec['device_window_ms']} ms ({rec['busy_pct']}%) | gaps {rec['n_gaps']} (>= {a.min_gap_ms} ms: {rec['n_gaps_ge_min']}, idle {rec['idle_in_gaps_ge_min_ms']} ms)")
    print("   ge-min partition:",rec["gaps_ge_min_partition_ms"])
    for g in rec["gaps"][:5]:
        nk=g["next_kernel"]
        print(f"   gap@{g['gap_start_rel_step_ms']:>8} {g['gap_ms']:<7} tok {g['tokenizer']:<7} prep+ {g['prepare_embedding_inputs_other']:<6} h2d {g['h2d']:<6} nb {g['next_batch']:<6} sync {g['loss_sync']:<7} gc {g['python_gc']:<6} launch {g['launch_submit']:<6} blk {g['blocking_runtime']:<6} ort {g['other_runtime']:<6} cpu {g['cpu_op']:<6} UNK {g['unknown']:<7} | {[m['name'] for m in g['enclosing_host_markers']]} | next {nk['launch_call'] if nk else None}@{nk['launch_rel_gap_start_ms'] if nk else None}")
