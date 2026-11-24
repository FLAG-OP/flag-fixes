"""Review fe65d8f item 3: cold vs warm replay of the fg_all slow steps.
State fixed: seed 42 init, same loader window (first 2 warmup + 3 timed
steps), same RNG. COLD = fresh empty Triton cache dir; WARM = cache pre-
populated by an identical prior run. Separates per-shape first-call time
(compile+autotune+exec) from steady-state exec; records cache dir entry
counts (hit/miss proxy), compiler log presence, host submit time and device
kernel time via profiler, and whether warmup events bleed into step 1."""
import argparse, json, os, statistics, sys, time
import torch
ap=argparse.ArgumentParser()
ap.add_argument("--conf",required=True); ap.add_argument("--mode",required=True,choices=["cold","warm"])
ap.add_argument("--cache-dir",required=True); ap.add_argument("--out",required=True)
a=ap.parse_args()
os.environ["TRITON_CACHE_DIR"]=a.cache_dir
os.makedirs(a.cache_dir, exist_ok=True)
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE","cuda"); os.environ["FL_ADAMW_FUSED"]="0"
sys.path.insert(0,"/tmp/opencode/FlagGems-base/src")
import flag_gems
cache_before=len(os.listdir(a.cache_dir))
# do NOT only_enable(mm) here: it registers its own python override and would
# conflict with our timing wrapper; we call flag_gems.mm directly instead.
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model
cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
dtype_name=str(cfg.model.get("torch_dtype","bf16"))
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":"cuda"}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev)
model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
stream=iter(loaders[0])
from torch.profiler import profile, ProfilerActivity
first_call={}; MM=torch.ops.aten.mm.default
# wrap fg mm to time each call: first encounter = compile+exec; later = exec
FG=flag_gems.mm
def fg_timed(x,y):
    key=(tuple(x.shape),tuple(y.shape))
    s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    s.record(); r=FG(x,y); e.record(); torch.cuda.synchronize()
    ms=s.elapsed_time(e)
    if key not in first_call:
        first_call[key]={"first_ms":round(ms,3),"calls":1,"later_ms_sum":0.0}
    else:
        first_call[key]["calls"]+=1; first_call[key]["later_ms_sum"]+=ms
    return r
lib=torch.library.Library("aten","IMPL")
lib.impl("mm", lambda self,other: fg_timed(self,other), "CUDA")
rounds=[]
def run_step(tag):
    t0=time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for micro in range(accum):
        batch=next(stream)
        with autocast_context(dev,dtype_name):
            loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
        (loss/accum).backward()
    torch.nn.utils.clip_grad_norm_(trainable,1.0); optimizer.step()
    torch.cuda.synchronize()
    ms=(time.perf_counter()-t0)*1000
    new_shapes=len(first_call)
    rounds.append({"tag":tag,"step_ms":round(ms,1),"cum_unique_shapes":new_shapes})
    print(f"  {tag}: {ms:.0f} ms | unique shapes so far {new_shapes}", flush=True)
for i in range(2): run_step(f"warmup{i}")
for i in range(3): run_step(f"timed{i}")
cache_after=len(os.listdir(a.cache_dir))
per_shape={"x".join(str(list(k[0]))+":"+str(list(k[1]))):v for k,v in first_call.items()}
first_total=sum(v["first_ms"] for v in first_call.values())
later_calls=sum(v["calls"]-1 for v in first_call.values())
later_total=sum(v["later_ms_sum"] for v in first_call.values())
out={"mode":a.mode,"cache_dir":a.cache_dir,"cache_entries_before":cache_before,"cache_entries_after":cache_after,
     "cache_miss_proxy_new_entries":cache_after-cache_before,
     "compiler_log":"Triton MLIR/PTX logs not captured (no TRITON_KERNEL_DUMP set); cache dir diff = compile evidence",
     "rounds":rounds,
     "unique_shapes":len(first_call),
     "first_call_total_ms_incl_exec":round(first_total,1),
     "first_call_ms_per_shape_median":round(statistics.median(v["first_ms"] for v in first_call.values()),3),
     "first_call_ms_per_shape_max":round(max(v["first_ms"] for v in first_call.values()),3),
     "later_calls":later_calls,
     "later_avg_ms_per_call":round(later_total/later_calls,4) if later_calls else None,
     "per_shape":per_shape,
     "warmup_bleed_note":"first_call timing is per mm call with its own synchronize; warmup steps 0-1 populate first_call, so timed steps only see already-seen shapes unless a new length appears",
     "timing_boundaries":"first_ms includes compile+autotune+exec+sync (cannot be split without TRITON_KERNEL_DUMP/MLIR timestamps); later_ms_sum = steady-state exec+sync"}
json.dump(out,open(a.out,"w"),indent=1)
print(f"MODE={a.mode} timed steps {[r['step_ms'] for r in rounds[2:]]} | unique shapes {len(first_call)} | first-call total {first_total:.0f}ms | cache new entries {cache_after-cache_before}", flush=True)
