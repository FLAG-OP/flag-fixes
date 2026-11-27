"""Review bqiozcA item 1: ONE process per mode (no aten registration stacking).
Modes (mutually exclusive per process):
  native           - plain mm
  py_route         - dW via flag_gems.mm through the Python IMPL override (current path)
  native_contig    - dW via NATIVE mm on an explicit contiguous() copy, same route/fallback
Reports: per-shape hits for all 12 whitelist shapes (attributable: only ONE
override exists per process), kernel names via profiler on a separate diagnostic
step, per-step wall (profiler OFF), valid tokens/s, input hashes.
dW fallback (non-whitelist) is IDENTICAL across the two routed modes (mm.out +
preallocated output), so the only factor that changes is the dW call entry."""
import argparse, json, os, sys, time, collections
import torch
ap=argparse.ArgumentParser()
ap.add_argument("--conf",required=True); ap.add_argument("--mode",required=True,choices=["native","py_route","native_contig"])
ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--out",required=True)
a=ap.parse_args()
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE","cuda"); os.environ["FL_ADAMW_FUSED"]="0"
sys.path.insert(0,"/tmp/opencode/FlagGems-base/src")
import flag_gems
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model
MM_OUT=torch.ops.aten.mm.out; FG=flag_gems.mm
N_MK=[(4096,5120,1024),(1024,5120,4096),(4096,4430,1024),(1024,4430,4096),(4096,4310,1024),(1024,4310,4096),
      (4096,3740,1024),(1024,3740,4096),(4096,3290,1024),(1024,3290,4096),(4096,2990,1024),(1024,2990,4096)]
DW={((N,M),(M,K)):(N,M,K) for (N,M,K) in N_MK}
MODE=a.mode; HITS=collections.Counter()
def route(self,other):
    key=(tuple(self.shape),tuple(other.shape))
    out=torch.empty(self.shape[0],other.shape[1],device=self.device,dtype=self.dtype)
    if key in DW:
        HITS[f"{list(key[0])}x{list(key[1])}"]+=1
        if MODE=="py_route": return FG(self,other)
        return MM_OUT(self.contiguous(),other,out=out)   # native_contig
    return MM_OUT(self,other,out=out)
_LIB=None
if MODE!="native":
    # HOLD the Library object: if it is garbage-collected the registration is
    # silently dropped (found in the first run of this script: all modes showed
    # hits=0 and identical aten::mm event counts).
    _LIB=torch.library.Library("aten","IMPL"); _LIB.impl("mm",route,"CUDA")
cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":"cuda"}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev); model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
stream=iter(loaders[0])
from transformers import AutoTokenizer
_tok=AutoTokenizer.from_pretrained(str(cfg.model.model_path))
import hashlib
def step(rec):
    t0=time.perf_counter(); optimizer.zero_grad(set_to_none=True); bs=[]
    for micro in range(accum):
        batch=next(stream); bs.append(batch)
        with autocast_context(dev,"bf16"):
            loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
        (loss/accum).backward()
    torch.nn.utils.clip_grad_norm_(trainable,1.0); optimizer.step(); torch.cuda.synchronize()
    ms=(time.perf_counter()-t0)*1000
    valid=0; h=hashlib.sha256()
    for b in bs:
        for k in sorted(b.keys()):
            h.update("|".join(b[k]).encode())
            enc=_tok(b[k],padding=True,truncation=True,max_length=int(cfg.model.get("max_length",8192)),return_tensors=None)
            valid+=sum(sum(1 for z in m if z==1) for m in enc["attention_mask"])
    if rec: return ms,valid,h.hexdigest()[:16]
for i in range(2): step(False)                  # warmup (identical across modes)
rounds=[step(True) for _ in range(a.rounds)]
# kernel names: one extra diagnostic step under the profiler (not part of timing)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
    step(False)
ks=collections.Counter()
for e in p.events():
    if "mm_kernel" in e.key or "gemm_ktype" in e.key or e.key=="aten::mm": ks[e.key[:70]]+=1
res={"mode":MODE,"rounds":[{"step_ms":round(m,1),"valid_tokens":v,"tokens_per_s":round(v/(m/1000)),"input_hash":h} for m,v,h in rounds],
     "per_shape_hits":dict(HITS),"total_hits":sum(HITS.values()),"n_shapes_hit":len(HITS),
     "all_12_hit":set(HITS)=={f"{list(k[0])}x{list(k[1])}" for k in DW},
     "diagnostic_kernels":dict(ks),"host_aten_mm_events":ks.get("aten::mm",0)}
json.dump(res,open(a.out,"w"),indent=1)
print(f"[{MODE}] steps {[round(m,1) for m,_,_ in rounds]} | hits {sum(HITS.values())} over {len(HITS)}/12 shapes | aten::mm events {ks.get('aten::mm',0)}")
