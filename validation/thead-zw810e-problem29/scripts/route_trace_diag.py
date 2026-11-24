"""Review bqWBBIQ item 1 (trace): same input step under native / dw_only_routed /
dw_native_contig with the profiler ON (diagnostic only - timings here are NOT
the perf numbers). Reports per mode: host-side aten::mm dispatch count+time,
device kernel time by kernel name, and the host gap (wall - device busy)."""
import argparse, json, os, sys, time, collections
import torch
ap=argparse.ArgumentParser(); ap.add_argument("--conf",required=True)
ap.add_argument("--mode",required=True,choices=["native","dw_only_routed","dw_native_contig"])
ap.add_argument("--out",required=True); a=ap.parse_args()
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
DW={((N,M),(M,K)) for (N,M,K) in N_MK}
MODE=a.mode
def route(self,other):
    key=(tuple(self.shape),tuple(other.shape))
    out=torch.empty(self.shape[0],other.shape[1],device=self.device,dtype=self.dtype)
    if key in DW:
        if MODE=="dw_only_routed": return FG(self,other)
        return MM_OUT(self.contiguous(),other,out=out)
    return MM_OUT(self,other,out=out)
if MODE!="native":
    lib=torch.library.Library("aten","IMPL"); lib.impl("mm",route,"CUDA")
cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":"cuda"}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev); model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
stream=iter(loaders[0])
def step():
    optimizer.zero_grad(set_to_none=True)
    for micro in range(accum):
        batch=next(stream)
        with autocast_context(dev,"bf16"):
            loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
        (loss/accum).backward()
    torch.nn.utils.clip_grad_norm_(trainable,1.0); optimizer.step(); torch.cuda.synchronize()
for _ in range(2): step()
from torch.profiler import profile, ProfilerActivity
t0=time.perf_counter()
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
    step()
wall=(time.perf_counter()-t0)*1000
host_mm_us=0.0; host_mm_n=0; dev_by=collections.defaultdict(float); dev_total=0.0
for e in p.key_averages():
    if e.key=="aten::mm":
        host_mm_us+=e.cpu_time_total; host_mm_n+=e.count   # host-side dispatch (cpu_time)
    if str(getattr(e,"device_type","")).endswith("CUDA") or e.device_time_total>0:
        dev_by[e.key[:60]]+=e.device_time_total/1000.0
dev_total=sum(dev_by.values())
top=sorted(dev_by.items(),key=lambda kv:-kv[1])[:6]
res={"mode":MODE,"wall_ms_with_profiler":round(wall,1),"note":"profiler ON: diagnostic only",
     "host_aten_mm":{"count":host_mm_n,"cpu_time_total_ms":round(host_mm_us/1000.0,2)},
     "device_kernel_total_ms":round(dev_total,1),"host_gap_ms":round(wall-dev_total,1),
     "top_device_kernels":[[k,round(v,1)] for k,v in top]}
json.dump(res,open(a.out,"w"),indent=1)
print(json.dumps(res,indent=1))
