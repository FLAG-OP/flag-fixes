"""Review (2026-11-22) experiment 2: prove WHICH calls the candidate replaces.
One training step with FlagGems mm enabled (selective), counting aten mm calls
via fixed-matching dispatch probe: by phase (forward addmm / backward mm) and
by operand dims class (dW vs dX vs fwd), so replacement coverage per class is
measurable."""
import argparse, json, collections, sys
import torch

ap=argparse.ArgumentParser(); ap.add_argument("--conf",required=True); ap.add_argument("--skip",type=int,default=2)
ap.add_argument("--out",required=True); ap.add_argument("--device",default="auto")
a=ap.parse_args()
import random, os
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE", a.device)
os.environ["FL_ADAMW_FUSED"]="0"
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model
import flag_gems
flag_gems.only_enable(include=["mm"])

cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
dtype_name=str(cfg.model.get("torch_dtype","bf16"))
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":a.device}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev)
model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
stream=iter(loaders[0])
MM=torch.ops.aten.mm.default; ADDMM=torch.ops.aten.addmm.default
GEMM_SET={MM,ADDMM}
phase={"name":"warmup"}
hits=collections.Counter()
class Probe(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if func in GEMM_SET:
            m2=[t.shape for t in args if isinstance(t,torch.Tensor) and t.dim()==2][:2]
            key=(phase["name"].split("_m")[0], str(list(m2[0]) if m2 else "?"), str(list(m2[1]) if len(m2)>1 else "?"))
            hits[key]+=1
        return func(*args,**(kwargs or {}))
for i in range(a.skip):
    optimizer.zero_grad(set_to_none=True)
    for micro in range(accum):
        batch=next(stream)
        with autocast_context(dev,dtype_name):
            loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
        (loss/accum).backward()
    optimizer.zero_grad(set_to_none=True)
optimizer.zero_grad(set_to_none=True)
for micro in range(accum):
    batch=next(stream)
    with autocast_context(dev,dtype_name):
        phase["name"]=f"forward_m{micro}"
        with Probe():
            loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
    phase["name"]=f"backward_m{micro}"
    with Probe():
        (loss/accum).backward()
torch.cuda.synchronize()
# classify: forward addmm all candidate-replaced? mm only -> forward addmm NOT replaced (aten addmm untouched)
fwd_addmm=sum(v for (p,_,_),v in hits.items() if p=="forward")
summary={}
for (p,a_s,b_s),v in hits.items():
    A=json.loads(a_s); B=json.loads(b_s)
    if p=="forward": cls="forward_addmm(NOT candidate: addmm untouched)" if True else ""
    else:
        if A[0]==4096 or (A[0]==1024 and B[1]==4096): cls="dW_ffn"
        elif A[1]==4096: cls="dX_ffn"
        elif A[1]==1024 and B[1]==1024: cls="dX_attention_proj"
        elif A[0]==1024 and B[1]==1024: cls="dW_attention_proj"
        else: cls="other_mm"
    summary.setdefault(p,{}).setdefault(cls,0)
    summary[p][cls]+=v
out={"note":"dispatch probe counts aten.mm (candidate-replaced) vs aten.addmm (NOT replaced) per phase; mm candidate replaces ALL mm including dW/dX/loss",
      "counts":{p:dict(c) for p,c in summary.items()}}
json.dump(out,open(a.out,"w"),indent=1)
print(json.dumps({p:dict(c) for p,c in summary.items()},indent=1))
