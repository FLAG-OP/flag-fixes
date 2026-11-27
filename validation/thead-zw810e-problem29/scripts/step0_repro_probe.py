"""Why does the FIRST optimizer update of the entry loop give different parameters in different
processes? (follow-up to review 7182886976 item 2; round-58 diagnostic)

Observation it explains: route_from_step0_consistency.py shows identical step-0 losses and RNG
states in every process, but different dense-parameter hashes right after the first update.

Each process (native path, entry-loop step, NO host access to the gradients before clip):
  1 step on batches 0-3; after optimizer.step() copy to the host: the clipped gradients (exactly
  what AdamW consumed), the parameters, the optimizer state.
--compare <dir> : load the saved tensors of all processes in <dir> and compare them pairwise.
"""
import argparse, glob, hashlib, json, os, sys, time
import torch
ap=argparse.ArgumentParser()
ap.add_argument("--conf"); ap.add_argument("--out"); ap.add_argument("--tag",default="p1"); ap.add_argument("--compare")
a=ap.parse_args()
def stats(x,y):
    """elementwise comparison of two same-shape tensors"""
    x,y=x.float(),y.float(); eq=(x==y); d=(x-y).abs()
    nz=(x!=0)|(y!=0)
    signflip=int(((x>0)&(y<0)).sum()+((x<0)&(y>0)).sum())
    rel=(d[nz]/torch.minimum(x[nz].abs(),y[nz].abs()).clamp_min(1e-30)) if int(nz.sum()) else torch.zeros(1)
    return {"numel":x.numel(),"differing":int((~eq).sum()),"sign_flips":signflip,"max_abs":float(d.max()),
            "max_rel":float(rel.max()) if rel.numel() else 0.0,"n_rel_gt_10pct":int((rel>0.10).sum()) if rel.numel() else 0,
            "max_abs_x":float(x.abs().max())}
if a.compare:
    files=sorted(glob.glob(os.path.join(a.compare,"*.pt"))); R=[torch.load(f) for f in files]
    out={"field_version":"step0_repro_v1","design":__doc__,"files":[os.path.basename(f) for f in files],"n_processes":len(R),"meta_first_process":R[0].get("meta"),
         "step0_losses_per_process":[r["losses"] for r in R],"step0_grad_norm_per_process":[r["grad_norm"] for r in R],
         "dense_param_sha256_after_step0_per_process":[r["meta"]["param_sha256_dense"] for r in R],"pairs":[]}
    for i in range(len(R)):
        for j in range(i+1,len(R)):
            pr={"a":os.path.basename(files[i]),"b":os.path.basename(files[j]),"sections":{}}
            for sec in ("grads_clipped","params","exp_avg","exp_avg_sq"):
                per={}; tot={"numel":0,"differing":0,"sign_flips":0,"n_rel_gt_10pct":0}
                top=[]
                for k in R[i][sec]:
                    s=stats(R[i][sec][k],R[j][sec][k])
                    if k.endswith(("word_embeddings.weight","position_embeddings.weight","token_type_embeddings.weight")): continue
                    per[k]=s
                    for f_ in ("numel","differing","sign_flips","n_rel_gt_10pct"): tot[f_]+=s[f_]
                    top.append((s["differing"],s["sign_flips"],s["max_rel"],k))
                top.sort(reverse=True)
                pr["sections"][sec]={"dense_totals":tot,"tensors":len(per),"tensors_with_differences":sum(1 for s in per.values() if s["differing"]),
                                     "max_abs":max(s["max_abs"] for s in per.values()),"max_rel":max(s["max_rel"] for s in per.values()),
                                     "top_differing":[{"name":k,"differing":d_,"sign_flips":sf,"max_rel":round(r,4)} for d_,sf,r,k in top[:6]]}
            out["pairs"].append(pr)
    json.dump(out,open(a.out,"w"),indent=1)
    for p in out["pairs"]:
        print(p["a"],"vs",p["b"])
        for sec,s in p["sections"].items(): print(f"   {sec:<15} dense differing elements {s['dense_totals']['differing']} / {s['dense_totals']['numel']} ({100*s['dense_totals']['differing']/max(1,s['dense_totals']['numel']):.4f}%) | sign flips {s['dense_totals']['sign_flips']} | rel>10% {s['dense_totals']['n_rel_gt_10pct']} | tensors with diffs {s['tensors_with_differences']}/{s['tensors']} | max_abs {s['max_abs']:.3g} max_rel {s['max_rel']:.3g}")
    sys.exit(0)
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE","cuda"); os.environ["FL_ADAMW_FUSED"]="0"
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model
EMB_SUFFIX=("word_embeddings.weight","position_embeddings.weight","token_type_embeddings.weight")
cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
TEMP=float(cfg.temperature); GN=float(cfg.optimizer.get("max_grad_norm",1.0))
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":"cuda"}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev); model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
def micro_step(b):
    with autocast_context(dev,"bf16"):
        raw=loss_for_batch(model,processor,str(cfg.task),b,dev,TEMP); loss=raw/accum
    if not torch.isfinite(raw): raise FloatingPointError("Non-finite training loss")
    loss.backward(); return float(raw.detach())
def to_cpu(t): return t.detach().to("cpu")
stream=iter(loaders[0]); optimizer.zero_grad(set_to_none=True); t0=time.time()
losses=[micro_step(next(stream)) for _ in range(accum)]
gn=float(torch.nn.utils.clip_grad_norm_(trainable,GN)); optimizer.step()
S={"losses":losses,"grad_norm":gn}
S["grads_clipped"]={n:to_cpu(p.grad) for n,p in model.named_parameters() if p.grad is not None}   # copied AFTER the step: no host access before clip
S["params"]={n:to_cpu(p) for n,p in model.named_parameters()}
S["exp_avg"]={n:to_cpu(optimizer.state[p]["exp_avg"]) for n,p in model.named_parameters() if p in optimizer.state}
S["exp_avg_sq"]={n:to_cpu(optimizer.state[p]["exp_avg_sq"]) for n,p in model.named_parameters() if p in optimizer.state}
dense=sorted(n for n in S["params"] if not n.endswith(EMB_SUFFIX))
S["meta"]={"tag":a.tag,"pid":os.getpid(),"started_unix":t0,
           "dtypes":{"param":str(S["params"][dense[0]].dtype),"grad":str(S["grads_clipped"][dense[0]].dtype),"exp_avg":str(S["exp_avg"][dense[0]].dtype),"exp_avg_sq":str(S["exp_avg_sq"][dense[0]].dtype)},
           "optimizer":type(optimizer).__name__,
           "param_sha256_dense":hashlib.sha256(b"".join(S["params"][n].contiguous().view(torch.uint8).numpy().tobytes() for n in dense)).hexdigest()[:16]}
torch.save(S,a.out)
print("step0 probe",a.tag,"| losses",losses,"| grad norm",gn,"| dense param sha",S["meta"]["param_sha256_dense"],"| dtypes",S["meta"]["dtypes"],"| optimizer",S["meta"]["optimizer"])
