"""Is the step-1 forward a reproducible function of the model state? (round-58 diagnostic)

Observation to explain: after the first optimizer update the dense parameters of two processes
differ in only ~0.0007% of the elements (1 bf16 ulp each, step0_repro_r58.json) and the CUDA / CPU
RNG states are hash-identical, yet the step-1 losses differ by up to ~1% between processes.

--save <file> : one process: entry-loop step 0 (batches 0-3) with the native path, then save
                (parameters, optimizer state, CPU+CUDA RNG); then run the step-1 forward (batches
                4-7, no update) in the SAME process and print its losses ("in-process history").
--load <file> : a FRESH process: build the model, restore the saved state, run the step-1 forward
                twice (restoring the state again before the second run) and print both results.
If the fresh processes agree bit by bit, the step-1 forward is a function of the state and the
cross-process spread seen in from-step-0 runs comes from the tiny state differences; if they do
not, the forward depends on process history beyond (parameters, RNG, data).
"""
import argparse, copy, hashlib, json, os, sys, time
import torch
ap=argparse.ArgumentParser()
ap.add_argument("--conf",required=True); ap.add_argument("--save"); ap.add_argument("--load"); ap.add_argument("--tag",default="x"); ap.add_argument("--out")
ap.add_argument("--perturb",type=int,default=0,help="--load only: before the FIRST replay move this many random dense parameter elements by exactly one bf16 ulp (away from zero), mimicking the cross-process difference after one update (seed 0)")
a=ap.parse_args()
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE","cuda"); os.environ["FL_ADAMW_FUSED"]="0"
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model
cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
TEMP=float(cfg.temperature); GN=float(cfg.optimizer.get("max_grad_norm",1.0))
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":"cuda"}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev); model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
def micro(b,update):
    with autocast_context(dev,"bf16"):
        raw=loss_for_batch(model,processor,str(cfg.task),b,dev,TEMP); loss=raw/accum
    if not torch.isfinite(raw): raise FloatingPointError("Non-finite training loss")
    if update: loss.backward()
    return float(raw.detach())
def batches(skip,n):
    s=iter(loaders[0]); [next(s) for _ in range(skip)]; return [next(s) for _ in range(n)]
def snap(): return {"params":copy.deepcopy(model.state_dict()),"opt":copy.deepcopy(optimizer.state_dict()),"rng_cpu":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state()}
def restore(s):
    model.load_state_dict(copy.deepcopy(s["params"])); optimizer.load_state_dict(copy.deepcopy(s["opt"]))
    torch.set_rng_state(s["rng_cpu"]); torch.cuda.set_rng_state(s["rng_cuda"])
def dsha():
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if not n.endswith(("word_embeddings.weight","position_embeddings.weight","token_type_embeddings.weight")): h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]
def bh(b):
    h=hashlib.sha256()
    for k in sorted(b): h.update("|".join(b[k]).encode())
    return h.hexdigest()[:16]
res={"tag":a.tag,"pid":os.getpid(),"started_unix":time.time()}
if a.save:
    optimizer.zero_grad(set_to_none=True); b0=batches(0,accum)
    res["step0_losses"]=[micro(b,True) for b in b0]
    res["step0_grad_norm"]=float(torch.nn.utils.clip_grad_norm_(trainable,GN)); optimizer.step()
    S=snap(); torch.save(S,a.save)
    res["step0_batch_hashes"]=[bh(b) for b in b0]; res["state_dense_param_sha256"]=dsha()
    b1=batches(accum,accum); res["step1_batch_hashes"]=[bh(b) for b in b1]
    res["step1_losses_in_process_history"]=[micro(b,False) for b in b1]
    print("SAVE",a.tag,"| step0 losses",res["step0_losses"],"| gn",res["step0_grad_norm"],"| state sha",res["state_dense_param_sha256"],"| step1 losses (same process)",res["step1_losses_in_process_history"])
else:
    S=torch.load(a.load); b1=batches(accum,accum); res["step1_batch_hashes"]=[bh(b) for b in b1]
    out=[]
    for rep in range(2):
        restore(S); res["state_dense_param_sha256"]=dsha()
        if rep==0 and a.perturb>0:
            g=torch.Generator().manual_seed(0); dense=[(n,p) for n,p in model.named_parameters() if not n.endswith(("word_embeddings.weight","position_embeddings.weight","token_type_embeddings.weight"))]
            flat=torch.cat([p.detach().flatten() for _,p in dense]); k=a.perturb
            idx=torch.randint(0,flat.numel(),(k,),generator=g).to(dev)
            v=flat.view(-1)[idx]; bits=v.view(torch.int16)
            v.new_empty(0); flat.view(-1)[idx]=(bits+torch.where(bits<0,torch.tensor(-1,dtype=torch.int16,device=dev),torch.tensor(1,dtype=torch.int16,device=dev))).view(torch.bfloat16)
            off=0
            for n,p in dense:
                c=p.numel(); p.data.copy_(flat[off:off+c].view_as(p)); off+=c
            res["perturbed_elements"]=k; res["perturbed_state_dense_param_sha256"]=dsha()
        out.append([micro(b,False) for b in b1])
    res["step1_losses_fresh_process_replays"]=out
    res["replays_bitwise_equal"]=out[0]==out[1]
    print("LOAD",a.tag,"| perturbed",a.perturb,"| state sha",res["state_dense_param_sha256"],"| step1 losses replay1",out[0],"| replay2",out[1],"| replays equal",res["step1_losses_fresh_process_replays"][0]==out[1])
if a.out: json.dump(res,open(a.out,"w"),indent=1)
