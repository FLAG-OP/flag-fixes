"""Review bqQqCnw item 1a: locate the SPECIFIED S=214 batch by text fingerprint
(step_2 / micro 0 / negative_2, loader_index=2, hash 4d84ea53dba8924d, 10 seqs,
1,356 valid tokens) from the original training entry; capture Q/K/V, per-sample
additive mask and the REAL dO (grad flowing back into the SDPA output) with
shape/dtype/stride for each. Saves tensors + manifest with input hashes."""
import argparse, hashlib, json, os, sys
import torch
import torch.nn.functional as F
ap=argparse.ArgumentParser()
ap.add_argument("--conf",required=True); ap.add_argument("--out-prefix",required=True)
ap.add_argument("--target-hash",default="4d84ea53dba8924d")
ap.add_argument("--step",type=int,default=2); ap.add_argument("--micro",type=int,default=0)
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
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":"cuda"}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev); model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
stream=iter(loaders[0])
def h16(texts): return hashlib.sha256("|".join(texts).encode()).hexdigest()[:16]

# advance: (step, micro) pairs before the target
target=(a.step,a.micro); idx=0; target_batch=None; loader_idx=0
for s in range(a.step+1):
    optimizer.zero_grad(set_to_none=True)
    for m in range(accum):
        batch=next(stream)
        if (s,m)==target:
            target_batch=batch
            # verify fingerprint on the negative group
            found=None
            for k in sorted(batch.keys()):
                if h16(batch[k])==a.target_hash: found=k
            print(f"target step {s} micro {m}: fingerprint {a.target_hash} found in group '{found}' | loader pair index {idx}", flush=True)
            break
        with autocast_context(dev,"bf16"):
            loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
        (loss/accum).backward()
        idx+=1
    if target_batch is not None: break
    torch.nn.utils.clip_grad_norm_(trainable,1.0); optimizer.step()

# capture Q/K/V/mask at the SDPA whose key-group S matches the target group's length,
# and the REAL dO via a tensor hook on the output
_orig=F.scaled_dot_product_attention; cap={}
group_key=found
from transformers import AutoTokenizer
_tok=AutoTokenizer.from_pretrained(str(cfg.model.model_path))
_enc=_tok(target_batch[group_key],padding=True,truncation=True,max_length=int(cfg.model.get("max_length",8192)),return_tensors=None)
cap["target_S"]=len(_enc["attention_mask"][0])
print("target group token S =",cap["target_S"],"| valid tokens =",sum(sum(1 for x in mm if x==1) for mm in _enc["attention_mask"]),flush=True)
def spy(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False,**kw):
    y=_orig(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal,**kw)
    # capture the FIRST layer's call for the group whose padded S equals the target group's S
    S=q.shape[2]
    # match the TARGET group's token length (resolved from its tokenized input),
    # not merely the first short-S call (the anchor group at S=16 comes first)
    if "q" not in cap and attn_mask is not None and attn_mask.shape[-1]==S and S==cap.get("target_S"):
        cap["layer_idx"]=cap.get("layer_idx",0)
        cap["q"]=q.detach().clone(); cap["k"]=k.detach().clone(); cap["v"]=v.detach().clone()
        cap["m"]=attn_mask.detach().clone(); cap["S"]=S; cap["dp"]=dropout_p
        cap["qs"]=tuple(q.stride()); cap["ks"]=tuple(k.stride()); cap["vs"]=tuple(v.stride()); cap["ms"]=tuple(attn_mask.stride())
        cap["md"]=str(attn_mask.dtype)
        def hook(g):
            cap["dO"]=g.detach().clone(); cap["dOs"]=tuple(g.stride()); cap["dOdt"]=str(g.dtype)
        y.register_hook(hook)
    return y
F.scaled_dot_product_attention=spy
optimizer.zero_grad(set_to_none=True)
for m in range(accum):
    batch=next(stream) if m!=a.micro else target_batch
    with autocast_context(dev,"bf16"):
        loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
    (loss/accum).backward()
F.scaled_dot_product_attention=_orig
torch.cuda.synchronize()
ok=all(k in cap for k in ("q","k","v","m","dO"))
man={"target":{"step":a.step,"micro":a.micro,"group":group_key,"text_hash16":a.target_hash,"loader_pair_index":idx,
      "n_seqs":len(target_batch[group_key]),"S_captured":cap.get("S")},
     "captured":ok,
     "q":{"shape":list(cap["q"].shape),"dtype":str(cap["q"].dtype),"stride":list(cap["qs"]),"contig":cap["q"].is_contiguous()},
     "k":{"shape":list(cap["k"].shape),"dtype":str(cap["k"].dtype),"stride":list(cap["ks"]),"contig":cap["k"].is_contiguous()},
     "v":{"shape":list(cap["v"].shape),"dtype":str(cap["v"].dtype),"stride":list(cap["vs"]),"contig":cap["v"].is_contiguous()},
     "mask":{"shape":list(cap["m"].shape),"dtype":cap["md"],"stride":list(cap["ms"]),"min":float(cap["m"].min()),
             "valid_frac":round(float((cap["m"]==0).float().mean()),4),"n_neg_inf":int((cap["m"]==float("-inf")).sum())},
     "dropout_p":cap.get("dp"),
     "dO":({"shape":list(cap["dO"].shape),"dtype":cap["dOdt"],"stride":list(cap["dOs"]),"contig":cap["dO"].is_contiguous(),
            "absmax":float(cap["dO"].abs().max())} if "dO" in cap else None),
     "note":"captured from the FIRST encoder layer's SDPA call at this S in the target micro; dO = real upstream grad via output tensor hook"}
if ok:
    torch.save({"q":cap["q"],"k":cap["k"],"v":cap["v"],"m":cap["m"],"dO":cap["dO"]},a.out_prefix+".pt")
    def th(t): return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
    man["tensor_hashes"]={k:th(cap[k]) for k in ("q","k","v","m","dO")}
json.dump(man,open(a.out_prefix+"_manifest.json","w"),indent=1)
print(json.dumps(man,indent=1))
