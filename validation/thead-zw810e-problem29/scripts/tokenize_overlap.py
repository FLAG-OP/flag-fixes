"""Review bqj7-Bw training decision: falsifiable hypothesis from the gap analysis.
GAP EVIDENCE (gap_attribution_r51.json): each micro-batch begins with a 3-5 ms
device-idle gap that ends exactly when 'Memcpy HtoD (Pageable->Device)' starts and
begins when the previous kernel ends; no cpu_op runs at gap start. That is the
host-side tokenize (prepare_embedding_inputs) of the 5 text groups per micro-batch
blocking the device.
HYPOTHESIS: overlapping the tokenize of group i+1 with the device compute of
group i removes most of that gap.
SINGLE FACTOR CHANGED: only WHEN tokenize happens (overlapped vs inline). Same
model snapshot, same inputs, same tokenizer, same H2D, same math.
Modes: inline (current) | overlap (tokenize next group while current computes)."""
import argparse, json, os, sys, time, hashlib
import torch
import torch.nn.functional as F
ap=argparse.ArgumentParser()
ap.add_argument("--conf",required=True); ap.add_argument("--mode",required=True,choices=["inline","overlap"])
ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--out",required=True)
a=ap.parse_args()
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE","cuda"); os.environ["FL_ADAMW_FUSED"]="0"
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model
from flagscale.train.losses.retrieval import move_features
cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps); temp=float(cfg.temperature)
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":"cuda"}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev)
model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
stream=iter(loaders[0])
unwrap=lambda m: (m.module if hasattr(m,"module") else m)
def emb_of(feats):
    """identical to flagscale.train.losses.retrieval.embedding_batch tail:
    sentence_embedding -> F.normalize(float(), p=2, dim=-1)"""
    e=model(feats).get("sentence_embedding")
    return F.normalize(e.float(),p=2,dim=-1)
def encode_group(g):
    feats=move_features(unwrap(model).prepare_embedding_inputs(g),dev)
    return emb_of(feats)
def step_overlap(batch):
    """tokenize group i+1 on the host while group i computes on the device"""
    optimizer.zero_grad(set_to_none=True)
    keys=sorted(batch.keys())
    # prepare features for group 0 (unavoidable: nothing to overlap with yet)
    feats=[move_features(unwrap(model).prepare_embedding_inputs(batch[k]),dev) for k in keys[:1]]
    outs=[]
    for i,k in enumerate(keys):
        with autocast_context(dev,"bf16"):
            if i+1<len(keys):
                # launch group i compute WITHOUT syncing, then tokenize group i+1 on the host
                y=emb_of(feats[i])
                feats.append(move_features(unwrap(model).prepare_embedding_inputs(batch[keys[i+1]]),dev))
            else:
                y=emb_of(feats[i])
        outs.append(y)
    anchors,positives=outs[0],outs[1]; negatives=outs[2:]
    grouped=torch.stack([positives,*negatives],dim=1).reshape(anchors.shape[0]*(1+len(negatives)),-1)
    logits=anchors@grouped.T/temp
    labels=torch.arange(anchors.shape[0],device=dev)*(1+len(negatives))
    return F.cross_entropy(logits,labels)
def step_inline(batch):
    optimizer.zero_grad(set_to_none=True)
    outs=[]
    for k in sorted(batch.keys()):
        with autocast_context(dev,"bf16"):
            outs.append(encode_group(batch[k]))
    anchors,positives=outs[0],outs[1]; negatives=outs[2:]
    grouped=torch.stack([positives,*negatives],dim=1).reshape(anchors.shape[0]*(1+len(negatives)),-1)
    logits=anchors@grouped.T/temp
    labels=torch.arange(anchors.shape[0],device=dev)*(1+len(negatives))
    return F.cross_entropy(logits,labels)
from transformers import AutoTokenizer
_tok=AutoTokenizer.from_pretrained(str(cfg.model.model_path))
def step(rec):
    t0=time.perf_counter(); losses=[]
    for micro in range(accum):
        batch=next(stream)
        loss=(step_overlap if a.mode=="overlap" else step_inline)(batch)
        losses.append(float(loss.detach())); (loss/accum).backward()
    torch.nn.utils.clip_grad_norm_(trainable,1.0); optimizer.step(); torch.cuda.synchronize()
    ms=(time.perf_counter()-t0)*1000
    valid=sum(sum(sum(1 for z in m if z==1) for m in _tok(b[k],padding=True,truncation=True,max_length=int(cfg.model.get("max_length",8192)),return_tensors=None)["attention_mask"]) for b in [batch] for k in sorted(b.keys()))
    h=hashlib.sha256("|".join("|".join(batch[k]) for k in sorted(batch.keys())).encode()).hexdigest()[:16]
    return ms,losses,valid,h
for _ in range(2): step(False)
rounds=[step(True) for _ in range(a.rounds)]
res={"mode":a.mode,"rounds":[{"step_ms":round(m,1),"micro_losses":[round(x,6) for x in l],"valid_tokens":v,"tokens_per_s":round(v/(m/1000)),"input_hash":h} for m,l,v,h in rounds]}
json.dump(res,open(a.out,"w"),indent=1)
print(f"[{a.mode}] {[round(r['step_ms'],1) for r in res['rounds']]} | loss0 {[r['micro_losses'][0] for r in res['rounds']]}")
