"""Review BqnWy6w item 2: tokenize-overlap experiment REDONE on the original
training entry. Fixes vs the discarded round-52 version:
  - explicit batch["anchor"] / batch["positive"] / negatives startswith("negative")
    (the old sorted() made negative_1 the positive)
  - zero_grad ONCE per optimizer step; 4 micro-batches accumulate; THEN step
  - identical loss to flagscale.train.losses.retrieval.embedding_loss
    (sentence_embedding -> F.normalize(float) -> cross_entropy on grouped logits)
  - the two modes differ in EXECUTION ORDER (verified by timestamps):
      inline : per group, tokenize -> H2D -> [launch fwd] ; host blocks on launch
      overlap: a HOST WORKER THREAD tokenizes group i+1 while the device runs
               group i; H2D of i+1 is issued after i's launch
  - per-micro timestamps for tokenize / H2D / device compute and measured overlap
  - all 4 micro-batches' input hashes + token counts recorded
Same snapshot/inputs/BF16/warmup in both modes."""
import argparse, json, os, sys, time, hashlib, threading, queue
import torch
import torch.nn.functional as F
ap=argparse.ArgumentParser()
ap.add_argument("--conf",required=True); ap.add_argument("--mode",required=True,choices=["inline","overlap"])
ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--out",required=True); a=ap.parse_args()
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE","cuda"); os.environ["FL_ADAMW_FUSED"]="0"
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.train.losses.retrieval import move_features
from flagscale.models.retrieval import build_retrieval_model
from transformers import AutoTokenizer
cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps); temp=float(cfg.temperature)
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":"cuda"}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev); model.train()
base=(model.module if hasattr(model,"module") else model)
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
stream=iter(loaders[0])
tok=AutoTokenizer.from_pretrained(str(cfg.model.model_path))
ML=int(cfg.model.get("max_length",8192))
def now(): return time.perf_counter()*1000.0
def tokenize(group):
    t0=now(); f=base.prepare_embedding_inputs(group); t1=now()
    return f,t0,t1
def h2d(feats):
    t0=now(); x=move_features(feats,dev); t1=now(); return x,t0,t1
def compute(feats):
    """identical to embedding_batch: sentence_embedding -> F.normalize(float)"""
    t0=now()
    with autocast_context(dev,"bf16"):
        e=model(feats).get("sentence_embedding")
    torch.cuda.synchronize()                 # makes the device interval observable
    t1=now()
    return F.normalize(e.float(),p=2,dim=-1),t0,t1
def loss_of(anchors,positives,negatives):
    grouped=torch.stack([positives,*negatives],dim=1).reshape(anchors.shape[0]*(1+len(negatives)),-1)
    logits=anchors@grouped.T/temp
    labels=torch.arange(anchors.shape[0],device=dev)*(1+len(negatives))
    return F.cross_entropy(logits,labels)
ORDER=["anchor","positive","negative_1","negative_2","negative_3"]   # explicit, matches the real loss (keys are 1-based)
def micro_inline(batch,rec):
    outs={}; ev=[]
    for k in ORDER:
        feats,tt0,tt1=tokenize(batch[k]); x,h0,h1=h2d(feats); y,c0,c1=compute(x)
        outs[k]=y; ev.append({"group":k,"tokenize":[round(tt0,3),round(tt1,3)],"h2d":[round(h0,3),round(h1,3)],"compute":[round(c0,3),round(c1,3)],"overlap_ms":0.0})
    return loss_of(outs["anchor"],outs["positive"],[outs[n] for n in ORDER[2:]]),ev
def micro_overlap(batch,rec):
    """host worker thread tokenizes group i+1 while the device runs group i.
    Only timestamps are recorded here; overlap is computed AFTERWARDS from the
    interval intersection (tokenize[i+1] vs compute[i]) - no queue replay."""
    outs={}; ev=[]; q=queue.Queue()
    def worker(k): q.put((k,)+tokenize(batch[k]))
    feats,tt0,tt1=tokenize(batch[ORDER[0]])
    th=threading.Thread(target=worker,args=(ORDER[1],)); th.start()
    for i,k in enumerate(ORDER):
        if i>0:
            kk,feats,tt0,tt1=q.get(); th.join()
        x,h0,h1=h2d(feats)
        if i+2<len(ORDER):
            th=threading.Thread(target=worker,args=(ORDER[i+2],)); th.start()
        y,c0,c1=compute(x); outs[k]=y
        ev.append({"group":k,"tokenize":[round(tt0,3),round(tt1,3)],"h2d":[round(h0,3),round(h1,3)],"compute":[round(c0,3),round(c1,3)]})
    return loss_of(outs["anchor"],outs["positive"],[outs[n] for n in ORDER[2:]]),ev
def step(rec):
    t0=now(); optimizer.zero_grad(set_to_none=True)     # ONCE per step
    evs=[]; losses=[]
    for micro in range(accum):
        batch=next(stream)
        loss,ev=(micro_inline if a.mode=="inline" else micro_overlap)(batch,rec)
        (loss/accum).backward()                          # accumulate 4 micros
        losses.append(float(loss.detach())); evs.append({"micro":micro,"events":ev,
            "input_hash":hashlib.sha256("|".join("|".join(batch[k]) for k in ORDER).encode()).hexdigest()[:16],
            "valid_tokens":sum(sum(sum(1 for z in m if z==1) for m in tok(batch[k],padding=True,truncation=True,max_length=ML,return_tensors=None)["attention_mask"]) for k in ORDER)})
    torch.nn.utils.clip_grad_norm_(trainable,1.0); optimizer.step(); torch.cuda.synchronize()
    return round(now()-t0,1),losses,evs
for _ in range(2): step(False)
rounds=[step(True) for _ in range(a.rounds)]
def overlap_of(evs):
    """sum over groups of |tokenize[i+1] ∩ compute[i]| (host tokenize hidden by device compute)"""
    tot=0.0
    for a_,b_ in zip(evs,evs[1:]):
        n0,n1=b_["tokenize"]; c0,c1=a_["compute"]
        tot+=max(0.0,min(n1,c1)-max(n0,c0))
    return round(tot,3)
rounds_out=[]
for m,l,e in rounds:
    for mi in e: mi["measured_overlap_ms"]=overlap_of(mi["events"])
    rounds_out.append({"step_ms":m,"micro_losses":[round(x,6) for x in l],"micro_events":e})
ov=sum(mi["measured_overlap_ms"] for r in rounds_out for mi in r["micro_events"])
alll=[x for m,l,e in rounds for x in l]
res={"mode":a.mode,
     "loss_semantics":"identical to embedding_loss (anchor/positive/negative_1..3, F.normalize, grouped CE)",
     "grad_accum":accum,"zero_grad":"once per step",
     "rounds":rounds_out,"measured_overlap_total_ms":round(ov,3),
     "loss_outliers":[round(x,4) for x in alll if x>1.5],
     "thread_safety_suspect":bool([x for x in alll if x>1.5])}
json.dump(res,open(a.out,"w"),indent=1)
print(f"[{a.mode}] {[r['step_ms'] for r in res['rounds']]} | loss0 {[round(r['micro_losses'][0],4) for r in res['rounds']]} | measured overlap total {round(ov,1)} ms | outliers {res['loss_outliers']}")
