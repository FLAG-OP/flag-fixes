"""Review 7166338004 item 1: full training-state consistency of a candidate
against the ORIGINAL training entry, measured in ONE process from ONE snapshot.

All runs (native x3, then candidate x2) start from the same snapshot taken after
2 warmup optimizer steps: model state, optimizer state, CPU RNG, CUDA RNG and the
4 micro-batches of the step. Each run executes one optimizer step exactly as the
entry loop does (flagscale/train/train_retrieval.py:646-673): loss under autocast
through the untouched flagscale.train.losses.retrieval.loss_for_batch, the entry's
isfinite(loss) host sync, backward, clip_grad_norm_, optimizer.step.

Compared after the step, per parameter tensor:
  gradients (after 4-micro accumulation, before clip), post-update parameters,
  optimizer state (exp_avg, exp_avg_sq, step) -> bitwise-identical count, max abs
  error, max relative error (floored at 1% of the tensor's max |value|),
  non-finite count, non-tensor state equality, SHA-256 of the parameter bytes.

Known device non-determinism: the three embedding tables (word / position /
token_type) get their gradient from an index-accumulating backward whose
summation order is not reproducible, so two NATIVE runs differ on exactly these
tensors. They are reported separately and excluded from the candidate verdict.

A cross-process design (candidate in another process) is NOT valid here: models
rebuilt from the same seed in different processes are not identical (0/72 tensors,
acc_hit_step_r46_rebuild_check), which showed up as fake grad max_rel 215.

--candidate dw_route              the 12 dW shapes routed to flag_gems.mm (aten::mm
                                  override, registered AFTER the native runs)
--candidate prefetch_faithful     the same loss_for_batch; only prepare_embedding_inputs
                                  + move_features run on a worker thread (by-group futures)
--candidate prefetch_loss_reimpl  the round-56 / round-57 run-1 prefetch implementation
                                  (model call under autocast, normalize / logits matmul /
                                  cross_entropy re-implemented OUTSIDE autocast)
--candidate prefetch_loss_reimpl_run1
                                  the same re-implementation invoked OUTSIDE autocast exactly as
                                  round-56 / run-1 did (only the model call under autocast), so the
                                  logits matmul runs in fp32 instead of the entry's autocast bf16
--candidate h2d_nonblocking       move_features through pinned buffers + non_blocking=True
                                  (same bytes, same kernels, same order)
"""
import argparse, copy, hashlib, json, os, sys, time, concurrent.futures
from collections.abc import Mapping
import torch
ap=argparse.ArgumentParser()
ap.add_argument("--conf")
ap.add_argument("--candidate",choices=["dw_route","prefetch_faithful","prefetch_loss_reimpl","prefetch_loss_reimpl_run1","h2d_nonblocking"])
ap.add_argument("--out",required=True,help="run mode: writes <out stem>.<candidate>.json; --merge: reads them all and writes <out>")
ap.add_argument("--merge",action="store_true")
ap.add_argument("--native-runs",type=int,default=3); ap.add_argument("--candidate-runs",type=int,default=2)
a=ap.parse_args()
if not a.merge and not (a.conf and a.candidate): ap.error("--conf and --candidate are required unless --merge")
STATIC={"field_version":"train_state_v2","time_unit":"ms",
 "design":"ONE process per candidate: after 2 warmup optimizer steps one snapshot (model, optimizer state, CPU+CUDA RNG, the 4 micro-batches) is taken; native x3 from the snapshot, then the candidate is installed and run x2 from the same snapshot. Each run = one optimizer step of the ORIGINAL entry loop (loss under autocast via the untouched loss_for_batch, isfinite(loss) host sync, backward, clip_grad_norm_, optimizer.step).",
 "rng_dropout_data_order":"CPU and CUDA RNG states are restored from the snapshot before every run (dropout masks identical); the 4 micro-batches are deep-copied in the snapshot (data order fixed); tokenization has no randomness; the prefetch worker thread consumes no torch RNG.",
 "thresholds":{"grad_rel_tol":0.02,"param_abs_tol":1e-3,"state_rel_tol":0.02,"nonfinite_max":0,"basis":"2% relative error floored at 1% of the tensor's max |value| (bf16 mantissa); params by absolute error; the verdict distinguishes bitwise identity from within-threshold"},
 "embedding_tables":"word / position / token_type embedding gradients come from an index-accumulating backward whose summation order is not reproducible on this device: native-vs-native differs on exactly these tensors in every native pair. Reported separately, excluded from the candidate verdict.",
 "cross_process_warning":"a candidate run in ANOTHER process is not a valid comparison: models rebuilt from the same seed in different processes are not identical (acc_hit_step_r46_rebuild_check 0/72). The first r57 attempt compared across processes and showed fake grad max_rel 215 / loss0 0.95186 vs 0.94985; discarded."}
if a.merge:
    import glob
    cands={}
    for f in sorted(glob.glob(a.out[:-5]+".*.json")):
        c=json.load(open(f)); cands[c["candidate"]]=c
    db=dict(STATIC); db["native_band_reference"]=cands["dw_route"]["native_band"]; db["dense_param_tensors"]=cands["dw_route"]["pairs"][0]["grads"]["dense"]["tensors"]
    db["candidates"]=cands
    json.dump(db,open(a.out,"w"),indent=1)
    print("merged",sorted(cands),"->",a.out); sys.exit(0)
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE","cuda"); os.environ["FL_ADAMW_FUSED"]="0"
import torch.nn.functional as F
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses import retrieval as Lmod
from flagscale.train.losses.retrieval import loss_for_batch, move_features
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
EMB=set(n for n,_ in model.named_parameters() if n.endswith(("word_embeddings.weight","position_embeddings.weight","token_type_embeddings.weight")))
def inline_loss(b): return loss_for_batch(model,processor,str(cfg.task),b,dev,TEMP)
OUTER_AUTOCAST=True
def micro_step(loss_fn,b):
    if OUTER_AUTOCAST:
        with autocast_context(dev,"bf16"):
            raw=loss_fn(b); loss=raw/accum
    else:
        raw=loss_fn(b); loss=raw/accum
    if not torch.isfinite(raw): raise FloatingPointError("Non-finite training loss")
    loss.backward(); return float(raw.detach())
_LIB=None; _EX=None; _FUT={}
if a.candidate=="dw_route":
    sys.path.insert(0,"/tmp/opencode/FlagGems-base/src"); import flag_gems
    FG=flag_gems.mm; MM_OUT=torch.ops.aten.mm.out
    N_MK=[(4096,5120,1024),(1024,5120,4096),(4096,4430,1024),(1024,4430,4096),(4096,4310,1024),(1024,4310,4096),
          (4096,3740,1024),(1024,3740,4096),(4096,3290,1024),(1024,3290,4096),(4096,2990,1024),(1024,2990,4096)]
    DW={((N,M),(M,K)) for (N,M,K) in N_MK}
    def route(self,other):
        if (tuple(self.shape),tuple(other.shape)) in DW: return FG(self,other)
        return MM_OUT(self,other,out=torch.empty(self.shape[0],other.shape[1],device=self.device,dtype=self.dtype))
    cand_loss=inline_loss
    def install():
        global _LIB
        _LIB=torch.library.Library("aten","IMPL"); _LIB.impl("mm",route,"CUDA")
elif a.candidate=="h2d_nonblocking":
    _HOLD=[]
    def _move_nb(features,device):
        if isinstance(features,Mapping): return {k:_move_nb(v,device) for k,v in features.items()}
        if isinstance(features,(list,tuple)): return type(features)(_move_nb(v,device) for v in features)
        if torch.is_tensor(features) and features.device.type!=device.type:
            buf=torch.empty_like(features,pin_memory=True); buf.copy_(features); _HOLD.append(buf)
            return buf.to(device,non_blocking=True)
        return move_features(features,device)
    cand_loss=inline_loss
    def install(): Lmod.move_features=_move_nb
else:
    _EX=concurrent.futures.ThreadPoolExecutor(max_workers=1)
    def _prep(texts): return move_features(base.prepare_embedding_inputs(texts),dev)
    if a.candidate=="prefetch_faithful":
        def _emb(m,inputs,device):
            f=_FUT.pop(id(inputs),None)
            feats=f.result() if f is not None else move_features(base.prepare_embedding_inputs(inputs),device)
            out=m(feats); e=out.get("sentence_embedding")
            if e is None: e=out.get("pooler_output")
            if e is None: raise RuntimeError(f"Retrieval model output has no embedding field: {list(out)}")
            return F.normalize(e.float(),p=2,dim=-1)
        Lmod.embedding_batch=_emb
        def cand_loss(b):
            keys=["anchor","positive"]+sorted(k for k in b if k.startswith("negative"))
            _FUT.clear()
            for k in keys: _FUT[id(b[k])]=_EX.submit(_prep,b[k])
            return loss_for_batch(model,processor,str(cfg.task),b,dev,TEMP)
    else:
        OUTER_AUTOCAST=(a.candidate!="prefetch_loss_reimpl_run1")
        def cand_loss(b):
            neg=sorted(k for k in b if k.startswith("negative")); keys=["anchor","positive"]+neg; fut={}; outs={}
            for k in keys[:2]: fut[k]=_EX.submit(_prep,b[k])
            for i,k in enumerate(keys):
                if i+2<len(keys) and keys[i+2] not in fut: fut[keys[i+2]]=_EX.submit(_prep,b[keys[i+2]])
                feats=fut[k].result()
                with autocast_context(dev,"bf16"): e=model(feats).get("sentence_embedding")
                outs[k]=F.normalize(e.float(),p=2,dim=-1)
            anchors,positives=outs["anchor"],outs["positive"]
            grouped=torch.stack([positives]+[outs[k] for k in neg],dim=1).reshape(anchors.shape[0]*(1+len(neg)),-1)
            return F.cross_entropy(anchors@grouped.T/TEMP,torch.arange(anchors.shape[0],device=dev)*(1+len(neg)))
    def install(): pass
def param_hash(params,skip):
    h=hashlib.sha256()
    for n in sorted(params):
        if skip and n in skip: continue
        t=params[n].contiguous(); h.update(n.encode()); h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def run(snap,loss_fn):
    model.load_state_dict(copy.deepcopy(snap["model"])); optimizer.load_state_dict(copy.deepcopy(snap["opt"]))
    torch.set_rng_state(snap["rng_cpu"]); torch.cuda.set_rng_state(snap["rng_cuda"])
    optimizer.zero_grad(set_to_none=True); t0=time.perf_counter()
    losses=[micro_step(loss_fn,b) for b in snap["batches"]]
    ms=(time.perf_counter()-t0)*1e3
    grads={n:p.grad.detach().to("cpu") for n,p in model.named_parameters() if p.grad is not None}
    gn=float(torch.nn.utils.clip_grad_norm_(trainable,GN)); optimizer.step()
    params={n:p.detach().to("cpu") for n,p in model.named_parameters()}
    state={n:{k:(v.detach().to("cpu") if torch.is_tensor(v) else v) for k,v in optimizer.state[p].items()} for n,p in model.named_parameters() if p in optimizer.state}
    _HOLD.clear() if a.candidate=="h2d_nonblocking" else None
    return {"step_ms":round(ms,1),"losses":losses,"grad_norm":gn,"grads":grads,"params":params,"state":state,
            "param_sha256_all":param_hash(params,None),"param_sha256_dense":param_hash(params,EMB)}
def cmp(x,y):
    if x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y): return 0.0,0.0,0
    p=x.float(); q=y.float(); e=(p-q).abs(); fl=0.01*float(q.abs().max())
    rel=float((e/torch.clamp(q.abs(),min=fl)).max()) if q.numel() else 0.0
    return float(e.max()),rel,int((~torch.isfinite(p)).sum())+int((~torch.isfinite(q)).sum())
def agg(Da,Db,skip=()):
    ma=mr=0.0; nf=0; n=0; over=0; diff=[]; ntm=0
    def one(x,y,name):
        nonlocal ma,mr,nf,n,over,ntm
        if isinstance(x,dict) and isinstance(y,dict):
            for k in x:
                if k in y: one(x[k],y[k],name+"."+str(k))
        elif torch.is_tensor(x) and torch.is_tensor(y):
            r=cmp(x,y); n+=1; ma=max(ma,r[0]); mr=max(mr,r[1]); nf+=r[2]; over+=(r[1]>0.02)
            if r[0] or r[1]: diff.append(name)
        elif type(x)!=type(y) or x!=y: ntm+=1
    for k in Da:
        if k in Db and k not in skip: one(Da[k],Db[k],k)
    return {"tensors":n,"bitwise_identical":n-len(diff),"max_abs":round(ma,8),"max_rel":round(mr,6),
            "over_2pct":over,"nonfinite":nf,"nontensor_mismatch":ntm,"n_differing":len(diff),"differing":sorted(diff)[:8]}
def pick(D,names): return {k:v for k,v in D.items() if k in names}
def pair(A,B,la,lb,kind):
    g={"dense":agg(A["grads"],B["grads"],EMB),"embedding_tables":agg(pick(A["grads"],EMB),pick(B["grads"],EMB))}
    p={"dense":agg(A["params"],B["params"],EMB),"embedding_tables":agg(pick(A["params"],EMB),pick(B["params"],EMB))}
    s={"dense":agg(A["state"],B["state"],EMB),"embedding_tables":agg(pick(A["state"],EMB),pick(B["state"],EMB))}
    return {"kind":kind,"a":la,"b":lb,"loss_equal":A["losses"]==B["losses"],
            "loss_max_abs":max(abs(x-y) for x,y in zip(A["losses"],B["losses"])),
            "grad_norm":[A["grad_norm"],B["grad_norm"]],"grads":g,"params":p,"optimizer_state":s,
            "param_sha256_dense_equal":A["param_sha256_dense"]==B["param_sha256_dense"],
            "param_sha256_all_equal":A["param_sha256_all"]==B["param_sha256_all"]}
for _ in range(2):
    optimizer.zero_grad(set_to_none=True)
    for m in range(accum): micro_step(inline_loss,next(stream))
batches=[next(stream) for _ in range(accum)]
snap={"model":copy.deepcopy(model.state_dict()),"opt":copy.deepcopy(optimizer.state_dict()),
      "rng_cpu":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state(),"batches":copy.deepcopy(batches)}
NAT=[run(snap,inline_loss) for _ in range(a.native_runs)]
install()
CAND=[run(snap,cand_loss) for _ in range(a.candidate_runs)]
pairs=[]
for i in range(len(NAT)):
    for j in range(i+1,len(NAT)): pairs.append(pair(NAT[i],NAT[j],f"native{i}",f"native{j}","native_vs_native"))
for c in range(len(CAND)):
    for n in range(len(NAT)): pairs.append(pair(CAND[c],NAT[n],f"{a.candidate}{c}",f"native{n}","candidate_vs_native"))
for i in range(len(CAND)):
    for j in range(i+1,len(CAND)): pairs.append(pair(CAND[i],CAND[j],f"{a.candidate}{i}",f"{a.candidate}{j}","candidate_vs_candidate"))
cv=[p for p in pairs if p["kind"]!="native_vs_native"]
def dense_ok(p,bitwise):
    for sec in ("grads","params","optimizer_state"):
        d=p[sec]["dense"]
        if d["nonfinite"]: return False
        if bitwise and d["bitwise_identical"]!=d["tensors"]: return False
    return True
def within(p):
    return (dense_ok(p,False) and p["grads"]["dense"]["max_rel"]<=0.02 and p["params"]["dense"]["max_abs"]<=1e-3
            and p["optimizer_state"]["dense"]["max_rel"]<=0.02 and p["grads"]["embedding_tables"]["nonfinite"]==0
            and p["params"]["embedding_tables"]["nonfinite"]==0 and p["optimizer_state"]["dense"]["nontensor_mismatch"]==0
            and p["loss_equal"] and p["grad_norm"][0]==p["grad_norm"][1])
bit=all(dense_ok(p,True) and p["loss_equal"] and p["grad_norm"][0]==p["grad_norm"][1] for p in cv)
win=all(within(p) for p in cv)
gd=max(p["grads"]["dense"]["n_differing"] for p in cv); pd=max(p["params"]["dense"]["n_differing"] for p in cv)
nd=next(p["grads"]["dense"]["tensors"] for p in cv if p["kind"]=="candidate_vs_native")
nv=[p for p in pairs if p["kind"]=="native_vs_native"]
res={"candidate":a.candidate,"native_runs":len(NAT),"candidate_runs":len(CAND),"pairs":pairs,
     "dense_param_tensors":nd,"embedding_tables_excluded":sorted(EMB),
     "native_band":{"pairs":len(nv),"dense_tensors_identical_in_all_pairs":all(p["grads"]["dense"]["n_differing"]==0 for p in nv),
                    "pairs_with_embedding_table_diffs":sum(p["grads"]["embedding_tables"]["n_differing"]>0 for p in nv),
                    "embedding_table_grad_max_abs":max(p["grads"]["embedding_tables"]["max_abs"] for p in nv)},
     "candidate_pairs":len(cv),"dense_grad_tensors_differing_max_per_pair":gd,"dense_param_tensors_differing_max_per_pair":pd,
     "dense_grad_max_abs":max(p["grads"]["dense"]["max_abs"] for p in cv),"dense_grad_max_rel":max(p["grads"]["dense"]["max_rel"] for p in cv),
     "dense_param_max_abs":max(p["params"]["dense"]["max_abs"] for p in cv),"dense_optimizer_state_max_abs":max(p["optimizer_state"]["dense"]["max_abs"] for p in cv),
     "loss_equal_all_pairs":all(p["loss_equal"] for p in cv),"grad_norm_equal_all_pairs":all(p["grad_norm"][0]==p["grad_norm"][1] for p in cv),
     "param_sha256_dense_equal_all_pairs":all(p["param_sha256_dense_equal"] for p in cv),
     "param_sha256_dense":[r["param_sha256_dense"] for r in NAT+CAND],"step_ms":[r["step_ms"] for r in NAT+CAND],
     "losses_micro":[r["losses"] for r in NAT+CAND],"grad_norm":[r["grad_norm"] for r in NAT+CAND]}
if bit: res["verdict"]="PASS (bitwise): in all %d candidate pairs every dense gradient / post-update parameter / optimizer-state tensor is bitwise identical to the same-process native runs; losses and grad norms equal; 0 non-finite. Only the %d embedding tables differ, and native-vs-native differs on the same tables (device accumulation-order non-determinism)."%(len(cv),len(EMB))
elif win: res["verdict"]="PASS (within declared thresholds, NOT bitwise): dense grad max_rel %g, param max_abs %g over %d candidate pairs; %d dense grad tensors differ."%(res["dense_grad_max_rel"],res["dense_param_max_abs"],len(cv),gd)
else: res["verdict"]="FAIL: candidate is not within the declared thresholds in all candidate pairs (dense grad tensors differing up to %d per pair, grad max_rel %g, param max_abs %g)."%(gd,res["dense_grad_max_rel"],res["dense_param_max_abs"])
json.dump(dict(STATIC,**res),open(a.out[:-5]+"."+a.candidate+".json","w"),indent=1)
print(a.candidate,"->",res["verdict"][:230])
print("  native band:",res["native_band"]); print("  candidate: dense grad differing<=%d, params differing<=%d, state max_abs %g | losses equal %s | gn equal %s"%(gd,pd,res["dense_optimizer_state_max_abs"],res["loss_equal_all_pairs"],res["grad_norm_equal_all_pairs"]))
