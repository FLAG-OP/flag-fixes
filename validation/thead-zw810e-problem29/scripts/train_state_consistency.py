"""Training-state consistency of a candidate against the ORIGINAL training entry, measured
in ONE process from ONE snapshot (review 7182886976 item 1).

Warmup = 2 REAL optimizer updates of the entry loop (4 micro-batches with the loss under
autocast and the entry's isfinite(loss) host sync, clip_grad_norm_, optimizer.step). The
snapshot is taken after them, so the optimizer state carries AdamW step = 2 and momentum
(exp_avg, exp_avg_sq). The snapshot records parameter hashes, the optimizer step count, the
momentum state and the hashes of the 4 micro-batches and of the CPU / CUDA RNG states.

Runs: native x3 with the untouched entry path, THEN the candidate is installed and run x2,
every run from the same snapshot (model, optimizer state, CPU+CUDA RNG, the same 4
micro-batches).

Compared per pair, in addition to what round 57 already compared:
  the ACCUMULATED DENSE GRADIENTS AFTER EVERY MICRO-BATCH (4 per run),
  the gradient norm returned by clip_grad_norm_ (pre-clip total norm),
  post-update parameters, optimizer state, non-finite counts and the KEY SETS of every
  compared dictionary.
Accumulated gradients are cloned on the device after each backward and copied to the host
only after the optimizer step, so no host synchronization is inserted between backward and
clip (the run pattern inside the step stays the one of the entry loop).

Known device non-determinism: the word / position embedding tables get their gradient from
an index-accumulating backward whose summation order is not reproducible, so two NATIVE
runs differ on exactly these two tensors. They are reported separately and excluded from
the candidate verdict.

The candidate patch is installed only for the candidate runs. For dw_route the whitelist
matches and the actual flag_gems.mm calls are counted; a run that requests the route but
never reaches flag_gems.mm fails.

A candidate run in ANOTHER process, or with the route registered before the warmup, is not
a valid comparison (see results/route_from_step0_r57*.json: numerics on this device depend
on the run pattern).

--candidate dw_route              the 12 dW shapes routed to flag_gems.mm (aten::mm override)
--candidate prefetch_faithful     the same loss_for_batch; only prepare_embedding_inputs
                                  + move_features run on a worker thread
--candidate prefetch_loss_reimpl  round-56 implementation invoked under autocast
--candidate prefetch_loss_reimpl_run1
                                  round-56 implementation invoked OUTSIDE autocast as round-56 did
--candidate h2d_nonblocking       move_features through pinned buffers + non_blocking=True
"""
import argparse, copy, hashlib, json, math, os, sys, time, concurrent.futures
from collections.abc import Mapping
import torch
ap=argparse.ArgumentParser()
ap.add_argument("--conf")
ap.add_argument("--candidate",choices=["dw_route","prefetch_faithful","prefetch_loss_reimpl","prefetch_loss_reimpl_run1","h2d_nonblocking"])
ap.add_argument("--out",required=True,help="run mode: writes <out stem>.<candidate>.json; --merge: reads them all and writes <out>")
ap.add_argument("--merge",action="store_true")
ap.add_argument("--warmup-steps",type=int,default=2); ap.add_argument("--native-runs",type=int,default=3); ap.add_argument("--candidate-runs",type=int,default=2)
a=ap.parse_args()
if not a.merge and not (a.conf and a.candidate): ap.error("--conf and --candidate are required unless --merge")
STATIC={"field_version":"train_state_v3","time_unit":"ms",
 "design":"ONE process per candidate: 2 warmup steps that are REAL optimizer updates of the entry loop (loss under autocast via the untouched flagscale.train.losses.retrieval.loss_for_batch, isfinite(loss) host sync, backward, clip_grad_norm_, optimizer.step), then ONE snapshot (model, optimizer state with AdamW momentum, CPU+CUDA RNG, the 4 micro-batches); native x3 from the snapshot, then the candidate is installed and run x2 from the same snapshot.",
 "rng_dropout_data_order":"CPU and CUDA RNG states are restored from the snapshot before every run (dropout masks identical); the 4 micro-batches are deep-copied in the snapshot (data order fixed); tokenization has no randomness; the prefetch worker thread consumes no torch RNG.",
 "thresholds":{"grad_rel_tol":0.02,"param_abs_tol":1e-3,"state_rel_tol":0.02,"nonfinite_max":0,"basis":"2% relative error floored at 1% of the tensor's max |value| (bf16 mantissa); params by absolute error; the verdict distinguishes bitwise identity from within-threshold"},
 "embedding_tables":"word / position embedding gradients come from an index-accumulating backward whose summation order is not reproducible on this device: in every native-vs-native and candidate-vs-native pair these 2 tables differ (gradients and optimizer state) while all dense gradient tensors, the dense parameters and the dense optimizer state are bitwise identical; token_type_embeddings is identical too. Reported separately, excluded from the candidate verdict.",
 "cross_process_warning":"SCOPE (round-57): the bitwise comparisons of this file are between runs that repeat an identical run pattern inside one process (same ops, same synchronization points, same restored snapshot). With a different run pattern the native warmup was NOT reproducible, not even across fresh processes (route_from_step0_r57.*.native_from_start_*.json) - see summary_r57.json reproducibility_scope; the cause is not determined."}
if a.merge:
    import glob
    cands={}
    for f in sorted(glob.glob(a.out[:-5]+".*.json")):
        c=json.load(open(f)); cands[c["candidate"]]=c
    db=dict(STATIC); db["native_band_reference"]=cands["dw_route"]["native_band"]; db["dense_grad_tensors"]=cands["dw_route"]["dense_grad_tensors"]
    db["dense_param_tensors"]=cands["dw_route"]["dense_param_tensors"]; db["candidates"]=cands
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
def micro_step(loss_fn,b,autocast=True):
    if autocast:
        with autocast_context(dev,"bf16"):
            raw=loss_fn(b); loss=raw/accum
    else:
        raw=loss_fn(b); loss=raw/accum
    if not torch.isfinite(raw): raise FloatingPointError("Non-finite training loss")
    loss.backward(); return float(raw.detach())
def train_step(loss_fn,bs,autocast=True):
    optimizer.zero_grad(set_to_none=True)
    return [micro_step(loss_fn,b,autocast) for b in bs]
OUTER_AUTOCAST=True; _LIB=None; _EX=None; _FUT={}; _HOLD=[]
CNT={"whitelist_match":0,"flag_gems_mm":0,"native_fallback":0}
if a.candidate=="dw_route":
    sys.path.insert(0,"/tmp/opencode/FlagGems-base/src"); import flag_gems
    FG=flag_gems.mm; MM_OUT=torch.ops.aten.mm.out
    N_MK=[(4096,5120,1024),(1024,5120,4096),(4096,4430,1024),(1024,4430,4096),(4096,4310,1024),(1024,4310,4096),
          (4096,3740,1024),(1024,3740,4096),(4096,3290,1024),(1024,3290,4096),(4096,2990,1024),(1024,2990,4096)]
    DW={((N,M),(M,K)) for (N,M,K) in N_MK}
    def route(self,other):
        if (tuple(self.shape),tuple(other.shape)) in DW:
            CNT["whitelist_match"]+=1; CNT["flag_gems_mm"]+=1; return FG(self,other)
        CNT["native_fallback"]+=1
        return MM_OUT(self,other,out=torch.empty(self.shape[0],other.shape[1],device=self.device,dtype=self.dtype))
    cand_loss=inline_loss
    def install():
        global _LIB
        _LIB=torch.library.Library("aten","IMPL"); _LIB.impl("mm",route,"CUDA")
elif a.candidate=="h2d_nonblocking":
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
def to_cpu(t): return t.detach().to("cpu")
def param_hash(params,skip):
    h=hashlib.sha256()
    for n in sorted(params):
        if skip and n in skip: continue
        t=params[n].detach().contiguous(); h.update(n.encode()); h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def opt_info():
    ents=optimizer.state_dict()["state"]
    if not ents: return {"state_entries":0,"steps":[],"state_keys":[]}
    return {"state_entries":len(ents),"steps":sorted({float(v["step"]) for v in ents.values()}),"state_keys":sorted({k for v in ents.values() for k in v}),
            "exp_avg_l2":math.sqrt(sum(float(v["exp_avg"].detach().float().pow(2).sum()) for v in ents.values())),
            "exp_avg_sq_sum":sum(float(v["exp_avg_sq"].detach().float().sum()) for v in ents.values()),
            "all_finite":bool(all(bool(torch.isfinite(v[k]).all()) for v in ents.values() for k in ("exp_avg","exp_avg_sq") if k in v))}
def batch_hash(b):
    h=hashlib.sha256()
    for k in sorted(b): h.update(("|".join(b[k])).encode())
    return h.hexdigest()[:16]
def run(snap,loss_fn,autocast=True):
    model.load_state_dict(copy.deepcopy(snap["model"])); optimizer.load_state_dict(copy.deepcopy(snap["opt"]))
    torch.set_rng_state(snap["rng_cpu"]); torch.cuda.set_rng_state(snap["rng_cuda"])
    optimizer.zero_grad(set_to_none=True); losses=[]; g_gpu=[]; e_gpu=[]
    for b in snap["batches"]:
        losses.append(micro_step(loss_fn,b,autocast))
        g_gpu.append({n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None and n not in EMB})
        e_gpu.append({n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None and n in EMB})
    gn=float(torch.nn.utils.clip_grad_norm_(trainable,GN)); optimizer.step()
    grads=[{n:to_cpu(t) for n,t in g.items()} for g in g_gpu]
    egrads=[{n:to_cpu(t) for n,t in g.items()} for g in e_gpu]
    params={n:to_cpu(p) for n,p in model.named_parameters()}
    state={n+"."+k:to_cpu(v) for n,p in model.named_parameters() if p in optimizer.state for k,v in optimizer.state[p].items() if torch.is_tensor(v)}
    nontensor={n+"."+k:str(v) for n,p in model.named_parameters() if p in optimizer.state for k,v in optimizer.state[p].items() if not torch.is_tensor(v)}
    if a.candidate=="h2d_nonblocking": _HOLD.clear()
    return {"losses":losses,"grad_norm":gn,"grads":grads,"egrad":egrads,"params":params,"state":state,"nontensor":nontensor,
            "keys":{"grads":sorted(grads[-1]),"params":sorted(params),"state":sorted(state),"state_fields":sorted({k.rsplit(".",1)[-1] for k in state})},
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
    return {"tensors":n,"bitwise_identical":n-len(diff),"n_differing":len(diff),"max_abs":round(ma,8),"max_rel":round(mr,6),
            "over_2pct":over,"nonfinite":nf,"nontensor_mismatch":ntm,"differing":sorted(diff)[:6]}
def pick(D,names): return {k:v for k,v in D.items() if k in names}
def split_state(D):
    return ({k:v for k,v in D.items() if k.rsplit(".",1)[0] not in EMB},{k:v for k,v in D.items() if k.rsplit(".",1)[0] in EMB})
def nonfinite_all(R):
    nf=0
    for D in R["grads"][-1:]+R["egrad"][-1:]+[R["params"],R["state"]]:
        for t in D.values(): nf+=int((~torch.isfinite(t.float())).sum())
    return nf
def pair(A,B,la,lb,kind):
    pm=[agg(A["grads"][i],B["grads"][i]) for i in range(accum)]
    pe=[agg(A["egrad"][i],B["egrad"][i]) for i in range(accum)]
    nt=sum(agg(A["nontensor"],B["nontensor"])["nontensor_mismatch"] for _ in [0])
    return {"kind":kind,"a":la,"b":lb,"loss_equal":A["losses"]==B["losses"],"loss_max_abs":max(abs(x-y) for x,y in zip(A["losses"],B["losses"])),
            "grad_norm":[A["grad_norm"],B["grad_norm"]],"grad_norm_equal":A["grad_norm"]==B["grad_norm"],
            "grads_dense_per_micro":pm,"grads_embedding_tables_per_micro":pe,
            "params_dense":agg(A["params"],B["params"],EMB),"params_embedding_tables":agg(pick(A["params"],EMB),pick(B["params"],EMB)),
            "optimizer_state_dense":agg(*[split_state(R["state"])[0] for R in (A,B)]),"optimizer_state_embedding_tables":agg(*[split_state(R["state"])[1] for R in (A,B)]),
            "optimizer_nontensor_mismatch":agg(A["nontensor"],B["nontensor"])["nontensor_mismatch"],
            "keys_equal":{k:A["keys"][k]==B["keys"][k] for k in A["keys"]},"nonfinite_total":nonfinite_all(A)+nonfinite_all(B),
            "param_sha256_dense_equal":A["param_sha256_dense"]==B["param_sha256_dense"],"param_sha256_all_equal":A["param_sha256_all"]==B["param_sha256_all"]}
# ---- warmup: REAL optimizer updates of the entry loop ----
for _ in range(a.warmup_steps):
    train_step(inline_loss,[next(stream) for _ in range(accum)])
    torch.nn.utils.clip_grad_norm_(trainable,GN); optimizer.step()
batches=[next(stream) for _ in range(accum)]
snap={"model":copy.deepcopy(model.state_dict()),"opt":copy.deepcopy(optimizer.state_dict()),
      "rng_cpu":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state(),"batches":copy.deepcopy(batches)}
SNAP={"param_sha256_dense":param_hash({n:to_cpu(p) for n,p in model.named_parameters()},EMB),
      "param_sha256_all":param_hash({n:to_cpu(p) for n,p in model.named_parameters()},None),
      "optimizer":opt_info(),"batch_hashes":[batch_hash(b) for b in batches],
      "rng_cpu_sha256":hashlib.sha256(snap["rng_cpu"].numpy().tobytes()).hexdigest(),"rng_cuda_sha256":hashlib.sha256(snap["rng_cuda"].numpy().tobytes()).hexdigest()}
NAT=[run(snap,inline_loss) for _ in range(a.native_runs)]
for k in CNT: CNT[k]=0
install()
CAND=[run(snap,cand_loss,OUTER_AUTOCAST) for _ in range(a.candidate_runs)]
if a.candidate=="dw_route" and CNT["flag_gems_mm"]==0:
    raise SystemExit("dw_route requested but flag_gems.mm was never called (whitelist matches %d)"%CNT["whitelist_match"])
pairs=[]
for i in range(len(NAT)):
    for j in range(i+1,len(NAT)): pairs.append(pair(NAT[i],NAT[j],f"native{i}",f"native{j}","native_vs_native"))
for c in range(len(CAND)):
    for n in range(len(NAT)): pairs.append(pair(CAND[c],NAT[n],f"{a.candidate}{c}",f"native{n}","candidate_vs_native"))
for i in range(len(CAND)):
    for j in range(i+1,len(CAND)): pairs.append(pair(CAND[i],CAND[j],f"{a.candidate}{i}",f"{a.candidate}{j}","candidate_vs_candidate"))
cv=[p for p in pairs if p["kind"]!="native_vs_native"]; nv=[p for p in pairs if p["kind"]=="native_vs_native"]
def dense_bitwise(p):
    return all(g["bitwise_identical"]==g["tensors"] for g in p["grads_dense_per_micro"]) and p["params_dense"]["bitwise_identical"]==p["params_dense"]["tensors"] and p["optimizer_state_dense"]["bitwise_identical"]==p["optimizer_state_dense"]["tensors"]
def keys_ok(p): return all(p["keys_equal"].values()) and p["optimizer_nontensor_mismatch"]==0
def strict(p): return dense_bitwise(p) and keys_ok(p) and p["loss_equal"] and p["grad_norm_equal"] and p["nonfinite_total"]==0
def within(p):
    return (all(g["max_rel"]<=0.02 and g["nonfinite"]==0 for g in p["grads_dense_per_micro"]) and p["params_dense"]["max_abs"]<=1e-3 and p["params_dense"]["nonfinite"]==0
            and p["optimizer_state_dense"]["max_rel"]<=0.02 and p["optimizer_state_dense"]["nonfinite"]==0 and p["nonfinite_total"]==0 and keys_ok(p))
bit=all(strict(p) for p in cv); win=all(within(p) for p in cv)
gd=max(max(g["n_differing"] for g in p["grads_dense_per_micro"]) for p in cv); pd=max(p["params_dense"]["n_differing"] for p in cv)
nd=nv[0]["grads_dense_per_micro"][-1]["tensors"]; npd=nv[0]["params_dense"]["tensors"]
res={"candidate":a.candidate,"warmup_steps_real_updates":a.warmup_steps,"native_runs":len(NAT),"candidate_runs":len(CAND),"pairs":pairs,
     "snapshot":SNAP,"dense_grad_tensors":nd,"dense_param_tensors":npd,"embedding_tables_excluded":sorted(EMB),
     "native_band":{"pairs":len(nv),"dense_bitwise_identical_in_all_pairs":all(dense_bitwise(p) for p in nv),"keys_equal_all_pairs":all(keys_ok(p) for p in nv),
                    "losses_and_grad_norm_equal_all_pairs":all(p["loss_equal"] and p["grad_norm_equal"] for p in nv),
                    "embedding_table_grad_tensors_differing_per_pair":[p["grads_embedding_tables_per_micro"][-1]["n_differing"] for p in nv],
                    "embedding_table_grad_max_abs":max(p["grads_embedding_tables_per_micro"][-1]["max_abs"] for p in nv)},
     "candidate_pairs":len(cv),"dense_grad_tensors_differing_max":gd,"dense_param_tensors_differing_max":pd,
     "dense_grad_max_abs":max(g["max_abs"] for p in cv for g in p["grads_dense_per_micro"]),"dense_grad_max_rel":max(g["max_rel"] for p in cv for g in p["grads_dense_per_micro"]),
     "dense_param_max_abs":max(p["params_dense"]["max_abs"] for p in cv),"dense_optimizer_state_max_abs":max(p["optimizer_state_dense"]["max_abs"] for p in cv),
     "loss_equal_all_pairs":all(p["loss_equal"] for p in cv),"grad_norm_equal_all_pairs":all(p["grad_norm_equal"] for p in cv),"keys_equal_all_pairs":all(keys_ok(p) for p in cv),
     "nonfinite_total":sum(p["nonfinite_total"] for p in cv),"param_sha256_dense_equal_all_pairs":all(p["param_sha256_dense_equal"] for p in cv),
     "call_counts_candidate_runs":dict(CNT) if a.candidate=="dw_route" else None,
     "param_sha256_dense":[r["param_sha256_dense"] for r in NAT+CAND],"grad_norm":[r["grad_norm"] for r in NAT+CAND],"losses_micro":[r["losses"] for r in NAT+CAND]}
if bit: res["verdict"]="PASS (bitwise): in all %d candidate pairs the accumulated dense gradients after every micro-batch (%d tensors), the pre-clip gradient norm, the post-update dense parameters (%d tensors) and the dense optimizer state are identical to the same-snapshot native runs; losses equal; key sets equal; 0 non-finite. Only the word / position embedding gradients (and the embedding tables' optimizer state) differ, and native-vs-native differs on the same tensors (device accumulation-order non-determinism)."%(len(cv),nd,npd)
elif win: res["verdict"]="PASS (within declared thresholds, NOT bitwise): dense grad max_rel %g, param max_abs %g over %d candidate pairs."%(res["dense_grad_max_rel"],res["dense_param_max_abs"],len(cv))
else: res["verdict"]="FAIL: candidate is not within the declared thresholds in all candidate pairs (dense grad tensors differing up to %d per comparison, grad max_rel %g, param max_abs %g)."%(gd,res["dense_grad_max_rel"],res["dense_param_max_abs"])
json.dump(dict(STATIC,**res),open(a.out[:-5]+"."+a.candidate+".json","w"),indent=1)
print(a.candidate,"->",res["verdict"][:200])
print("  snapshot:",json.dumps(SNAP)[:600])
print("  native band:",res["native_band"])
print("  candidate: dense grad differing<=%d, params differing<=%d, state max_abs %g | losses %s | gn %s | keys %s | nonfinite %d | calls %s"%(gd,pd,res["dense_optimizer_state_max_abs"],res["loss_equal_all_pairs"],res["grad_norm_equal_all_pairs"],res["keys_equal_all_pairs"],res["nonfinite_total"],res["call_counts_candidate_runs"]))
