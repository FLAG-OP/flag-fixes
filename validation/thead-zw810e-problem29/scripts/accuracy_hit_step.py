"""Review bqPbopQ item 1: accuracy acceptance on a step where the dW candidate
ACTUALLY HITS. Run to just before step 2 (a known-hit step), snapshot model /
optimizer / pending batches / CPU+CUDA RNG; then restore that snapshot and run
the SAME step under native, wrapped-stock, dw-only, and native-again
(nondeterminism baseline). Record in ONE record: per-shape hits IN THIS STEP,
pre-update target grads, post-update params. Zero target hits => FAIL (exit 4).
Criteria: existing 2% floored grad gate, reported per group (FFN up / FFN down
/ attn output proj) with nonfinite counts and PASS/FAIL."""
import argparse, copy, hashlib, json, os, pickle, statistics, sys
import torch

ap=argparse.ArgumentParser(); ap.add_argument("--conf",required=True)
ap.add_argument("--hit-step",type=int,default=2,help="step index (0-based) known to hit the dW whitelist")
ap.add_argument("--out",required=True); ap.add_argument("--device",default="auto")
ap.add_argument("--phase",default="", help="A=native-only x2 (no override) | B=wrapped_stock+dw_only (override) | empty=orchestrate subprocesses")
a=ap.parse_args()
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE", a.device); os.environ["FL_ADAMW_FUSED"]="0"
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
STATE={"mode":"native","hits":{}}
def route(self,other):
    key=(tuple(self.shape),tuple(other.shape))
    if STATE["mode"]=="wrapped_stock":
        return MM_OUT(self,other,out=torch.empty(self.shape[0],other.shape[1],device=self.device,dtype=self.dtype))
    if STATE["mode"]=="dw_only":
        if key in DW:
            STATE["hits"][key]=STATE["hits"].get(key,0)+1   # counted ONLY while in the hit step
            return FG(self,other)
        return MM_OUT(self,other,out=torch.empty(self.shape[0],other.shape[1],device=self.device,dtype=self.dtype))
    raise RuntimeError("route in native mode")

cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
dtype_name=str(cfg.model.get("torch_dtype","bf16"))
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":a.device}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev); model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
stream=iter(loaders[0])
TARGET=lambda name: ("intermediate.dense" in name or ("encoder.layer" in name and ".output.dense" in name) or "attention.output.dense" in name)

def run_step(batches):
    optimizer.zero_grad(set_to_none=True)
    for batch in batches:
        with autocast_context(dev,dtype_name):
            loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
        (loss/accum).backward()
    gn=float(torch.nn.utils.clip_grad_norm_(trainable,1.0))
    pre_grads={n:m.weight.grad.detach().float().cpu().clone() for n,m in model.named_modules() if TARGET(n) and hasattr(m,"weight") and m.weight.grad is not None}
    optimizer.step()
    post={n:m.weight.detach().float().cpu().clone() for n,m in model.named_modules() if TARGET(n) and hasattr(m,"weight")}
    return pre_grads, post, gn

# ---- advance to just before the hit step (native, no override registered) ----
pending=[]
for i in range(a.hit_step):
    bl=[next(stream) for _ in range(accum)]
    run_step(bl)
hit_batches=[next(stream) for _ in range(accum)]
# ---- snapshot: model / optimizer / hit batches / RNG ----
def clone_opt(o):
    return copy.deepcopy(o.state_dict())
snap={"model":copy.deepcopy(model.state_dict()),"opt":clone_opt(optimizer),
      "batches":copy.deepcopy(hit_batches),
      "rng_cpu":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state(),
      "input_hashes":[hashlib.sha256("|".join("|".join(b[k]) for k in sorted(b.keys())).encode()).hexdigest()[:16] for b in hit_batches]}
h=hashlib.sha256()
for p in trainable: h.update(p.detach().cpu().view(torch.int16).numpy().tobytes() if p.dtype==torch.bfloat16 else p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
snap["param_hash"]=h.hexdigest()[:16]
rng_cpu_h=hashlib.sha256(snap["rng_cpu"].numpy().tobytes()).hexdigest()[:16]
rng_cuda_h=hashlib.sha256(snap["rng_cuda"].numpy().tobytes()).hexdigest()[:16]
SNAP_F=a.out+".snapshot.pkl"
PHASE=a.phase or ""
if PHASE in ("A",""):
    pickle.dump({"snap":snap,"rng_cpu_h":rng_cpu_h,"rng_cuda_h":rng_cuda_h},open(SNAP_F,"wb"))
if PHASE=="B":
    d=pickle.load(open(SNAP_F,"rb")); snap=d["snap"]; rng_cpu_h=d["rng_cpu_h"]; rng_cuda_h=d["rng_cuda_h"]

def restore():
    model.load_state_dict(snap["model"]); optimizer.load_state_dict(copy.deepcopy(snap["opt"]))
    torch.set_rng_state(snap["rng_cpu"]); torch.cuda.set_rng_state(snap["rng_cuda"])
    return copy.deepcopy(snap["batches"])

records={}
def do(group, mode):
    STATE["mode"]=mode; STATE["hits"]={}
    restore()
    pre,post,gn=run_step(copy.deepcopy(snap["batches"]))
    hits={f"{list(k[0])}x{list(k[1])}":v for k,v in STATE["hits"].items()}
    records[group]={"pre_grads":pre,"post_params":post,"grad_norm":gn,"hits_this_step":hits,
                    "hit_total":sum(hits.values()),"mode":mode}
    print(f"[{group}] hits={sum(hits.values())} gn={gn:.4f} target_layers={len(pre)}", flush=True)
    STATE["mode"]="native"

if PHASE == "A":
    do("native_run1","native"); do("native_run2","native")      # true native x2, no override
    pickle.dump(records,open(a.out+".A.pkl","wb"))
    print("PHASE_A_DONE"); sys.exit(0)
if PHASE == "B":
    lib=torch.library.Library("aten","IMPL"); lib.impl("mm", route, "CUDA")
    do("wrapped_stock","wrapped_stock"); do("dw_only","dw_only")
    pickle.dump(records,open(a.out+".B.pkl","wb"))
    print("PHASE_B_DONE"); sys.exit(0)
# orchestrator: subprocesses for true isolation
import subprocess
env=dict(os.environ); env["PYTHONPATH"]=env.get("PYTHONPATH","")+":/tmp/opencode/FlagGems-base/src"
for ph in ("A","B"):
    r=subprocess.run([sys.executable,__file__,"--conf",a.conf,"--hit-step",str(a.hit_step),"--out",a.out,"--phase",ph],env=env,capture_output=True,text=True,timeout=2000)
    print(f"phase {ph}: exit={r.returncode} | {r.stdout.strip()[-200:]}", flush=True)
    if r.returncode!=0: print("STDERR:", r.stderr.strip()[-600:]); sys.exit(1)
RA=pickle.load(open(a.out+".A.pkl","rb")); RB=pickle.load(open(a.out+".B.pkl","rb"))
records={"native_run1":RA["native_run1"],"native_run2":RA["native_run2"],"wrapped_stock":RB["wrapped_stock"],"dw_only":RB["dw_only"]}
d=pickle.load(open(SNAP_F,"rb")); snap=d["snap"]; rng_cpu_h=d["rng_cpu_h"]; rng_cuda_h=d["rng_cuda_h"]

# ---- compare ----
def floored_rel(x,y):
    floor=0.01*float(y.abs().max()); return float((x-y).abs().max()/max(floor,1e-12))
def group_of(n):
    if "intermediate.dense" in n: return "FFN_up"
    if "encoder.layer" in n and ".output.dense" in n and "attention" not in n: return "FFN_down"
    if "attention.output.dense" in n: return "attn_out_proj"
    return None
def compare(A,B):
    out={}
    for g in ("FFN_up","FFN_down","attn_out_proj"):
        rels=[floored_rel(A["pre_grads"][n],B["pre_grads"][n]) for n in B["pre_grads"] if n in A["pre_grads"] and group_of(n)==g]
        pdiff=[float((A["post_params"][n]-B["post_params"][n]).abs().max()) for n in B["post_params"] if n in A["post_params"] and group_of(n)==g]
        nf=sum(int((~torch.isfinite(A["pre_grads"][n])).sum()) for n in A["pre_grads"] if group_of(n)==g)
        out[g]={"grad_rel_max":round(max(rels),4),"grad_rel_median":round(statistics.median(rels),4),
                "post_param_absdiff_max":round(max(pdiff),6),"grad_nonfinite":nf,
                "PASS_2pct":max(rels)<=0.02}
    return out
TH={"grad_rel_tol":0.02,"basis":"existing 2% floored rel gate (floor=0.01*max|ref| per tensor), same as prior rounds"}
res={"hit_step":a.hit_step,"snapshot":{"param_hash":snap["param_hash"],"rng_cpu":rng_cpu_h,"rng_cuda":rng_cuda_h,"input_hashes":snap["input_hashes"]},
     "thresholds":TH,
     "hits_this_step":{"dw_only":records["dw_only"]["hits_this_step"],"total":records["dw_only"]["hit_total"],
                       "note":"hits counted ONLY inside the hit step; native/wrapped_stock routes record none"},
     "comparisons":{
       "A_native_run1_vs_native_run2_nondet":compare(records["native_run1"],records["native_run2"]),
       "B_wrapped_stock_vs_native":compare(records["native_run1"],records["wrapped_stock"]),
       "C_dw_only_vs_native":compare(records["native_run1"],records["dw_only"])},
     "grad_norms":{k:v["grad_norm"] for k,v in records.items()},
     "verdict":None}
if records["dw_only"]["hit_total"]==0:
    res["verdict"]="FAIL - zero target hits in the hit step"; json.dump(res,open(a.out,"w"),indent=1); sys.exit(4)
all_pass=all(v["PASS_2pct"] for g in res["comparisons"]["C_dw_only_vs_native"].values() for v in [g])
base_pass=all(g["PASS_2pct"] for g in res["comparisons"]["A_native_run1_vs_native_run2_nondet"].values())
res["verdict"]=f"dw-only {'PASS' if all_pass else 'FAIL'} at 2% grad gate on the hit step; native-vs-native baseline itself {'PASS' if base_pass else 'FAIL'} at the same gate"
json.dump({k:v for k,v in res.items()},open(a.out,"w"),indent=1)
pickle.dump(records,open(a.out+".records.pkl","wb"))
print(json.dumps(res["comparisons"],indent=1)); print("VERDICT:",res["verdict"])
