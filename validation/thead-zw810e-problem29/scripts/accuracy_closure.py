"""Review fe65d8f item 1: close the dW accuracy-FAIL loop.
Three same-state controls: native/native (nondeterminism baseline),
wrapped-stock/native, dW-only/native. Records: init-param hash, RNG states,
per-micro input hashes, per-element target-layer grads (first warmup step =
identical init/RNG/batch, only mm kernel differs).
Then captures REAL dW gy/x from the training backward, freezes them, and
compares stock vs FlagGems dW output directly (nonfinite + thresholds)."""
import argparse, hashlib, json, os, sys, time
import torch

ap=argparse.ArgumentParser(); ap.add_argument("--conf",required=True)
ap.add_argument("--skip",type=int,default=2); ap.add_argument("--out",required=True)
ap.add_argument("--device",default="auto")
a=ap.parse_args()
GROUP = os.environ.get("MM_GROUP","native")
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE", a.device)
os.environ["FL_ADAMW_FUSED"]="0"
sys.path.insert(0,"/tmp/opencode/FlagGems-base/src")
import flag_gems
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model

MM_OUT = torch.ops.aten.mm.out
FG = flag_gems.mm
STATE={"mode":GROUP,"dW_shapes":set(),"capture":None}
N_MK = [(4096,5120,1024),(1024,5120,4096),(4096,4430,1024),(1024,4430,4096),
        (4096,4310,1024),(1024,4310,4096),(4096,3740,1024),(1024,3740,4096),
        (4096,3290,1024),(1024,3290,4096),(4096,2990,1024),(1024,2990,4096)]
STATE["dW_shapes"] = {((N,M),(M,K)) for (N,M,K) in N_MK}
CAPTURED = {}   # key -> (gy, x, strides, dtype) first encounter
HITS = {}       # key -> count (per-shape hit accounting)

def route(self, other):
    key=(tuple(self.shape), tuple(other.shape))
    if STATE["mode"]=="wrapped_stock":
        out=torch.empty(self.shape[0], other.shape[1], device=self.device, dtype=self.dtype)
        return MM_OUT(self, other, out=out)
    if STATE["mode"]=="dw_only":
        # review df851e0 item 1: SAME exact 12-shape whitelist as the perf group
        # (wholestep_mm_groups.py) - no broad dimension rule, so attention dW
        # (N=1024 square) can NOT leak into the candidate here.
        if key in STATE["dW_shapes"]:
            HITS[key] = HITS.get(key, 0) + 1
            if key not in CAPTURED:
                CAPTURED[key]=(self.detach().clone(), other.detach().clone(),
                               tuple(self.stride()), tuple(other.stride()), str(self.dtype))
            return FG(self, other)
        out=torch.empty(self.shape[0], other.shape[1], device=self.device, dtype=self.dtype)
        return MM_OUT(self, other, out=out)
    raise RuntimeError("route in native mode")

if GROUP != "native":
    lib=torch.library.Library("aten","IMPL"); lib.impl("mm", route, "CUDA")

cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
dtype_name=str(cfg.model.get("torch_dtype","bf16"))
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":a.device}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev)
model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
stream=iter(loaders[0])

# init param hash + RNG states (post model/optimizer build, pre-training)
def param_hash():
    h=hashlib.sha256()
    for p in trainable:
        h.update(p.detach().cpu().view(torch.int16).numpy().tobytes() if p.dtype==torch.bfloat16 else p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]
init_hash = param_hash()
rng_cpu = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16]
rng_cuda = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()[:16]

# first warmup step: capture per-micro input hashes + target grads (same state)
micro_hashes=[]; STATE["capture"]=True
optimizer.zero_grad(set_to_none=True)
for micro in range(accum):
    batch=next(stream)
    mh=hashlib.sha256(("|".join("|".join(batch[k]) for k in sorted(batch.keys()))).encode()).hexdigest()[:16]
    micro_hashes.append(mh)
    with autocast_context(dev,dtype_name):
        loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
    (loss/accum).backward()
# capture stays ON for remaining steps (review df851e0 item 2:
# all 12 shapes must be covered; first warmup step alone hits none of them)
target_grads={}
for name, mod in model.named_modules():
    if ("intermediate.dense" in name or ("encoder.layer" in name and ".output.dense" in name)
        or "attention.output.dense" in name):
        if hasattr(mod,"weight") and mod.weight is not None and mod.weight.grad is not None:
            target_grads[name]=mod.weight.grad.detach().float().cpu().clone()

# remaining warmup + 3 timed steps so the 12-shape whitelist is fully encountered
for i in range(a.skip - 1 + 3):
    optimizer.zero_grad(set_to_none=True)
    for micro in range(accum):
        batch=next(stream)
        with autocast_context(dev,dtype_name):
            loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
        (loss/accum).backward()
    torch.nn.utils.clip_grad_norm_(trainable,1.0); optimizer.step(); torch.cuda.synchronize()

# frozen dW comparison on ALL captured shapes (review df851e0 item 2):
# bit-level comparison of raw output patterns, unequal-element count, UNROUNDED
# max abs err, nonfinite, dtype/stride/input-hash preserved.
def bits(t):
    return t.detach().cpu().contiguous().view(torch.int16) if t.dtype==torch.bfloat16 else t.detach().cpu().contiguous().view(torch.uint8)
frozen={}
for key,(gy,x,sg,sx,dt) in CAPTURED.items():
    gy_t=gy
    out_s=torch.empty(gy_t.shape[0], x.shape[1], device=gy_t.device, dtype=gy_t.dtype)
    y_s=torch.ops.aten.mm.out(gy_t, x, out=out_s)      # stock kernel (same as perf group's stock path)
    y_f=FG(gy_t, x)                                     # candidate
    b_s, b_f = bits(y_s), bits(y_f)
    bit_equal = bool(torch.equal(b_s.view(torch.int16) if b_s.dtype==torch.bfloat16 else b_s, b_f.view(torch.int16) if b_f.dtype==torch.bfloat16 else b_f))
    unequal = int((b_s.view(-1) != b_f.view(-1)).sum())
    max_abs = float((y_s.float()-y_f.float()).abs().max())
    frozen[f"{list(key[0])}x{list(key[1])}"]={
        "hits": HITS.get(key,0), "dtype": dt, "stride_gy": list(sg), "stride_x": list(sx),
        "input_sha256_16": hashlib.sha256((bits(gy_t).numpy().tobytes()+bits(x).numpy().tobytes())).hexdigest()[:16],
        "bitwise_identical": bit_equal, "unequal_elements": unequal, "numel": y_s.numel(),
        "max_abs_err_unrounded": max_abs,
        "stock_nonfinite": int((~torch.isfinite(y_s.float())).sum()),
        "fg_nonfinite": int((~torch.isfinite(y_f.float())).sum())}
json.dump({"group":GROUP,"init_param_hash":init_hash,"rng_cpu":rng_cpu,"rng_cuda":rng_cuda,
           "micro_input_hashes":micro_hashes,"target_grads":target_grads,"frozen_dW":frozen,
           "captured_dW_keys":len(CAPTURED)},
          open(f"{a.out}.{GROUP}.pkl","wb")) if False else None
import pickle
pickle.dump({"group":GROUP,"init_param_hash":init_hash,"rng_cpu":rng_cpu,"rng_cuda":rng_cuda,
             "micro_input_hashes":micro_hashes,"target_grads":target_grads,"frozen_dW":frozen,
             "captured_dW_keys":len(CAPTURED),"hits":{f"{list(k[0])}x{list(k[1])}":v for k,v in HITS.items()}}, open(f"{a.out}.{GROUP}.pkl","wb"))
print(f"[{GROUP}] init {init_hash} rng_cpu {rng_cpu} rng_cuda {rng_cuda} micros {micro_hashes} captured_dW {len(CAPTURED)} frozen {json.dumps(frozen)[:300]}", flush=True)
