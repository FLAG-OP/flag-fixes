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
CAPTURED = {}   # key -> (gy, x) first encounter

def route(self, other):
    key=(tuple(self.shape), tuple(other.shape))
    if STATE["mode"]=="wrapped_stock":
        out=torch.empty(self.shape[0], other.shape[1], device=self.device, dtype=self.dtype)
        return MM_OUT(self, other, out=out)
    if STATE["mode"]=="dw_only":
        # FFN dW signature: A=[N,M] with N in {1024,4096}, M>1000 (row idx),
        # B=[M,K] with K in {1024,4096}; A is the col-major gy.t() view.
        A, B = self.shape, other.shape
        is_ffn_dw = (len(A)==2 and len(B)==2 and A[1]>1000 and A[0] in (1024,4096)
                     and B[1] in (1024,4096) and self.stride(0)==1)
        if is_ffn_dw:
            if key not in CAPTURED and STATE.get("capture"):
                CAPTURED[key]=(self.detach().clone(), other.detach().clone())
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
STATE["capture"]=False
target_grads={}
for name, mod in model.named_modules():
    if ("intermediate.dense" in name or ("encoder.layer" in name and ".output.dense" in name)
        or "attention.output.dense" in name):
        if hasattr(mod,"weight") and mod.weight is not None and mod.weight.grad is not None:
            target_grads[name]=mod.weight.grad.detach().float().cpu().clone()

# frozen dW comparison: stock vs FlagGems on captured gy/x
frozen={}
for key,(gy,x) in list(CAPTURED.items())[:4]:
    y_stock=(gy.t()@x).float() if False else None
    # gy stored as self ([N,M] col-major view originally); reconstruct stock path
    gy_t=gy  # already the transposed operand as passed
    y_s=torch.ops.aten.mm.out(gy_t, x, out=torch.empty(gy_t.shape[0], x.shape[1], device=gy_t.device, dtype=gy_t.dtype)).float()
    y_f=FG(gy_t, x).float()
    ref=(gy_t.float()@x.float())
    floor=0.01*float(ref.abs().max())
    d_s=(y_s-ref).abs(); d_f=(y_f-ref).abs()
    frozen[f"{list(key[0])}x{list(key[1])}"]={
        "stock_vs_fp32_reldiff_floored": round(float((d_s/torch.maximum(ref.abs(),torch.full_like(ref,floor))).max()),4),
        "fg_vs_fp32_reldiff_floored": round(float((d_f/torch.maximum(ref.abs(),torch.full_like(ref,floor))).max()),4),
        "fg_vs_stock_reldiff_floored": round(float(((y_f-y_s).abs()/torch.maximum(y_s.abs(),torch.full_like(y_s,floor))).max()),4),
        "stock_nonfinite": int((~torch.isfinite(y_s)).sum()), "fg_nonfinite": int((~torch.isfinite(y_f)).sum())}
json.dump({"group":GROUP,"init_param_hash":init_hash,"rng_cpu":rng_cpu,"rng_cuda":rng_cuda,
           "micro_input_hashes":micro_hashes,"target_grads":target_grads,"frozen_dW":frozen,
           "captured_dW_keys":len(CAPTURED)},
          open(f"{a.out}.{GROUP}.pkl","wb")) if False else None
import pickle
pickle.dump({"group":GROUP,"init_param_hash":init_hash,"rng_cpu":rng_cpu,"rng_cuda":rng_cuda,
             "micro_input_hashes":micro_hashes,"target_grads":target_grads,"frozen_dW":frozen,
             "captured_dW_keys":len(CAPTURED)}, open(f"{a.out}.{GROUP}.pkl","wb"))
print(f"[{GROUP}] init {init_hash} rng_cpu {rng_cpu} rng_cuda {rng_cuda} micros {micro_hashes} captured_dW {len(CAPTURED)} frozen {json.dumps(frozen)[:300]}", flush=True)
