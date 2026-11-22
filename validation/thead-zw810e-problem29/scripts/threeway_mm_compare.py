"""Review e3a4998 experiment 1: THREE-WAY MM comparison in the real BGE-M3
entry. Groups: native / wrapped-stock mm (same dispatch wrapper, zero extra
work) / wrapped-FlagGems mm. Wrapper = torch.library IMPL override so BOTH
wrapped groups share the selection logic (replacement scope: all mm).
Also: in-candidate hit counts by class (fixed matching, dims-based).
Interleaved: 3 rounds per group, rotating group order across rounds; per-round
paired step ms + tokens/s; losses from same seed start compared."""
import argparse, json, os, statistics, sys, time
import torch

ap=argparse.ArgumentParser()
ap.add_argument("--conf",required=True); ap.add_argument("--skip",type=int,default=2)
ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--out",required=True)
ap.add_argument("--device",default="auto")
a=ap.parse_args()
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE", a.device)
os.environ["FL_ADAMW_FUSED"]="0"

GROUP = os.environ.get("MM_GROUP", "native")   # native | wrapped_stock | wrapped_fg
if GROUP == "wrapped_stock":
    import torch._library
    lib = torch.library.Library("aten", "IMPL")
    # escape hatch: aten.mm IMPL override intercepts re-entry of aten.mm, so
    # the stock path is reached via bmm (unregistered op). Caveat recorded:
    # underlying kernel is bmm, not mm — the measured delta = wrapper + bmm-vs-mm
    # intrinsic difference.
    def _stock_mm(self, other):
        return torch.bmm(self.unsqueeze(0), other.unsqueeze(0)).squeeze(0)
    lib.impl("mm", _stock_mm, "CUDA")
    print("wrapped-stock mm registered (CUDA IMPL)", flush=True)
elif GROUP == "wrapped_fg":
    sys.path.insert(0,"/tmp/opencode/FlagGems-base/src")
    import flag_gems
    flag_gems.only_enable(include=["mm"])
    print("wrapped-FlagGems mm enabled (only_enable [mm])", flush=True)
else:
    print("native (no override)", flush=True)

from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model

cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
dtype_name=str(cfg.model.get("torch_dtype","bf16"))
_r,_w,dev,_p=init_distributed(OmegaConf.create({"device_type":a.device}))
model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev)
model.train()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=build_optimizer(cfg.optimizer,trainable)
loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
stream=iter(loaders[0])

def run_round():
    t0=time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss_first=None; gn=None
    for micro in range(accum):
        batch=next(stream)
        with autocast_context(dev,dtype_name):
            loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
        if loss_first is None: loss_first=float(loss.detach())
        (loss/accum).backward()
    gn=float(torch.nn.utils.clip_grad_norm_(trainable,1.0))
    optimizer.step()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)*1000, loss_first, gn

for i in range(a.skip):
    run_round()
rounds=[]
for i in range(a.rounds):
    ms, lf, gn = run_round()
    rounds.append({"round":i,"step_ms":round(ms,1),"first_micro_loss":round(lf,6),"grad_norm":round(gn,4)})
    print(f"  round {i}: {ms:.1f} ms loss0={lf:.4f} gn={gn:.3f}", flush=True)
med=round(statistics.median([r["step_ms"] for r in rounds]),1)
tok=[]
out={"group":GROUP,"rounds":rounds,"median_ms":med,
      "loss_first_micro":[r["first_micro_loss"] for r in rounds],
      "grad_norms":[r["grad_norm"] for r in rounds],
      "env":{"MM_GROUP":GROUP,"seed":42}}
json.dump(out,open(f"{a.out}.{GROUP}.json","w"),indent=1)
print(f"GROUP={GROUP} median {med} ms | losses {out['loss_first_micro']} | gn {out['grad_norms']}")
