"""Review 60e2311 experiments: four-group whole-step comparison in the real
BGE-M3 entry — native / wrapped-stock(mm.out escape = TRUE stock kernel) /
wrapped-fg-all / dW-only-routed (fg only on measured FFN dW shapes) — plus
in-candidate hit/timing accounting, cold-vs-warm compile accounting, and
loss/grad/param accuracy checks from identical init."""
import argparse, hashlib, json, os, statistics, sys, time
import torch

ap=argparse.ArgumentParser(); ap.add_argument("--conf",required=True)
ap.add_argument("--skip",type=int,default=2); ap.add_argument("--rounds",type=int,default=3)
ap.add_argument("--out",required=True); ap.add_argument("--device",default="auto")
a=ap.parse_args()
import random
random.seed(42); torch.manual_seed(42)
os.environ.setdefault("FLAGSCALE_DEVICE", a.device)
os.environ["FL_ADAMW_FUSED"]="0"
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model

MM = torch.ops.aten.mm.default
MM_OUT = torch.ops.aten.mm.out
FG = None
STATE = {"mode":"none","dW_shapes":set(),"stats":{},"first_call":{},"pending":[]}
def flush_stats():
    for grp, key, s, e, is_cold in STATE["pending"]:
        st = STATE["stats"].setdefault((grp, key), {"count":0,"ms":0.0,"cold_ms":None})
        ms = s.elapsed_time(e)
        st["count"] += 1; st["ms"] += ms
        if is_cold and st["cold_ms"] is None:
            st["cold_ms"] = round(ms, 3)
    STATE["pending"].clear()

def fg_mm(x,y):
    if FG is None: raise RuntimeError("fg not loaded")
    return FG(x,y)

def route(self, other):
    key = (tuple(self.shape), tuple(other.shape))
    mode = STATE["mode"]
    if mode == "wrapped_stock_mmout":
        out = torch.empty(self.shape[0], other.shape[1], device=self.device, dtype=self.dtype)
        return MM_OUT(self, other, out=out)
    if mode == "fg_all":
        # events recorded, elapsed computed at flush (after synchronize) — avoids
        # 'events must be completed' inside autograd engine (review-safe timing)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        r = fg_mm(self, other)
        e.record()
        STATE["pending"].append(("fg_all", key, s, e, key not in STATE["first_call"]))
        STATE["first_call"][key] = True
        return r
    if mode == "dw_only_routed":
        if key in STATE["dW_shapes"]:
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); r = fg_mm(self, other); e.record()
            STATE["pending"].append(("dw_fg", key, s, e, False))
            return r
        out = torch.empty(self.shape[0], other.shape[1], device=self.device, dtype=self.dtype)
        return MM_OUT(self, other, out=out)
    raise RuntimeError("route called in mode "+mode)

def flush_stats():
    for grp, key, s, e, is_cold in STATE["pending"]:
        st = STATE["stats"].setdefault((grp, key), {"count":0,"ms":0.0,"cold_ms":None})
        ms = s.elapsed_time(e)
        st["count"] += 1; st["ms"] += ms
        if is_cold and st["cold_ms"] is None:
            st["cold_ms"] = round(ms, 3)
    STATE["pending"].clear()

GROUPS = ["native", "wrapped_stock_mmout", "fg_all", "dw_only_routed"]
results = {}
ref_accuracy = None
param_snap = {}

for gi, GROUP in enumerate(GROUPS):
    import random as _r
    _r.seed(42); torch.manual_seed(42)
    if GROUP != "native":
        if FG is None:
            sys.path.insert(0,"/tmp/opencode/FlagGems-base/src")
            import flag_gems as _fgmod
            FG = _fgmod.mm
        if gi == 0 or not STATE["dW_shapes"]:
            # measured dW shapes (r36): ffn dW S512 + mid-seq top
            STATE["dW_shapes"] = {
                (4096,5120,1024),(1024,5120,4096),
                (4096,4430,1024),(1024,4430,4096),(4096,4310,1024),(1024,4310,4096),
                (4096,3740,1024),(1024,3740,4096),(4096,3290,1024),(1024,3290,4096),
                (4096,2990,1024),(1024,2990,4096)}
        if GROUP == "wrapped_stock_mmout" and not STATE.get("stock_reg"):
            STATE["lib"] = torch.library.Library("aten","IMPL")
            STATE["lib"].impl("mm", route, "CUDA")
            STATE["stock_reg"] = True
        elif GROUP == "fg_all" and not STATE.get("fgall_reg"):
            STATE["liball"] = torch.library.Library("aten","IMPL")
            STATE["liball"].impl("mm", route, "CUDA")
            STATE["fgall_reg"] = True
        elif GROUP == "dw_only_routed" and not STATE.get("dw_reg"):
            STATE["libdw"] = torch.library.Library("aten","IMPL")
            STATE["libdw"].impl("mm", route, "CUDA")
            STATE["dw_reg"] = True
    STATE["mode"] = GROUP
    STATE["stats"] = {}
    STATE["pending"] = []
    STATE["first_call"] = {}
    STATE["flush"] = flush_stats
    cfg=OmegaConf.load(a.conf); accum=int(cfg.gradient_accumulation_steps)
    dtype_name=str(cfg.model.get("torch_dtype","bf16"))
    _r2,_w2,dev,_p=init_distributed(OmegaConf.create({"device_type":a.device}))
    model,processor=build_retrieval_model(str(cfg.model.model_type),cfg.model,dev)
    model.train()
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=build_optimizer(cfg.optimizer,trainable)
    loaders=build_split_loaders(str(cfg.task),cfg.data,int(cfg.batch_size),pin_memory=True,seed=42)
    stream=iter(loaders[0])
    rounds_data=[]; micro_losses=[]; gns=[]
    for i in range(a.skip):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch=next(stream)
            with autocast_context(dev,dtype_name):
                loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
            (loss/accum).backward()
        gns.append(float(torch.nn.utils.clip_grad_norm_(trainable,1.0)))
        optimizer.step()
    for i in range(a.rounds):
        t0=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        ml=[]
        for micro in range(accum):
            batch=next(stream)
            with autocast_context(dev,dtype_name):
                loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
            ml.append(round(float(loss.detach()),6))
            (loss/accum).backward()
        gn=float(torch.nn.utils.clip_grad_norm_(trainable,1.0))
        optimizer.step()
        torch.cuda.synchronize()
        if "flush" in STATE: STATE["flush"]()
        ms=(time.perf_counter()-t0)*1000
        rounds_data.append({"round":i,"step_ms":round(ms,1),"micro_losses":ml,"grad_norm":round(gn,4)})
        micro_losses.extend(ml)
        gns.append(gn)
    # param snapshot (updated params) + grad snapshot from LAST warmup grad (pre-step grads overwritten; use final param hash)
    ph = hashlib.sha256()
    for p in trainable:
        ph.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() if p.dtype!=torch.bfloat16 else p.detach().cpu().view(torch.int16).numpy().tobytes())
    param_hash = ph.hexdigest()[:16]
    pmax = max(float(p.detach().abs().max()) for p in trainable)
    res={"group":GROUP,"median_ms":round(statistics.median([r["step_ms"] for r in rounds_data]),1),
          "step_ms":[r["step_ms"] for r in rounds_data],
          "micro_losses_last_round":rounds_data[-1]["micro_losses"],
          "grad_norms_warmup":gns[:2],"grad_norms_rounds":[r["grad_norm"] for r in rounds_data],
          "param_hash16":param_hash,"param_absmax":round(pmax,4),
          "wrapper_stats":{f"{k[0]}|{k[1]}":v for k,v in STATE["stats"].items()},
          "cold_first_calls":len(STATE["first_call"])}
    if GROUP == "native":
        ref_accuracy = {"micro_losses_last_round":rounds_data[-1]["micro_losses"],
                         "grad_norms_rounds":[r["grad_norm"] for r in rounds_data],
                         "param_hash16":param_hash}
    else:
        # accuracy vs native
        dl = [round(x-y,6) for x,y in zip(rounds_data[-1]["micro_losses"], ref_accuracy["micro_losses_last_round"])]
        gn_d = [round(x-y,3) for x,y in zip([r["grad_norm"] for r in rounds_data], ref_accuracy["grad_norms_rounds"])]
        nonfinite = int((~torch.isfinite(torch.cat([p.detach().float().view(-1) for p in trainable]))).sum())
        res["accuracy_vs_native"]={"last_round_micro_loss_diff":dl,
            "grad_norm_diff_rounds":gn_d,
            "param_hash_equal":param_hash==ref_accuracy["param_hash16"],
            "param_absmax":round(pmax,4),"nonfinite_params":nonfinite,
            "criteria":"bf16-level deviation expected for candidate kernels; nonfinite must be 0; param hash equality NOT required (different kernels)"}
    results[GROUP]=res
    print(f"[{GROUP}] median {res['median_ms']} ms | rounds {res['step_ms']} | param {param_hash} | cold_calls {res['cold_first_calls']}", flush=True)
    # teardown override libraries between groups (best effort)
    for lk in ("lib","liball","libdw"):
        if STATE.get(lk):
            try: STATE[lk]._destroy()
            except Exception: pass
            STATE[lk]=None
            STATE.pop("stock_reg",None); STATE.pop("fgall_reg",None); STATE.pop("dw_reg",None)
    time.sleep(2)
json_out={"device":torch.cuda.get_device_name(0),"groups":results,
           "dW_shape_set_size":len(STATE["dW_shapes"]),
           "notes":"wrapped-stock uses aten.mm.out escape = TRUE stock kernel; dw_only routes measured FFN dW shapes to FlagGems mm, all else stock mm.out; cold_first_calls = shapes triggering first-encounter compile in fg_all"}
json.dump(json_out,open(a.out,"w"),indent=1)
print("saved", a.out)
