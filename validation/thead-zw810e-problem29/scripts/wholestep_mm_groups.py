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
PERF_MODE = os.environ.get("WSPERF","") == "1"   # perf run: routing only, no events/counts
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
        if STATE.get("perf_mode"):
            return fg_mm(self, other)   # TRUE perf: no events, no dicts
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        r = fg_mm(self, other)
        e.record()
        STATE["pending"].append(("fg_all", key, s, e, key not in STATE["first_call"]))
        STATE["first_call"][key] = True
        return r
    if mode == "dw_native_contig":
        # review bqWBBIQ item 1: single-factor control. SAME whitelist, SAME Python
        # routing, SAME output allocation - but the dW product is computed by the
        # NATIVE mm on an explicitly contiguous() copy (no Triton kernel).
        if key in STATE["dW_shapes"]:
            if STATE.get("perf_mode"):
                STATE["dw_perf_count"] = STATE.get("dw_perf_count", 0) + 1
            # whitelisted dW: NATIVE kernel on an explicitly contiguous() copy
            out = torch.empty(self.shape[0], other.shape[1], device=self.device, dtype=self.dtype)
            return MM_OUT(self.contiguous(), other, out=out)
        # non-whitelist: IDENTICAL to dw_only_routed's fallback (mm.out + preallocated out),
        # so the two candidates differ ONLY in the whitelisted-dW kernel
        out = torch.empty(self.shape[0], other.shape[1], device=self.device, dtype=self.dtype)
        return MM_OUT(self, other, out=out)
    if mode == "dw_only_routed":
        if key in STATE["dW_shapes"]:
            if STATE.get("perf_mode"):
                STATE["dw_perf_count"] = STATE.get("dw_perf_count", 0) + 1  # single int, no dicts/events
                return fg_mm(self, other)
            ph = "warmup" if STATE.get("phase") == "warmup" else "timed"
            STATE["dw_hits"] = STATE.get("dw_hits", {})
            STATE["dw_hits"][(key, ph)] = STATE["dw_hits"].get((key, ph), 0) + 1
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

ACC_MODE = os.environ.get("WSACC","") == "1"
if ACC_MODE:
    GROUPS = ["native", "dw_only_routed"]
else:
    GROUPS = ["native", "wrapped_stock_mmout", "fg_all", "dw_only_routed", "dw_native_contig"]
# review df851e0 item 3: rotate group order between independent repeats
_ORDER = os.environ.get("GROUP_ORDER", "")
if _ORDER:
    GROUPS = _ORDER.split(",")
    print("GROUP_ORDER:", GROUPS, flush=True)
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
            # measured FFN dW shapes as (self.shape, other.shape) PAIRS —
            # dW: self = gy.t()[N,M], other = x[M,K]  (from r36 operand_dims_measured)
            N_MK = [(4096,5120,1024),(1024,5120,4096),
                    (4096,4430,1024),(1024,4430,4096),(4096,4310,1024),(1024,4310,4096),
                    (4096,3740,1024),(1024,3740,4096),(4096,3290,1024),(1024,3290,4096),
                    (4096,2990,1024),(1024,2990,4096)]
            STATE["dW_shapes"] = {((N,M),(M,K)) for (N,M,K) in N_MK}
        if GROUP == "wrapped_stock_mmout" and not STATE.get("stock_reg"):
            STATE["lib"] = torch.library.Library("aten","IMPL")
            STATE["lib"].impl("mm", route, "CUDA")
            STATE["stock_reg"] = True
        elif GROUP == "fg_all" and not STATE.get("fgall_reg"):
            STATE["liball"] = torch.library.Library("aten","IMPL")
            STATE["liball"].impl("mm", route, "CUDA")
            STATE["fgall_reg"] = True
        elif GROUP == "dw_native_contig" and not STATE.get("dwn_reg"):
            STATE["libn"] = torch.library.Library("aten","IMPL")
            STATE["libn"].impl("mm", route, "CUDA")
            STATE["dwn_reg"] = True
        elif GROUP == "dw_only_routed" and not STATE.get("dw_reg"):
            STATE["libdw"] = torch.library.Library("aten","IMPL")
            STATE["libdw"].impl("mm", route, "CUDA")
            STATE["dw_reg"] = True
    STATE["mode"] = GROUP
    STATE["perf_mode"] = PERF_MODE
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
    # tokens/s: valid tokens per step from r36 fingerprint window (steps 2-4)
    # review df851e0 item 3: count valid tokens from the ACTUAL batches consumed
    # (tokenizer attention_mask), never a stored constant; also hash each step's
    # inputs so groups can be checked for identical batches.
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(str(cfg.model.model_path))
    def step_tokens_and_hash(step_batches):
        valid = 0; h = hashlib.sha256()
        for batch in step_batches:
            for k in sorted(batch.keys()):
                h.update(("|".join(batch[k])).encode())
                enc = _tok(batch[k], padding=True, truncation=True,
                           max_length=int(cfg.model.get("max_length", 8192)), return_tensors=None)
                valid += sum(sum(1 for x in m if x == 1) for m in enc["attention_mask"])
        return valid, h.hexdigest()[:16]
    vt = [None]*a.rounds      # filled per round below
    rounds_data=[]; micro_losses=[]; gns=[]
    STATE["phase"] = "warmup"
    warm1_grads = None
    for i in range(a.skip):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch=next(stream)
            with autocast_context(dev,dtype_name):
                loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
            (loss/accum).backward()
        if i == 0 and ACC_MODE:
            # SAME-STATE gradient capture: first warmup step, identical init/RNG/batch
            # (only mm kernel differs between native and dw_only) - review 5b8fb29 item 1
            tp = {}
            for name, mod in model.named_modules():
                if "intermediate.dense" in name or ("encoder.layer" in name and ".output.dense" in name):
                    if hasattr(mod, "weight") and mod.weight is not None and mod.weight.grad is not None:
                        tp[name] = mod.weight.grad.detach().float().cpu().clone()
            warm1_grads = tp
        gns.append(float(torch.nn.utils.clip_grad_norm_(trainable,1.0)))
        optimizer.step()
    STATE["phase"] = "timed"
    for i in range(a.rounds):
        t0=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        ml=[]; step_batches=[]
        for micro in range(accum):
            batch=next(stream)
            step_batches.append(batch)
            with autocast_context(dev,dtype_name):
                loss=loss_for_batch(model,processor,str(cfg.task),batch,dev,float(cfg.temperature))
            ml.append(round(float(loss.detach()),6))
            (loss/accum).backward()
        gn=float(torch.nn.utils.clip_grad_norm_(trainable,1.0))
        optimizer.step()
        torch.cuda.synchronize()
        if "flush" in STATE: STATE["flush"]()
        ms=(time.perf_counter()-t0)*1000
        # tokenise AFTER timing (outside the measured window)
        n_valid, in_hash = step_tokens_and_hash(step_batches)
        vt[i] = n_valid
        rounds_data.append({"round":i,"step_ms":round(ms,1),"micro_losses":ml,"grad_norm":round(gn,4),
                             "valid_tokens":n_valid,"input_hash":in_hash,
                             "valid_tokens_per_s":round(n_valid/(ms/1000))})
        micro_losses.extend(ml)
        gns.append(gn)
    # accuracy capture: per-micro losses (all rounds) + LAST-round grads (p.grad still set) + params
    acc = {"micro_losses_all_rounds": [r["micro_losses"] for r in rounds_data]}
    if ACC_MODE:
        target_params = {}
        for name, mod in model.named_modules():
            if "intermediate.dense" in name or ("encoder.layer" in name and ".output.dense" in name):
                if hasattr(mod, "weight") and mod.weight is not None:
                    target_params[name] = mod.weight
        acc["target_grad_tensors"] = {n: w.grad.detach().float().cpu().clone() for n, w in target_params.items() if w.grad is not None}
        acc["target_params_post_tensors"] = {n: w.detach().float().cpu().clone() for n, w in target_params.items()}
    if ACC_MODE and GROUP == "native":
        import pickle
        pickle.dump({"micro_losses_all": acc["micro_losses_all_rounds"],
                      "grads": acc["target_grad_tensors"],
                      "warm1_grads": warm1_grads,
                      "params_post": acc["target_params_post_tensors"]},
                    open(a.out + ".native_ref.pkl","wb"))
    # param snapshot hash + result record
    ph = hashlib.sha256()
    for p in trainable:
        ph.update(p.detach().cpu().view(torch.int16).numpy().tobytes() if p.dtype==torch.bfloat16 else p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    param_hash = ph.hexdigest()[:16]
    pmax = max(float(p.detach().abs().max()) for p in trainable)
    res={"group":GROUP,"median_ms":round(statistics.median([r["step_ms"] for r in rounds_data]),1),
          "step_ms":[r["step_ms"] for r in rounds_data],
          "valid_tokens_per_s":[r.get("valid_tokens_per_s") for r in rounds_data],
          "valid_tokens":[r.get("valid_tokens") for r in rounds_data],
          "input_hashes":[r.get("input_hash") for r in rounds_data],
          "micro_losses_last_round":rounds_data[-1]["micro_losses"],
          "grad_norms_warmup":gns[:2],"grad_norms_rounds":[r["grad_norm"] for r in rounds_data],
          "param_hash16":param_hash,"param_absmax":round(pmax,4),
          "wrapper_stats":{f"{k[0]}|{k[1]}":v for k,v in STATE["stats"].items()},
          "cold_first_calls":len(STATE["first_call"])}
    if GROUP == "native":
        ref_accuracy = {"micro_losses_last_round":rounds_data[-1]["micro_losses"],
                         "grad_norms_rounds":[r["grad_norm"] for r in rounds_data],
                         "param_hash16":param_hash}
    elif ref_accuracy is not None:
        dl = [round(x-y,6) for x,y in zip(rounds_data[-1]["micro_losses"], ref_accuracy["micro_losses_last_round"])]
        gn_d = [round(x-y,3) for x,y in zip([r["grad_norm"] for r in rounds_data], ref_accuracy["grad_norms_rounds"])]
        nonfinite = int((~torch.isfinite(torch.cat([p.detach().float().view(-1) for p in trainable]))).sum())
        res["accuracy_vs_native"]={"last_round_micro_loss_diff":dl,
            "grad_norm_diff_rounds":gn_d,
            "param_hash_equal":param_hash==ref_accuracy["param_hash16"],
            "param_absmax":round(pmax,4),"nonfinite_params":nonfinite,
            "criteria":"bf16-level deviation expected for candidate kernels; nonfinite must be 0; param hash equality NOT required (different kernels)"}
        if ACC_MODE and GROUP == "dw_only_routed":
            # BF16 acceptance thresholds DECLARED BEFORE results (review 5b8fb29 item 1)
            import pickle
            TH = {"loss_abs_tol": 0.05, "grad_rel_tol": 0.02, "param_abs_tol": 0.02,
                   "nonfinite_max": 0,
                   "basis": "bf16 mantissa 8 bits -> ~0.4% rel; 2% grad rel covers accumulation; loss abs 0.05 vs values ~0.5-1.0"}
            ref = pickle.load(open(a.out + ".native_ref.pkl","rb"))
            assert "warm1_grads" in ref and warm1_grads is not None, "same-state warm1 grads missing"
            dl_all = [round(x-y,6) for x,y in zip(acc["micro_losses_all_rounds"][-1], ref["micro_losses_all"][-1])]
            gmax_abs = {}; gmax_rel = {}; gmax_rel_raw = {}; g_nonfin = 0
            for n, gt in ref["warm1_grads"].items():
                mine = warm1_grads[n]
                g_nonfin += int((~torch.isfinite(mine)).sum())
                d = (mine - gt).abs()
                gmax_abs[n] = round(float(d.max()), 6)
                # floored rel (SAME convention as accepted AdamW judge):
                # floor = 0.01 * max|ref| per tensor -> near-zero ref positions
                # excluded from rel denominator (raw clamp_min(1e-6) explodes on noise)
                floor = 0.01 * float(gt.abs().max())
                gmax_rel[n] = round(float((d / torch.maximum(gt.abs(), torch.full_like(gt, floor))).max()), 4)
                gmax_rel_raw[n] = round(float((d / gt.abs().clamp_min(1e-6)).max()), 2)
            pmax_abs = {}
            for n, pt in ref["params_post"].items():
                mine = acc["target_params_post_tensors"][n]
                pmax_abs[n] = round(float((mine - pt).abs().max()), 6)
            worst_g_rel = max(gmax_rel.values()); worst_p_abs = max(pmax_abs.values())
            verdict = ("PASS" if (max(abs(x) for x in dl_all) <= TH["loss_abs_tol"]
                                   and worst_g_rel <= TH["grad_rel_tol"]
                                   and worst_p_abs <= TH["param_abs_tol"]
                                   and nonfinite + g_nonfin <= TH["nonfinite_max"]) else "FAIL")
            res["dw_accuracy_acceptance"] = {
                "thresholds_declared": TH,
                "last_round_per_micro_loss_diff": dl_all,
                "target_grad_max_abs_diff": gmax_abs,
                "target_grad_max_rel_diff": gmax_rel,
                "grad_nonfinite": g_nonfin,
                "post_update_param_max_abs_diff": pmax_abs,
                "worst_grad_rel_floored": worst_g_rel,
                "worst_grad_rel_raw_nofloor": max(gmax_rel_raw.values()),
                "floor_convention": "per-tensor floor = 0.01*max|ref| (AdamW judge convention)",
                "grad_compare_scope": "SAME-STATE first-warmup-step target grads (identical init/RNG/batch; only mm kernel differs); end-of-trajectory grads kept as diagnostic (bf16 trajectory divergence expected)",
                "worst_param_abs": worst_p_abs,
                "verdict": verdict}
            print(f"[ACC] verdict={verdict} worst_g_rel={worst_g_rel} worst_p_abs={worst_p_abs} g_nonfin={g_nonfin}", flush=True)
            if verdict == "FAIL":
                json.dump({"fatal":"accuracy FAIL","detail":res["dw_accuracy_acceptance"]}, open(a.out+".accfail.json","w"), indent=1)
                sys.exit(5)

    if GROUP in ("dw_only_routed","dw_native_contig"):
        hits = STATE.get("dw_hits", {})
        timed_total = sum(c for (k, ph), c in hits.items() if ph == "timed")
        warm_total = sum(c for (k, ph), c in hits.items() if ph == "warmup")
        shapes_hit = {k for (k, ph) in hits}
        missing = STATE["dW_shapes"] - shapes_hit
        res["dw_hits_summary"] = {"timed_hits": timed_total, "warmup_hits": warm_total,
                                    "shapes_hit": len(shapes_hit), "shapes_expected": len(STATE["dW_shapes"]),
                                    "missing_shapes": [list(a_)+list(b_) for (a_,b_) in list(missing)[:6]],
                                    "per_shape_timed": {f"{list(k[0])}x{list(k[1])}|{ph}": c for (k,ph),c in sorted(hits.items()) if ph=="timed"}}
        if STATE.get("perf_mode"):
            timed_total = STATE.get("dw_perf_count", 0)
            STATE["dW_shapes"] = STATE.get("dW_shapes")
            if timed_total == 0:
                print("FATAL: dw perf-mode int-counter hits ZERO", file=sys.stderr)
                sys.exit(4)
            res["dw_perf_int_count"] = timed_total
            results[GROUP]=res
            print(f"[{GROUP}] median {res['median_ms']} ms | dw int-hits {timed_total}", flush=True)
            for lk in ("lib","liball","libdw","libn"):
                if STATE.get(lk):
                    try: STATE[lk]._destroy()
                    except Exception: pass
                    STATE[lk]=None
            time.sleep(2)
            continue
        if timed_total == 0 or missing:
            print(f"FATAL: dw_only candidate hits ZERO or missing shapes (timed={timed_total}, missing={len(missing)})", file=sys.stderr)
            json.dump({"group":GROUP,"fatal":"zero/missing dw hits","dw_hits_summary":res["dw_hits_summary"]},
                      open(a.out + ".dw_fatal.json","w"), indent=1)
            sys.exit(4)
    results[GROUP]=res
    print(f"[{GROUP}] median {res['median_ms']} ms | rounds {res['step_ms']} | param {param_hash} | cold_calls {res['cold_first_calls']}", flush=True)
    # teardown override libraries between groups (best effort)
    for lk in ("lib","liball","libdw","libn"):
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
