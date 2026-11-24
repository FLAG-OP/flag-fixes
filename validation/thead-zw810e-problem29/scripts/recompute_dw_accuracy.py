"""Review bqQqCnw item 2: independent recomputation of the dW accuracy verdict
from the archived records pickle (no reuse of the original compare logic).
Reference = native_run1. Metric = per-element floored relative error on the
CLIPPED gradient (grad_norm was taken via clip_grad_norm_; grads recorded
after clipping, before optimizer.step - stated explicitly).
Saves UNROUNDED errors, per-group NaN/Inf counts, post-update param bit
comparison with unequal-element counts. Exit codes:
  0 PASS | 2 missing targets | 3 zero hits | 4 non-finite | 5 gate FAIL"""
import pickle, sys, json, torch, hashlib
RECF=sys.argv[1]; OUT=sys.argv[2]; TOL=0.02
R=pickle.load(open(RECF,"rb"))
def grp(n):
    if "intermediate.dense" in n: return "FFN_up"
    if "encoder.layer" in n and ".output.dense" in n and "attention" not in n: return "FFN_down"
    if "attention.output.dense" in n: return "attn_out_proj"
def nonfinite(t): return int(torch.isnan(t).sum()+torch.isinf(t).sum())
def bits(t): return t.detach().contiguous().view(torch.int16)   # float32 stored -> use raw bits via int32
ref=R["native_run1"]; res={"record_file":RECF,"tol":TOL,
    "grad_sampling":"grads recorded AFTER clip_grad_norm_ and BEFORE optimizer.step (clipped values)",
    "metric":"per-element |x-ref| / max(|ref|, 0.01*max|ref|)) -> max over tensor; UNROUNDED",
    "groups":{},"exit_reason":None}
dw=R["dw_only"]; hits=dw.get("hit_total",0)
res["dw_only_hits_this_step"]=hits
missing=[n for n in ref["pre_grads"] if n not in dw["pre_grads"]]
res["missing_targets"]=missing
comparisons={"A_native_run2_vs_native_run1":R["native_run2"],
             "B_wrapped_stock_vs_native_run1":R["wrapped_stock"],
             "C_dw_only_vs_native_run1":dw}
for cname,C in comparisons.items():
    per={}
    for g in ("FFN_up","FFN_down","attn_out_proj"):
        errs=[]; nf_ref=0; nf_c=0; n_t=0
        for n in ref["pre_grads"]:
            if grp(n)!=g or n not in C["pre_grads"]: continue
            a=C["pre_grads"][n]; b=ref["pre_grads"][n]
            nf_c+=nonfinite(a); nf_ref+=nonfinite(b); n_t+=1
            floor=0.01*float(b.abs().max())
            e=float((a-b).abs().max()/max(floor,1e-12))
            errs.append(e)
        # post-update params: raw bit comparison
        uneq=0; numel=0; maxabs=0.0
        for n in ref["post_params"]:
            if grp(n)!=g or n not in C["post_params"]: continue
            pa=C["post_params"][n].detach().contiguous().view(torch.int32)
            pb=ref["post_params"][n].detach().contiguous().view(torch.int32)
            uneq+=int((pa!=pb).sum()); numel+=pa.numel()
            maxabs=max(maxabs,float((C["post_params"][n]-ref["post_params"][n]).abs().max()))
        per[g]={"tensors":n_t,"grad_rel_max_unrounded":max(errs) if errs else None,
                "grad_rel_median_unrounded":(sorted(errs)[len(errs)//2] if errs else None),
                "nan_inf_candidate":nf_c,"nan_inf_reference":nf_ref,
                "post_param_bits_unequal":uneq,"post_param_numel":numel,"post_param_absdiff_max_unrounded":maxabs,
                "PASS":(max(errs)<=TOL if errs else False)}
    res["groups"][cname]=per
# verdict + exit code
C=res["groups"]["C_dw_only_vs_native_run1"]
gate_pass=all(v["PASS"] for v in C.values())
total_nf=sum(x["nan_inf_candidate"]+x["nan_inf_reference"] for c in res["groups"].values() for x in c.values())
if missing: res["exit_reason"]=2
elif hits==0: res["exit_reason"]=3
elif total_nf>0: res["exit_reason"]=4
elif not gate_pass: res["exit_reason"]=5
else: res["exit_reason"]=0
res["verdict"]=("dw-only PASS at %.0f%% floored-rel gate on the hit step (unrounded recomputation)"%(TOL*100)
                if res["exit_reason"]==0 else f"FAIL/ABORT exit={res['exit_reason']}")
json.dump(res,open(OUT,"w"),indent=1)
for c,per in res["groups"].items():
    for g,v in per.items():
        print(f"{c:<34}{g:<14} grad_rel_max {v['grad_rel_max_unrounded']:.3e} | nf {v['nan_inf_candidate']}/{v['nan_inf_reference']} | param bits unequal {v['post_param_bits_unequal']}/{v['post_param_numel']} | {'PASS' if v['PASS'] else 'FAIL'}")
print("VERDICT:",res["verdict"]); sys.exit(res["exit_reason"])
