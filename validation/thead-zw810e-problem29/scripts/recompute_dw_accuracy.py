"""Independent recomputation of the dW accuracy verdict from archived records.
Review bqR2JZg fixes:
  * PER-ELEMENT floored relative error:  |x-ref| / max(|ref|, 0.01*max|ref|, eps)
    then max over the tensor (previous tensor-level floor gave 10% for
    ref=[1,1e-4], x=[1.001,1e-4] where the correct answer is 0.1%).
  * Structural validation BEFORE numbers: for each of the 72 target layers the
    candidate must have the grad key AND the post-update param key, with the
    same shape and element count as the reference, and all finite.
  * Post-update params compared by raw int32 bits; unequal-element count reported.
  * Grads are the CLIPPED values (recorded after clip_grad_norm_, before optimizer.step).
Exit codes: 0 PASS | 2 missing/extra target keys (vs independent 72-key expectation) | 3 zero hits | 4 non-finite | 5 candidate gate FAIL | 6 shape/size mismatch | 7 native self-comparison INVALID
Regression: --selftest runs three scenarios (identical -> 0; emptied post_params -> 2;
ref=[1,1e-4]/x=[1.001,1e-4] -> ~0.1%)."""
import pickle, sys, json, torch, hashlib
TOL=0.02; EPS=1e-12
def grp(n):
    if "intermediate.dense" in n: return "FFN_up"
    if "encoder.layer" in n and ".output.dense" in n and "attention" not in n: return "FFN_down"
    if "attention.output.dense" in n: return "attn_out_proj"
def nonfinite(t): return int(torch.isnan(t).sum()+torch.isinf(t).sum())
def elem_rel_max(x, ref):
    """per-element |x-ref| / max(|ref|, 0.01*max|ref|, eps); max over elements (UNROUNDED)"""
    denom=torch.clamp(torch.maximum(ref.abs(), torch.full_like(ref, 0.01*float(ref.abs().max()))), min=EPS)
    return float(((x-ref).abs()/denom).max())
def validate(ref, cand, label):
    """structural check: keys, shape, numel, finiteness. Returns list of problems."""
    probs=[]
    for kind in ("pre_grads","post_params"):
        Rk, Ck = ref[kind], cand[kind]
        for n in Rk:
            if n not in Ck: probs.append(f"{label}:{kind}:{n} missing"); continue
            if tuple(Rk[n].shape)!=tuple(Ck[n].shape): probs.append(f"{label}:{kind}:{n} shape {tuple(Rk[n].shape)}!={tuple(Ck[n].shape)}")
            if Rk[n].numel()!=Ck[n].numel(): probs.append(f"{label}:{kind}:{n} numel mismatch")
            if nonfinite(Ck[n])>0: probs.append(f"{label}:{kind}:{n} nonfinite={nonfinite(Ck[n])}")
    return probs
# The expected key set below corresponds to the CURRENT BGE-M3 configuration:
# XLM-RoBERTa-large backbone, 24 encoder layers, 3 target weight classes per
# layer (intermediate.dense / output.dense / attention.output.dense) = 72 keys.
# A different model config needs a different expected set (make N_LAYERS /
# key templates configurable before reusing this script elsewhere).
N_LAYERS=24
EXPECTED_CONFIG="BGE-M3 (XLM-RoBERTa-large): 24 encoder layers x 3 target classes"
def expected_keys():
    """Independent expected key set from the MODEL STRUCTURE (BGE-M3: 24 encoder
    layers x 3 target classes = 72). Not derived from the records - so a record
    that silently dropped targets cannot pass (review bqWBBIQ scenario 1)."""
    ks=[]
    for i in range(N_LAYERS):
        ks.append(f"backbone.encoder.layer.{i}.intermediate.dense")          # FFN up
        ks.append(f"backbone.encoder.layer.{i}.output.dense")                # FFN down
        ks.append(f"backbone.encoder.layer.{i}.attention.output.dense")      # attn out proj
    return set(ks)
EXP=expected_keys()
def file_sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()
def run(RECF, OUT):
    R=pickle.load(open(RECF,"rb"))
    ref=R["native_run1"]; dw=R["dw_only"]
    res={"record_file":RECF,"record_sha256":file_sha(RECF),"tol":TOL,"expected_config":EXPECTED_CONFIG,
         "grad_sampling":"CLIPPED grads: recorded after clip_grad_norm_ and before optimizer.step",
         "metric":"per-element |x-ref| / max(|ref|, 0.01*max|ref|, eps=1e-12), max over tensor; UNROUNDED",
         "dw_only_hits_this_step":dw.get("hit_total",0),
         "n_target_layers":len(ref["pre_grads"]),"comparisons":{},"problems":[],"exit_reason":None}
    comps={"A_native_run2_vs_native_run1":R["native_run2"],
           "B_wrapped_stock_vs_native_run1":R["wrapped_stock"],
           "C_dw_only_vs_native_run1":dw}
    structural=[]
    # coverage vs the independent expected key set (72) - for EVERY group
    res["expected_targets"]=len(EXP)
    res["coverage"]={}
    for cname,C in comps.items():
        for kind in ("pre_grads","post_params"):
            have=set(C[kind].keys())
            missing=sorted(EXP-have); extra=sorted(have-EXP)
            res["coverage"][f"{cname}.{kind}"]={"have":len(have),"expected":len(EXP),"missing":len(missing),"extra":len(extra)}
            if missing: structural.append(f"{cname}:{kind}: {len(missing)} expected targets MISSING (e.g. {missing[:2]})")
            if extra: structural.append(f"{cname}:{kind}: {len(extra)} unexpected keys (e.g. {extra[:2]})")
        structural+=validate(ref,C,cname)
        per={}
        for g in ("FFN_up","FFN_down","attn_out_proj"):
            errs=[]; nf_c=0; nf_r=0; n_t=0; uneq=0; numel=0
            for n in ref["pre_grads"]:
                if grp(n)!=g or n not in C["pre_grads"]: continue
                nf_c+=nonfinite(C["pre_grads"][n]); nf_r+=nonfinite(ref["pre_grads"][n]); n_t+=1
                errs.append(elem_rel_max(C["pre_grads"][n], ref["pre_grads"][n]))
            for n in ref["post_params"]:
                if grp(n)!=g or n not in C["post_params"]: continue
                pa=C["post_params"][n].detach().contiguous().view(torch.int32)
                pb=ref["post_params"][n].detach().contiguous().view(torch.int32)
                uneq+=int((pa!=pb).sum()); numel+=pa.numel()
            if n_t==0: structural.append(f"{cname}:{g}: no target tensors found for this group")
            per[g]={"tensors":n_t,"grad_rel_max_unrounded":(max(errs) if errs else None),
                "per_target_unrounded":{n:elem_rel_max(C["pre_grads"][n], ref["pre_grads"][n]) for n in ref["pre_grads"] if grp(n)==g and n in C["pre_grads"]},
                "per_target_param_bits_unequal":{n:int((C["post_params"][n].detach().contiguous().view(torch.int32)!=ref["post_params"][n].detach().contiguous().view(torch.int32)).sum()) for n in ref["post_params"] if grp(n)==g and n in C["post_params"]},
                "per_target_nan_inf":{n:nonfinite(C["pre_grads"][n]) for n in ref["pre_grads"] if grp(n)==g and n in C["pre_grads"]},
                    "grad_rel_median_unrounded":(sorted(errs)[len(errs)//2] if errs else None),
                    "nan_inf_candidate":nf_c,"nan_inf_reference":nf_r,
                    "post_param_bits_unequal":uneq,"post_param_numel":numel,
                    "PASS":(bool(errs) and max(errs)<=TOL)}
        res["comparisons"][cname]=per
    res["problems"]=structural
    total_nf=sum(v["nan_inf_candidate"]+v["nan_inf_reference"] for c in res["comparisons"].values() for v in c.values())
    C=res["comparisons"]["C_dw_only_vs_native_run1"]
    A=res["comparisons"]["A_native_run2_vs_native_run1"]
    res["native_self_comparison_valid"]=all(v["PASS"] for v in A.values())
    res["candidate_vs_native_pass"]=all(v["PASS"] for v in C.values())
    if res["dw_only_hits_this_step"]==0: res["exit_reason"]=3
    elif structural: res["exit_reason"]=(6 if any("shape" in p or "numel" in p for p in structural) else 2)
    elif total_nf>0: res["exit_reason"]=4
    elif not res["native_self_comparison_valid"]: res["exit_reason"]=7
    elif not res["candidate_vs_native_pass"]: res["exit_reason"]=5
    else: res["exit_reason"]=0
    res["verdict"]=("dw-only PASS at {:.0%} per-element floored-rel gate (unrounded recomputation); native self-comparison valid".format(TOL) if res["exit_reason"]==0
                    else f"FAIL/ABORT exit={res['exit_reason']}: {structural[:3] if structural else ('native self-comparison INVALID' if res['exit_reason']==7 else 'gate/nonfinite')}")
    json.dump(res,open(OUT,"w"),indent=1)
    for c,per in res["comparisons"].items():
        for g,v in per.items():
            gm=v['grad_rel_max_unrounded']; gs=f"{gm:.3e}" if gm is not None else "n/a"
            print(f"{c:<34}{g:<14} grad_rel_max {gs} | nf {v['nan_inf_candidate']}/{v['nan_inf_reference']} | param bits unequal {v['post_param_bits_unequal']}/{v['post_param_numel']} | {'PASS' if v['PASS'] else 'FAIL'}")
    print("VERDICT:",res["verdict"]); return res["exit_reason"]
def selftest():
    """Five regression scenarios (review bqWBBIQ): identical -> 0; emptied post_params -> 2;
    per-element formula -> 0.1%; shrunk to 3 targets -> 2; doubled grad in the
    native self-comparison -> 7."""
    import copy, tempfile, io, contextlib
    torch.manual_seed(0)
    full={"pre_grads":{n:torch.randn(4,4) for n in EXP},"post_params":{n:torch.randn(4,4) for n in EXP},"hit_total":5}
    R={"native_run1":copy.deepcopy(full),"native_run2":copy.deepcopy(full),
       "wrapped_stock":copy.deepcopy(full),"dw_only":copy.deepcopy(full)}
    def go(recs):
        f=tempfile.mktemp(suffix=".pkl"); pickle.dump(recs,open(f,"wb"))
        with contextlib.redirect_stdout(io.StringIO()):
            rc=run(f, f+".json")
        return rc, json.load(open(f+".json"))
    bad=0
    def check(name, rc, exp, extra=None):
        nonlocal bad
        ok=(rc==exp) and (extra is None or extra)
        print(("PASS " if ok else "FAIL ")+name+f" (exit {rc})"); bad+=(not ok)
    rc,_=go(R);                                            check("1 identical full records -> exit 0", rc, 0)
    R2=copy.deepcopy(R); R2["dw_only"]["post_params"]={}; rc,_=go(R2); check("2 emptied candidate post_params -> exit 2", rc, 2)
    e=elem_rel_max(torch.tensor([1.001,0.0001]), torch.tensor([1.0,0.0001]))
    check("3 per-element formula ref=[1,1e-4] x=[1.001,1e-4] -> ~0.1%", (0 if abs(e-0.001)<2e-4 else 1), 0); print(f"   value {e:.4%}")
    def shrink(rec):
        keep=[]; seen=set()
        for n in rec["pre_grads"]:
            g=grp(n)
            if g not in seen: seen.add(g); keep.append(n)
        return {"pre_grads":{n:rec["pre_grads"][n] for n in keep},"post_params":{n:rec["post_params"][n] for n in keep},"hit_total":rec.get("hit_total",0)}
    R3={g:shrink(v) for g,v in R.items()}; rc,d=go(R3)
    check("4 all groups shrunk to 3 targets -> exit 2", rc, 2, d["coverage"]["C_dw_only_vs_native_run1.pre_grads"]["missing"]==69)
    R4=copy.deepcopy(R); n0=sorted(R4["native_run2"]["pre_grads"])[0]
    R4["native_run2"]["pre_grads"][n0]=R4["native_run2"]["pre_grads"][n0]*2
    rc,d=go(R4); check("5 doubled grad in native self-comparison -> exit 7", rc, 7, d["native_self_comparison_valid"] is False)
    print("SELFTEST", "PASS" if bad==0 else f"FAIL({bad})"); return 1 if bad else 0

if __name__=="__main__":
    if len(sys.argv)>=2 and sys.argv[1]=="--selftest": sys.exit(selftest())
    sys.exit(run(sys.argv[1], sys.argv[2]))
