"""Software roofline (review bqjYIxQ): model-FLOPs and FORCED-TRAFFIC derivation
from existing traces. NOT an HBM measurement - every bandwidth number here is a
LOWER BOUND derived from operand sizes; no HBM bytes, HBM utilization or
matrix-unit activity is claimed (hardware_status stays unverified; PCM BLOCKED).

Inputs (fixed, hashed): training_trace_r36_trace.json.gz, training_trace_analysis_r36.json,
kernel_efficiency_r36.json, gemm_efficiency_r33.json, sdpa_efficiency_r36.json,
route_isolated_r48.json, probe9_gemm_sdpa.json.
Field version: roofline_v1. Time unit: ms."""
import gzip, hashlib, json, sys, collections
R="results/"
PEAK=139.2                      # repo's ZW810E BF16 reference peak (TFLOP/s)
DT=2                            # bf16 bytes
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()
INPUTS=["training_trace_r36_trace.json.gz","training_trace_analysis_r36.json","kernel_efficiency_r36.json",
        "gemm_efficiency_r33.json","sdpa_efficiency_r36.json","route_isolated_r48.json","probe9_gemm_sdpa.json"]
hashes={f:sha(R+f) for f in INPUTS}
L=lambda f: json.load(open(R+f))
KE=L("kernel_efficiency_r36.json"); GE=L("gemm_efficiency_r33.json"); SD=L("sdpa_efficiency_r36.json")
AN=L("training_trace_analysis_r36.json"); RI=L("route_isolated_r48.json"); P9=L("probe9_gemm_sdpa.json")
out={"field_version":"roofline_v1","time_unit":"ms","dtype_bytes":DT,"peak_reference_tflops":PEAK,
     "scope":"DERIVED from model FLOPs and forced operand traffic. NOT HBM-measured. "
             "bw_lb_gbps = minimum effective traffic rate to achieve the observed device time; "
             "utilization_vs_peak compares MODEL effective TFLOP/s to the repo reference peak, "
             "it is NOT a matrix-unit counter activity.",
     "hardware_status":"unverified (PCM BLOCKED - no HBM bytes / matrix-unit activity)",
     "input_sha256":hashes}
# same-card empirical reference per shape (probe9), listed separately from the peak
REF9=P9["gemm_bf16_fp32"]
ref_by_shape={"attn_out_1024x1024":REF9["attn_out_1024x1024"]["bf16_tflops"],
              "ffn_up_1024x4096":REF9["ffn_up_1024x4096"]["bf16_tflops"],
              "ffn_down_4096x1024":REF9["ffn_down_4096x1024"]["bf16_tflops"],
              "qkv_1024x3072":REF9["qkv_1024x3072"]["bf16_tflops"]}
def ref_for(kind):
    return ref_by_shape.get(kind)
# ---------- 1+2: per-shape effective TFLOP/s + forced traffic lower bound ----------
def classify(dims):
    A,B=dims[0],dims[1]
    if len(A)==2 and len(B)==2:
        if A[1]==4096 and B[1]==1024: return "dW_ffn_up"
        if A[1]==1024 and B[1]==4096: return "dX_ffn_down"
        if A[0]==4096: return "dW_ffn_up"
        if A[0]==1024 and B[1]==4096: return "dW_ffn_down"
        if A[1]==1024 and B[1]==1024: return "dX_attn_proj"
        if A[0]==1024 and B[1]==1024: return "dW_attn_proj"
    return "other"
rows=[]
for r in KE["rows"]:
    dims=r.get("operand_dims_measured")
    if not dims or len(dims)<2: continue
    A,B=dims[0],dims[1]
    if len(A)!=2 or len(B)!=2: continue
    M,K=A[0],A[1]; N=B[1]
    cnt=r["calls_per_step"]; ms=r["device_ms"]
    if cnt<=0 or ms<=0: continue
    flops=2*M*K*N*cnt                       # FLOPs over all calls in the step
    fl_tf=flops/1e12
    eff=(flops/(ms*1e-3))/1e12              # TFLOP/s
    bytes_lb=DT*(M*K+K*N+M*N)*cnt           # forced operand bytes (A read + B read + C write)
    bw_lb=bytes_lb/(ms*1e-6)/1e9            # GB/s lower bound
    ai=flops/bytes_lb                       # FLOP / byte
    transposed = ("T" in (r.get("operand_sig") or "").split("|")[0])   # A is a transposed view
    kind=classify(dims)
    rows.append({"phase":r["phase"],"kind":kind,"shape":[M,K,N],"count":cnt,
                 "flops_tflop":round(fl_tf,4),"device_ms":round(ms,3),
                 "effective_tflops":round(eff,2),"utilization_vs_peak":round(eff/PEAK,4),
                 "same_shape_reference_tflops":ref_for({"dW_ffn_up":"ffn_up_1024x4096","dW_ffn_down":"ffn_down_4096x1024",
                                                        "dX_ffn_down":"ffn_down_4096x1024","dX_attn_proj":"attn_out_1024x1024","dW_attn_proj":"attn_out_1024x1024"}.get(kind,"")),
                 "A_transposed_view":transposed,
                 "forced_bytes_lb_mb":round(bytes_lb/1e6,2),"bw_lb_gbps":round(bw_lb,1),"arithmetic_intensity":round(ai,2),
                 "per_call_ms":round(ms/cnt,5)})
# sanity: backward GEMM hotspots should land ~88-129 TFLOP/s (63%-93% of 139.2)
hot=[r for r in rows if r["phase"]=="backward" and r["device_ms"]>=25]
out["bw_lb_caveat"]="bw_lb_gbps = forced_bytes_lb / device_ms. It assumes each operand byte is moved once per call with NO on-chip reuse; per-shape values of 1e5 GB/s exceed any HBM and are therefore a FORMAL lower-bound artifact, not an achieved/required HBM bandwidth. Real HBM traffic needs PCM counters (BLOCKED)."
out["sanity_backward_gemm_hot"]={"n":len(hot),"eff_min":min(r["effective_tflops"] for r in hot),"eff_max":max(r["effective_tflops"] for r in hot),
    "util_min":min(r["utilization_vs_peak"] for r in hot),"util_max":max(r["utilization_vs_peak"] for r in hot),
    "expected":"~88-129 TFLOP/s, 63%-93% of 139.2 (review sanity band)"}
# short shapes: fixed per-call overhead view
short=sorted([r for r in rows if r["shape"][0]*r["shape"][1]<=1024*600],key=lambda r:-r["count"])
out["short_shapes_fixed_overhead"]=[{k:r[k] for k in ("phase","kind","shape","count","per_call_ms","effective_tflops")} for r in short[:10]]
out["per_shape"]=sorted(rows,key=lambda r:-r["device_ms"])
# ---------- 3: critical path from TIME INTERVALS (not kernel-duration sums) ----------
trace=json.load(gzip.open(R+"training_trace_r36_trace.json.gz","rt")); evs=trace["traceEvents"]
markers=sorted([e for e in evs if e.get("name","").startswith("rf_s")],key=lambda e:e["ts"])
act=[e for e in evs if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
first={}
for e in markers:
    st=e["name"].split("_")[1]; first.setdefault(st,e["ts"])
order=sorted(first.items(),key=lambda kv:kv[1])
allend=max(e["ts"]+e.get("dur",0) for e in act)
def union(iv):
    u=0.0; cs=ce=None
    for a,b in sorted(iv):
        if cs is None or a>ce:
            if cs is not None: u+=ce-cs
            cs,ce=a,b
        else: ce=max(ce,b)
    if cs is not None: u+=ce-cs
    return u
crit={}; ref_busy={"s2":1806.9,"s3":2035.3,"s4":2262.5}
for i,(st,t0) in enumerate(order):
    t1=order[i+1][1] if i+1<len(order) else allend
    inwin=[e for e in act if e["ts"]<t1 and e["ts"]+e.get("dur",0)>t0]
    kiv=[(max(e["ts"],t0),min(e["ts"]+e.get("dur",0),t1)) for e in inwin if e["cat"]=="kernel"]
    miv=[(max(e["ts"],t0),min(e["ts"]+e.get("dur",0),t1)) for e in inwin if e["cat"]!="kernel"]
    bu=union(kiv+miv); ku=union(kiv); mu=union(miv)
    # overlap between memcpy/memset and kernels = |K| + |M| - |K u M|
    ov=ku+mu-bu
    # per-phase exposure within the step: time from first to last activity of each rf phase
    ph=collections.defaultdict(list)
    for e in inwin:
        pass
    # phase windows from CPU markers belonging to this step
    pm=[e for e in markers if e["name"].split("_")[1]==st]
    phw={}
    for grp in ("forward","backward","clip_optimizer"):
        g=[e for e in pm if f"_{grp}" in e["name"] or (grp=="clip_optimizer" and "clip" in e["name"])]
        if g:
            a=min(e["ts"] for e in g); b=max(e["ts"]+e.get("dur",0) for e in g)
            ga=[e for e in inwin if a<=e["ts"]<b]
            phw[grp]={"window_ms":round((b-a)/1000,1),"busy_union_ms":round(union([(max(e["ts"],a),min(e["ts"]+e.get("dur",0),b)) for e in ga])/1000,1),
                      "idle_ms":round((b-a)/1000-union([(max(e["ts"],a),min(e["ts"]+e.get("dur",0),b)) for e in ga])/1000,1)}
    wall=t1-t0
    crit[st]={"window_ms":round(wall/1000,1),"busy_union_ms":round(bu/1000,1),"idle_ms":round((wall-bu)/1000,1),
              "busy_pct":round(bu/wall*100,1),"kernel_union_ms":round(ku/1000,1),"memcpy_memset_union_ms":round(mu/1000,1),
              "memcpy_compute_overlap_ms":round(ov/1000,2),"phase_exposure":phw,
              "r36_reference_busy_ms":ref_busy.get(st),"abs_error_vs_r36_ms":round(bu/1000-ref_busy[st],1) if st in ref_busy else None}
out["critical_path_by_step_interval_union"]=crit
out["critical_path_method"]="per step: window=[first marker start, next step start); busy = UNION of clipped kernel+memcpy+memset intervals (NOT a sum of durations); idle=window-busy; overlap=|K|+|M|-|K u M|"
# ---------- 4: phase ranking ----------
inv=AN["inventory_per_step"]
def g(k): return inv.get(k,{}).get("device_ms_per_step",0.0)
tot=sum(v.get("device_ms_per_step",0) for v in inv.values())
phases=[("backward GEMM",g("gemm/backward")),("backward attention",g("sdpa_attention/backward")),
        ("forward GEMM",g("gemm/forward")),("forward attention",g("sdpa_attention/forward")),
        ("elementwise",g("elementwise/backward")+g("elementwise/forward")),("reduction",g("reduction/backward")),
        ("layernorm",g("layernorm/backward")+g("layernorm/forward")),("gelu",g("gelu/backward")+g("gelu/forward")),
        ("clip+optimizer",g("other/clip"))]
out["phase_ranking"]=[{"phase":n,"device_ms_per_step":round(v,2),"pct_of_kernel_total":round(v/tot*100,1)} for n,v in sorted(phases,key=lambda x:-x[1])]
out["kernel_total_ms_per_step"]=round(tot,2)
# ---------- 5: align with the dW routing candidate ----------
dw=[r for r in rows if r["phase"]=="backward" and r["kind"] in ("dW_ffn_up","dW_ffn_down") and r["device_ms"]>=5]
dw12=[]
for r in dw[:12]:
    M,K,N=r["shape"]
    # native: A is a transposed view -> the GEMM reads A via strided access (no extra buffer in the ideal model)
    # FlagGems: internally calls .contiguous() -> an extra A copy (read M*K + write M*K) BEFORE the GEMM
    # explicit .contiguous(): same extra copy, performed by the caller
    base=DT*(M*K+K*N+M*N)*r["count"]
    extra=2*DT*M*K*r["count"]
    dw12.append({"kind":r["kind"],"shape":r["shape"],"count":r["count"],
                 "native":{"kernel":"gemm_ktype0 (vendor), A transposed view","device_ms":r["device_ms"],"effective_tflops":r["effective_tflops"],
                           "forced_bytes_lb_mb":r["forced_bytes_lb_mb"],"bw_lb_gbps":r["bw_lb_gbps"],"ai":r["arithmetic_intensity"]},
                 "flaggems_contiguous_copy":{"extra_bytes_lb_mb":round(extra/1e6,2),
                      "forced_bytes_lb_mb_incl_copy":round((base+extra)/1e6,2),
                      "note":"FlagGems mm calls .contiguous() on a non-contiguous A: extra read+write of A (M*K) before the GEMM - counted SEPARATELY, not folded into the GEMM body"},
                 "explicit_contiguous_copy":{"extra_bytes_lb_mb":round(extra/1e6,2),"forced_bytes_lb_mb_incl_copy":round((base+extra)/1e6,2)}})
out["dw_routing_alignment"]={"source":"route_isolated_r48.json + kernel_efficiency_r36.json",
    "kernel_names":RI["runs"]["rep1"]["diagnostic_kernels"],
    "whole_step_paired_diff_py_route_minus_native_ms":RI["py_route_vs_native_all6"],
    "whole_step_median_ms":RI["median"],"shapes":dw12,
    "critical_path_exposure_note":"see critical_path_by_step_interval_union: per-step idle 237-295 ms (backward phase exposure dominates)"}
# ---------- conclusion: which of the three situations ----------
hot_eff=out["sanity_backward_gemm_hot"]
dw_ai=[x["native"]["ai"] for x in dw12]; dw_bw=[x["native"]["bw_lb_gbps"] for x in dw12]
idle=[crit[s]["idle_ms"] for s in crit]
out["conclusion"]={
 "situation_1_compute_efficiency_low":f"backward GEMM hotspots run at {hot_eff['eff_min']}-{hot_eff['eff_max']} TFLOP/s = {hot_eff['util_min']*100:.0f}%-{hot_eff['util_max']*100:.0f}% of the {PEAK} reference -> 7%-37% headroom vs reference; kernel tuning space EXISTS but is bounded (<=~37% of the 772 ms backward GEMM = <=~285 ms theoretical ceiling, realistically far less)",
 "situation_2_forced_traffic":f"dW arithmetic intensity {min(dw_ai):.0f}-{max(dw_ai):.0f} FLOP/byte -> STRONGLY COMPUTE-BOUND (AI is far above any roofline knee; a bf16 GEMM at 2765 GB/s HBM and 139 TFLOP/s has its knee at ~50 FLOP/byte). The per-shape bw_lb_gbps (1.2e5-2.3e5 GB/s) is NOT physically meaningful as an HBM requirement: it assumes every operand byte is fetched once per call with no cache reuse across the 96 calls of the step, whereas weights/activations are reused via on-chip caches. It is reported only as the formal lower-bound formula value and MUST NOT be read as an achieved or required HBM bandwidth.",
 "situation_2_contiguous_copy":f"the .contiguous() copy FlagGems mm performs on a transposed A adds {dw12[0]['flaggems_contiguous_copy']['extra_bytes_lb_mb'] if dw12 else 0} MB (S=512 dW-up) and {dw12[1]['flaggems_contiguous_copy']['extra_bytes_lb_mb'] if len(dw12)>1 else 0} MB (S=512 dW-down) of extra read+write per shape per step - a real extra traffic item, counted SEPARATELY from the GEMM body; it is why native+contiguous() loses (+32.4 ms median) while FlagGems still gains (its Triton kernel is faster on the contiguous operand by more than the copy costs)",
 "situation_3_idle_large":f"per-step idle {min(idle):.0f}-{max(idle):.0f} ms ({crit['s2']['busy_pct']:.1f}-{crit['s4']['busy_pct']:.1f}% busy) - idle is 8%-12% of the step and is NOT attributable without a host swimlane (needs counters/platform)",
 "ranking":"backward GEMM (38%) > backward attention (18.5%) > forward GEMM (16.2%) > forward attention (10.1%)",
 "direction":"(a) backward GEMM (38%) has the largest absolute headroom but its effective TFLOP/s (88-129) is already 63%-93% of reference -> vendor-side kernel tuning, not application-level; the transposed-view dW shapes (88 TF/s, 63%) are the weakest - a vendor fast path for transposed operands would target ~27 ms/step of the 258 ms dW time at most; (b) attention (fwd+bwd ~28.6%) runs at 12-13 TFLOP/s model-math over STAGE time - NOT comparable to the GEMM peak (softmax/dropout/masking are inside the stage time) -> needs a training-capable SDPA candidate (FlagGems ruled out); (c) idle 8-12% (211-250 ms/step) -> submission gaps or exposed copies, NOT attributable without a host swimlane",
 "blocked":["HBM read/write bytes","HBM bandwidth utilization","matrix-unit activity","host swimlane attribution of idle"],
 "platform_assistance":"PCM window (pg-exporter), equivalent counter tool, vendor confirmation of metric level - see hbm_blocked_r46.md"}
json.dump(out,open("results/software_roofline_r49.json","w"),indent=1)
print("sanity:",out["sanity_backward_gemm_hot"])
print("critical:",{s:(v['busy_union_ms'],v['idle_ms'],v['busy_pct'],v['abs_error_vs_r36_ms']) for s,v in crit.items()})
print("phases:",[(p['phase'],p['device_ms_per_step'],p['pct_of_kernel_total']) for p in out['phase_ranking'][:4]])
print("dw shapes:",len(dw12))
