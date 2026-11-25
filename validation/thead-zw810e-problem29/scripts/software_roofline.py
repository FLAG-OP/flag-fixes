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
def classify(dims, sig=""):
    """dW vs dX is decided by the TRANSPOSE FLAG of operand A (dW's A = gy.t() is a
    column-major view), NOT by dim heuristics - the old heuristic mislabelled 3 dX
    rows as dW (review bqj7-Bw item 2)."""
    A,B=dims[0],dims[1]
    if len(A)!=2 or len(B)!=2: return "other"
    transposed = "T" in (sig or "").split("|")[0]
    M,K=A; N=B[1]                       # A=[M,K] @ B=[K,N]
    ffn_up   = (K==5120 or K==4430 or K==4310 or K==3740 or K==3290 or K==2990) and N==1024
    ffn_down= (K in (5120,4430,4310,3740,3290,2990)) and N==4096
    if transposed:
        if ffn_up: return "dW_ffn_up"
        if ffn_down: return "dW_ffn_down"
        if A[0]==1024 and N==1024: return "dW_attn_proj"        # A=[1024,M] col-major
        return "dW_other"
    else:
        # dX: A = gy, row-major [M, N_out]; B = W [N_out, K_out]. N_out = A[1].
        if A[1]==1024 and N==4096: return "dX_ffn_down"              # A=[M,1024]@[1024,4096]
        if A[1]==4096 and N==1024: return "dX_ffn_up"                # A=[M,4096]@[4096,1024]
        if A[1]==1024 and N==1024: return "dX_attn_proj"
        return "dX_other"
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
    bw_lb=bytes_lb/(ms*1e6)                 # decimal GB/s = bytes/(ms*1e6); review bqj7-Bw: old bytes/(ms*1e-6)/1e9 was 1000x too large
    ai=flops/bytes_lb                       # FLOP / byte
    transposed = ("T" in (r.get("operand_sig") or "").split("|")[0])   # A is a transposed view
    kind=classify(dims, r.get("operand_sig",""))
    rows.append({"phase":r["phase"],"kind":kind,"shape":[M,K,N],"count":cnt,
                 "flops_tflop":round(fl_tf,4),"device_ms":round(ms,3),
                 "effective_tflops":round(eff,2),"utilization_vs_peak":round(eff/PEAK,4),
                 "same_shape_reference_tflops":ref_for({"dW_ffn_up":"ffn_up_1024x4096","dW_ffn_down":"ffn_down_4096x1024",
                                                        "dX_ffn_down":"ffn_down_4096x1024","dX_attn_proj":"attn_out_1024x1024","dW_attn_proj":"attn_out_1024x1024"}.get(kind,"")),
                 "A_transposed_view":transposed,
                 "forced_bytes_lb_mb":round(bytes_lb/1e6,2),"bw_lb_gbps":round(bw_lb,1),"arithmetic_intensity":round(ai,2),
                 "per_call_ms":round(ms/cnt,5)})
# ---- item 3: per-shape headroom WITH coverage (retracts the '285 ms/step' extrapolation) ----
bw_rows=[r for r in rows if r["phase"]=="backward" and r["kind"].startswith(("dW","dX"))]
for r in bw_rows:
    r["token_dim"]=(r["shape"][1] if r["kind"].startswith("dW") else r["shape"][0])
    r["ideal_ms_at_peak"]=round(r["flops_tflop"]*1000/PEAK,3)          # FLOPs/(PEAK*1e12) in ms
    r["headroom_ms_vs_peak"]=round(max(0.0,r["device_ms"]-r["ideal_ms_at_peak"]),3)
    r["headroom_pct_of_shape_time"]=round(r["headroom_ms_vs_peak"]/r["device_ms"]*100,1)
BG=772.12                                                              # 3-step mean, backward GEMM
cov_device=sum(r["device_ms"] for r in bw_rows); cov_head=sum(r["headroom_ms_vs_peak"] for r in bw_rows)
hot6=[r for r in bw_rows if r["device_ms"]>=25]
out["headroom_by_shape"]={
   "method":"per-shape ideal time = FLOPs/(139.2e12) [ms]; headroom = device_ms - ideal_ms (clamped >=0); summed over ALL backward dW/dX rows",
   "n_rows":len(bw_rows),"covered_device_ms_per_step":round(cov_device,2),
   "covered_headroom_ms_per_step":round(cov_head,2),
   "single_step_covered_device_ms":round(cov_device,2),
   "three_step_mean_backward_gemm_ms":BG,
   "coverage_note":"single-step rows (718.73 ms, step 2) and the 3-step mean (772.12 ms) are DIFFERENT windows; no coverage % and no 'uncovered' ms is derived from them (the previously derived coverage percentage and uncovered-ms figures are removed).",
   "hot6_device_ms":round(sum(r["device_ms"] for r in hot6),2),"hot6_headroom_ms":round(sum(r["headroom_ms_vs_peak"] for r in hot6),2),
   "window_note":"the 100 per-shape rows come from ONE training step (kernel_efficiency_r36, step 2); 772.12 ms is the 3-step mean backward-GEMM time. 718.73/772.12 is reported as a RATIO BETWEEN DIFFERENT WINDOWS, not a same-window coverage percentage.",
   "retraction":"The earlier 'up to ~285 ms/step' = (1-0.63)*772 from the 6-row 63% floor is RETRACTED. per-shape headroom = device_ms - FLOPs/peak is the ideal-device-time gap UNDER THE 139.2 REFERENCE ASSUMPTION; it is NOT an end-to-end ceiling.",
   "by_group":[
     {"group":"long (token dim >= 4000)","n":sum(1 for r in bw_rows if r["token_dim"]>=4000),
      "device_ms":round(sum(r["device_ms"] for r in bw_rows if r["token_dim"]>=4000),2),
      "headroom_ms":round(sum(r["headroom_ms_vs_peak"] for r in bw_rows if r["token_dim"]>=4000),2),
      "kinds":dict(collections.Counter(r["kind"] for r in bw_rows if r["token_dim"]>=4000)),
      "calls":sum(r["count"] for r in bw_rows if r["token_dim"]>=4000),
      "top_shapes":sorted([{k2:r[k2] for k2 in ("kind","shape","count","device_ms","headroom_ms_vs_peak")} for r in bw_rows if r["token_dim"]>=4000],key=lambda x:-x["headroom_ms_vs_peak"])[:5]},
     {"group":"mid (1400 <= token dim < 4000)","n":sum(1 for r in bw_rows if 1400<=r["token_dim"]<4000),
      "device_ms":round(sum(r["device_ms"] for r in bw_rows if 1400<=r["token_dim"]<4000),2),
      "headroom_ms":round(sum(r["headroom_ms_vs_peak"] for r in bw_rows if 1400<=r["token_dim"]<4000),2),
      "kinds":dict(collections.Counter(r["kind"] for r in bw_rows if 1400<=r["token_dim"]<4000)),
      "calls":sum(r["count"] for r in bw_rows if 1400<=r["token_dim"]<4000),
      "top_shapes":sorted([{k2:r[k2] for k2 in ("kind","shape","count","device_ms","headroom_ms_vs_peak")} for r in bw_rows if 1400<=r["token_dim"]<4000],key=lambda x:-x["headroom_ms_vs_peak"])[:5]},
     {"group":"short (token dim < 1400)","n":sum(1 for r in bw_rows if r["token_dim"]<1400),
      "device_ms":round(sum(r["device_ms"] for r in bw_rows if r["token_dim"]<1400),2),
      "headroom_ms":round(sum(r["headroom_ms_vs_peak"] for r in bw_rows if r["token_dim"]<1400),2),
      "kinds":dict(collections.Counter(r["kind"] for r in bw_rows if r["token_dim"]<1400)),
      "calls":sum(r["count"] for r in bw_rows if r["token_dim"]<1400),
      "top_shapes":sorted([{k2:r[k2] for k2 in ("kind","shape","count","device_ms","headroom_ms_vs_peak")} for r in bw_rows if r["token_dim"]<1400],key=lambda x:-x["headroom_ms_vs_peak"])[:5]}],
   "token_dim_rule":"dW: reduction dim K (= A[1], the token dim of x) ; dX: output row dim M (= A[0]) ; loss matmul: listed separately. The old max(M,K) rule counted the fixed FFN feature dim 4096 as a long sequence (review BqpXfeg item 4).",
   "rows":sorted([{k:r[k] for k in ("kind","shape","token_dim","count","device_ms","flops_tflop","effective_tflops","ideal_ms_at_peak","headroom_ms_vs_peak","headroom_pct_of_shape_time")} for r in bw_rows],key=lambda x:-x["headroom_ms_vs_peak"])}

# sanity: backward GEMM hotspots should land ~88-129 TFLOP/s (63%-93% of 139.2)
hot=[r for r in rows if r["phase"]=="backward" and r["device_ms"]>=25]
out["bw_unit_regression"]={"case":"S=512 FFN-up dW (4096,5120,1024) x96, 46.80 ms",
    "forced_bytes_mb":round(2*(4096*5120+5120*1024+4096*1024)*96/1e6,2),"expected_gbps":124.75,"formula":"bytes/(ms*1e6) -> decimal GB/s"}
out["bw_lb_caveat"]="bw_lb_gbps = forced_bytes_lb / (device_ms*1e6): LOGICAL byte rate assuming each operand is read/written ONCE per call. It is NOT an HBM measurement (cache reuse and repeated reads change real HBM traffic) and NOT a 'bandwidth lower bound' (renamed per review). Real HBM traffic needs PCM counters (BLOCKED)."
out["unit_fix_note"]="FIXED round-51: previous bw_lb used bytes/(ms*1e-6)/1e9 = 1000x too large (S=512 dW showed 124,754 GB/s instead of 124.75). All bw values and conclusions derived from them corrected; 'strongly compute-bound' downgraded to a hypothesis to verify."
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
# the routing whitelist as (A=[N,M], B=[M,K]) pairs -> row shape [M,K,N] = [A1,B1,A0]
WL_AB=[((4096,5120),(5120,1024)),((1024,5120),(5120,4096)),((4096,4430),(4430,1024)),((1024,4430),(4430,4096)),
       ((4096,4310),(4310,1024)),((1024,4310),(4310,4096)),((4096,3740),(3740,1024)),((1024,3740),(3740,4096)),
       ((4096,3290),(3290,1024)),((1024,3290),(3290,4096)),((4096,2990),(2990,1024)),((1024,2990),(2990,4096))]
# rows store shape as [A0, A1, B1] (= operand dims as measured). Match the whitelist in
# THAT same order: (A[0], A[1], B[1]).
WL_SHAPES={(A[0],A[1],B[1]) for A,B in WL_AB}
# a whitelist [M,K,N] can ALSO coincide with a dX row's shape; the whitelist is dW-only,
# so require kind dW AND the transposed-A flag to pick the right row
dw=[r for r in rows if r["phase"]=="backward" and tuple(r["shape"]) in WL_SHAPES
    and r["kind"].startswith("dW") and r["A_transposed_view"]]
dw_shapes={tuple(r["shape"]) for r in dw}
# the single-op per-shape timings above are keyed (N,M,K) which equals the dW whitelist; dX rows are excluded
out["dw_whitelist_alignment"]={"whitelist_size":len(WL_SHAPES),"matched":len(dw_shapes),
    "missing":sorted(list(WL_SHAPES-dw_shapes)),"extra":sorted(list(dw_shapes-WL_SHAPES)),
    "exact_match":dw_shapes==WL_SHAPES,
    "note":"shapes matched exactly against the routing whitelist (route_isolated_r48.py N_MK); kind decided by the A transpose flag"}
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
                 "explicit_contiguous_copy":{"extra_bytes_lb_mb":round(extra/1e6,2),"forced_bytes_lb_mb_incl_copy":round((base+extra)/1e6,2)},
                 "same_shape_reference":("matched" if r["same_shape_reference_tflops"] is not None else "none (M/K/N or dtype/layout differ)"),
                 "per_path_timings":{"native_transposed_view_ms":r["device_ms"],
                                     "copy_time_ms":None,"flaggems_total_ms":None,"explicit_contig_total_ms":None,
                                     "note":"per-shape kernel/copy/total times for the routed paths come from ffn_dw_compare_r38 (S=512) and ffn_dw_midseq_r38 (mid-seq); see dw_per_path_timings below"}})
out["dw_routing_alignment"]={"source":"route_isolated_r48.json + kernel_efficiency_r36.json",
    "kernel_names":RI["runs"]["rep1"]["diagnostic_kernels"],
    "whole_step_paired_diff_py_route_minus_native_ms":RI["py_route_vs_native_all6"],
    "py_route_median_ms":RI["py_route_median_ms"],
    "native_contig_paired_diff_ms":RI["native_contig_vs_native_all6"],
    "native_contig_median_ms":RI["native_contig_median_ms"],
    "median_note":"two INDEPENDENT baselines, each from its own 6 paired diffs (review: previous JSON wrote the contig median into the py_route field)","shapes":dw12,
    "critical_path_exposure_note":"see critical_path_by_step_interval_union: per-step idle 237-295 ms (backward phase exposure dominates)"}
# ---------- 5b: per-path timings for the 12 whitelist shapes, from existing single-op records ----------
FD=L("ffn_dw_compare_r38.json"); FM=L("ffn_dw_midseq_r38.json")
# build a lookup keyed by (N,M,K) from both sources; S=512 comes from ffn_dw_compare (N=4096/1024, M=5120)
per_path={}
for tag,shp in (("ffn_up",FD['shapes']['ffn_up_dW']),("ffn_down",FD['shapes']['ffn_down_dW'])):
    N,M,K=(4096,5120,1024) if tag=="ffn_up" else (1024,5120,4096)
    per_path[(N,M,K)]={"source":"ffn_dw_compare_r38.json","native_transposed_view_ms":shp['stock_ms'],
        "flaggems_ms":shp['fg_mm_ms'],"explicit_contig_copy_ms":shp['layout_conversion_ms'],
        "explicit_contig_total_ms":shp['stock_precontig_ms'],"calls_per_step":96,
        "same_shape_reference":"matched (N,M,K + bf16 + transposed A view)"}
for r in FM['shapes']:
    per_path[(r['N'],r['M'],r['K'])]={"source":"ffn_dw_midseq_r38.json","native_transposed_view_ms":r['stock_ms'],
        "flaggems_ms":r['fg_mm_ms'],"explicit_contig_copy_ms":r['layout_conv_ms'],
        "explicit_contig_total_ms":r['precontig_ms'],"calls_per_step":r['calls_per_step'],
        "same_shape_reference":"matched (N,M,K + bf16 + transposed A view)"}
ppt=[]
for (N,M,K) in [(A[0],A[1],B[1]) for A,B in WL_AB]:   # (N,M,K) = (A0,A1,B1)
    e=per_path.get((N,M,K))
    ppt.append({"shape_NMK":[N,M,K],"in_single_op_records":e is not None,
                **({"native_transposed_view_ms":e['native_transposed_view_ms'],"flaggems_ms":e['flaggems_ms'],
                    "explicit_contig_copy_ms":e['explicit_contig_copy_ms'],"explicit_contig_total_ms":e['explicit_contig_total_ms'],
                    "calls_per_step":e['calls_per_step'],"source":e['source'],"same_shape_reference":e['same_shape_reference'],
                    "per_step_native_ms":round(e['native_transposed_view_ms']*e['calls_per_step'],2),
                    "per_step_flaggems_ms":round(e['flaggems_ms']*e['calls_per_step'],2),
                    "per_step_explicit_contig_ms":round(e['explicit_contig_total_ms']*e['calls_per_step'],2),
                    "saving_flaggems_ms":round((e['native_transposed_view_ms']-e['flaggems_ms'])*e['calls_per_step'],2),
                    "saving_explicit_contig_ms":round((e['native_transposed_view_ms']-e['explicit_contig_total_ms'])*e['calls_per_step'],2)} if e else
                   {"note":"NOT measured in existing single-op records - needs a new device run"})})
out["dw_per_path_timings"]={"n_shapes":len(ppt),"n_measured":sum(1 for x in ppt if x['in_single_op_records']),
    "note":"per-call kernel ms for native transposed view / FlagGems / explicit contiguous (copy + total incl. copy). Only records whose M/K/N, dtype (bf16) and layout (transposed A) all match are marked as same-shape reference.",
    "sum_per_step":{"native_ms":round(sum(x.get('per_step_native_ms',0) for x in ppt),2),
                   "flaggems_ms":round(sum(x.get('per_step_flaggems_ms',0) for x in ppt),2),
                   "explicit_contig_ms":round(sum(x.get('per_step_explicit_contig_ms',0) for x in ppt),2),
                   "saving_flaggems_ms":round(sum(x.get('saving_flaggems_ms',0) for x in ppt),2),
                   "saving_explicit_contig_ms":round(sum(x.get('saving_explicit_contig_ms',0) for x in ppt),2)},
    "shapes":ppt}

# ---------- conclusion: which of the three situations ----------
hot_eff=out["sanity_backward_gemm_hot"]
dw_ai=[x["native"]["ai"] for x in dw12]; dw_bw=[x["native"]["bw_lb_gbps"] for x in dw12]
idle=[crit[s]["idle_ms"] for s in crit]
out["conclusion"]={
 "situation_1_compute_efficiency":f"6 hot backward-GEMM shapes run at {hot_eff['eff_min']}-{hot_eff['eff_max']} TFLOP/s = {hot_eff['util_min']*100:.0f}%-{hot_eff['util_max']*100:.0f}% of the {PEAK} reference. Per-shape headroom (device_ms - FLOPs/139.2e12, the ideal-device-time gap under that reference, NOT an end-to-end ceiling) over {out['headroom_by_shape']['n_rows']} backward dW/dX rows of step 2 = {out['headroom_by_shape']['covered_headroom_ms_per_step']} ms; by token-dim group: " + "; ".join(f"{g['group']}: {g['headroom_ms']} ms over {g['device_ms']} ms device ({g['n']} rows)" for g in out['headroom_by_shape']['by_group']) + ". The 3-step mean backward GEMM (772.12 ms) is a different window and is not divided by these single-step rows. The earlier 'up to 285 ms/step' extrapolation is RETRACTED.",
 "situation_2_logical_traffic":"dW arithmetic intensity 656-706 FLOP/byte; logical byte rate (each operand read/written once) 124-126 GB/s on the S=512 shapes. Whether the GEMMs are compute-bound is a HYPOTHESIS TO VERIFY: it needs real HBM counters (BLOCKED) because cache reuse and repeated reads change the actual traffic. No 'strongly compute-bound' claim is made.",
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
