"""Round-57 summary: assembled ONLY from the round's result files (no hand-typed numbers).
Usage: python3 scripts/build_summary_r57.py  (run from the directory that holds results/)
"""
import glob, json, os, statistics, sys
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
L=lambda n: json.load(open(os.path.join(R,n)))
ts=L("train_state_r57.json"); tn=L("tokenize_overlap_native_r57.json"); rt=L("route_entry_ab_r57.json")
ge=L("gap_markers_r57.entry.json"); gc=L("gap_markers_r57.capture.json"); smi=L("smi_utilization_r57.json")
hl={k:L(f"host_stage_timeline_r57.{k}.json") for k in ("entry","capture","entry_h2d_nb","entry_gc_track","entry_gc_disabled","entry_12steps")}
KEYS=["step_ms","prepare_embedding_inputs_ms","h2d_ms","forward_launch_ms","forward_call_ms","loss_sync_ms","backward_launch_ms","next_batch_ms","clip_launch_ms","optimizer_step_launch_ms","end_wait_ms"]
def mp(x): return {k:x["mean_per_step"][k] for k in KEYS if k in x["mean_per_step"]}
def split(gaps):
    fw=[x for x in gaps if x["enclosing_host_markers"] and "forward" in x["enclosing_host_markers"][0]["name"]]
    bw=[x for x in gaps if x["enclosing_host_markers"] and "backward" in x["enclosing_host_markers"][0]["name"]]
    oth=[x for x in gaps if x not in fw and x not in bw]
    def s(gs,k): return round(sum(x[k] for x in gs),2)
    sh=sorted(100*x["tokenizer"]/x["gap_ms"] for x in fw); nt=[x for x in fw if x["tokenizer"]/x["gap_ms"]<0.8]
    off=sorted(round(x["next_kernel"]["launch_rel_gap_start_ms"]-x["tokenizer"]-x["prepare_embedding_inputs_other"],3) for x in fw if x["next_kernel"] and x["tokenizer"]/x["gap_ms"]>=0.8)
    return {"n_gaps":len(gaps),"idle_ms":s(gaps,"gap_ms"),"forward_window":{"n_gaps":len(fw),"idle_ms":s(fw,"gap_ms"),"tokenizer_ms":s(fw,"tokenizer"),
             "other_explained_ms":round(s(fw,"gap_ms")-s(fw,"tokenizer")-s(fw,"unknown"),2),"unknown_ms":s(fw,"unknown"),"largest_gap_ms":max([x["gap_ms"] for x in fw] or [0]),
             "unknown_share_pct":round(100*s(fw,"unknown")/s(fw,"gap_ms"),1) if s(fw,"gap_ms") else None,
             "tokenizer_share_of_gap_pct":({"min":round(sh[0],1),"median":round(statistics.median(sh),1),"max":round(sh[-1],1)} if fw else None),
             "gaps_tokenizer_dominated":len(fw)-len(nt),
             "gaps_not_tokenizer_dominated":[{"gap_ms":x["gap_ms"],"tokenizer_ms":x["tokenizer"],"h2d_ms":x["h2d"],"launch_submit_ms":x["launch_submit"],"cpu_op_ms":x["cpu_op"],"other_runtime_ms":x["other_runtime"],"unknown_ms":x["unknown"]} for x in nt],
             "unknown_max_per_gap_ms":max([x["unknown"] for x in fw] or [0]),
             "next_kernel_launch_after_tokenize_end_ms":({"min":off[0],"median":round(statistics.median(off),3),"max":off[-1]} if off else None)},
            "backward_window":{"n_gaps":len(bw),"idle_ms":s(bw,"gap_ms"),"unknown_ms":s(bw,"unknown"),"largest_gap_ms":max([x["gap_ms"] for x in bw] or [0])},
            "other_window":{"n_gaps":len(oth),"idle_ms":s(oth,"gap_ms"),"unknown_ms":s(oth,"unknown")}}
def loop_summary(g):
    st=split([x for s in g["steps"].values() for x in s["gaps"]])
    per=lambda v: round(v/3,2)
    return {"profiled_steps":g["profiled_steps"],"device_busy_pct":g["totals"]["busy_pct"],"device_window_ms_per_step":round(g["totals"]["device_window_ms"]/3,2),
            "gaps_ge_min_total":st,"gaps_ge_min_per_step":{k:per(v) for k,v in st["forward_window"].items() if k.endswith("_ms") and k!="largest_gap_ms" and isinstance(v,(int,float))} | {"forward_n_gaps":round(st["forward_window"]["n_gaps"]/3,1)},
            "totals_gaps_ge_min_partition_ms":g["totals_gaps_ge_min_partition_ms"],"totals_all_gaps_partition_ms":g["totals_all_gaps_partition_ms"],
            "largest_gap_ms":max(x["gap_ms"] for s in g["steps"].values() for x in s["gaps"]),
            "backward_unknown_gaps":[{"step":k,"gap_ms":x["gap_ms"],"unknown_ms":x["unknown"],"prev_kernel":x["prev_kernel"],"next_kernel":x["next_kernel"]["name"][:60] if x["next_kernel"] else None,
                                      "next_kernel_launch_rel_gap_start_ms":x["next_kernel"]["launch_rel_gap_start_ms"] if x["next_kernel"] else None,"top_runtime_calls":x["top_runtime_calls"]} for k,s in g["steps"].items() for x in s["gaps"] if x["enclosing_host_markers"] and "backward" in x["enclosing_host_markers"][0]["name"]] or [None]}
cands={}
for k,c in ts["candidates"].items():
    cands[k]={"verdict":c["verdict"],"candidate_pairs":c["candidate_pairs"],"dense_grad_tensors_differing_max_per_pair":c["dense_grad_tensors_differing_max_per_pair"],
              "dense_grad_max_abs":c["dense_grad_max_abs"],"dense_grad_max_rel":c["dense_grad_max_rel"],"dense_param_tensors_differing_max_per_pair":c["dense_param_tensors_differing_max_per_pair"],
              "dense_param_max_abs":c["dense_param_max_abs"],"dense_optimizer_state_max_abs":c["dense_optimizer_state_max_abs"],"loss_equal_all_pairs":c["loss_equal_all_pairs"],
              "grad_norm_equal_all_pairs":c["grad_norm_equal_all_pairs"],"param_sha256_dense_equal_all_pairs":c["param_sha256_dense_equal_all_pairs"],
              "losses_micro0_native_then_candidate":[round(x[0],6) for x in c["losses_micro"]],"grad_norm_native_then_candidate":c["grad_norm"]}
S=tn["stats_by_run_variant"]; pf=S["run2|prefetch_faithful"]; p1=S["run1|prefetch_loss_reimpl"]; hb=S["run2|h2d_nb"]
def cls(s): lo,hi=s["mean_ci95_ms"]; return "REGRESSION (slower)" if lo>0 else ("GAIN (faster)" if hi<0 else "NO SIGNIFICANT DIFFERENCE")
def sent(name,s): lo,hi=s["mean_ci95_ms"]; return f"{name}: {cls(s)} - slower in {s['n_slower']}/{s['n']}, median {s['median_ms']:+.1f} ms, mean {s['mean_ms']:+.1f} ms, 95% CI of the mean [{lo:+.1f}, {hi:+.1f}], range [{s['min_ms']:+.1f}, {s['max_ms']:+.1f}]"
T=tn["valid_tokens_per_s_run2"]
dd=rt["pooled"]["diff_in_diff"]; lo,hi=dd["ci95_ms"]; sg=rt["process_level_sign_test"]
LS_E=loop_summary(ge); LS_C=loop_summary(gc)
T12=mp(hl["entry_12steps"]); E3=mp(hl["entry"])
out={
 "round":57,"review":"TruNcat3, PR #8 comment 7166338004 (2026-11-25T15:07:23Z)","base_commit":"9bf250be",
 "headlines":[
  "round-56's '-25.2 ms, 12/12 faster' tokenize prefetch is RETRACTED: it was a measurement-path artefact (experiment-only synchronize in compute). On the exact entry loop the faithful prefetch is SLOWER: 12/12, median +27.7 ms.",
  "the round-56 / run-1 prefetch implementation was NOT numerically equivalent to the entry (loss re-implemented outside autocast): loss 0.928161 vs 0.927814, grad norm 9.9375 vs 9.5, 386/386 dense grads differ.",
  "forward gaps are now attributed with explicit markers: of the %d forward gaps >= 2 ms in the profiled entry-loop trace (3 steps) %d are tokenizer-dominated (%.0f-%.0f%% of the gap); forward UNKNOWN is %.2f ms in total (%.1f%%). What is left stays UNKNOWN."%(LS_E["gaps_ge_min_total"]["forward_window"]["n_gaps"],LS_E["gaps_ge_min_total"]["forward_window"]["gaps_tokenizer_dominated"],LS_E["gaps_ge_min_total"]["forward_window"]["tokenizer_share_of_gap_pct"]["median"]-2,LS_E["gaps_ge_min_total"]["forward_window"]["tokenizer_share_of_gap_pct"]["max"],LS_E["gaps_ge_min_total"]["forward_window"]["unknown_ms"],LS_E["gaps_ge_min_total"]["forward_window"]["unknown_share_pct"]),
  "in the unprofiled entry the forward phase is device-bound: the pageable H2D copy is a full stream sync (host blocked 245 ms/step inside move_features) but removing it (pinned + non_blocking, bitwise-identical state) does not change the step time (-1.8 ms median, CI [-10.1, +8.1]).",
  "dW Python route: training state bitwise identical (386/386 dense grads, 391/391 params, optimizer state); on the entry loop diff-in-diff -9.0 ms, 95% CI [-17.5, -0.6], 13/14 processes negative; ~0.4% of a step, candidate only."],
 "item1_training_state_consistency":{"file":"results/train_state_r57.json (+ results/train_state_r57.<candidate>.json)","script":"scripts/train_state_consistency.py",
   "command":"for c in dw_route prefetch_faithful prefetch_loss_reimpl prefetch_loss_reimpl_run1 h2d_nonblocking; do python3 examples/retrieval/train_state_consistency.py --conf examples/retrieval/conf/run/bge_m3_ppu.yaml --candidate $c --out results/train_state_r57.json; done; python3 examples/retrieval/train_state_consistency.py --merge --out results/train_state_r57.json",
   "design":ts["design"],"rng_dropout_data_order":ts["rng_dropout_data_order"],"thresholds":ts["thresholds"],"embedding_tables":ts["embedding_tables"],
   "native_band":ts["native_band_reference"],"dense_param_tensors":ts["dense_param_tensors"],"candidates":cands,"cross_process_warning":ts["cross_process_warning"],
   "verdict":("overlap (prefetch_faithful) vs inline, same snapshot: PASS bitwise - every dense gradient / post-update parameter / optimizer-state tensor identical in all 7 candidate pairs, losses and grad norms equal, 0 non-finite; only the 3 index-accumulated embedding tables differ, and native-vs-native differs on the same tables. "
              "The round-56 / run-1 implementation (prefetch_loss_reimpl_run1) FAILS: it is a different computation, not a prefetch of the same one.")},
 "item2_prefetch_not_a_measurement_artefact":{"file":"results/tokenize_overlap_native_r57.json (+ results/tokenize_overlap_native_r57_run1.json, results/tokenize_overlap_native_r57_run2/*.json)","script":"scripts/tokenize_overlap_native.py",
   "command":"python3 examples/retrieval/tokenize_overlap_native.py --conf examples/retrieval/conf/run/bge_m3_ppu.yaml --mode {inline|prefetch_faithful|h2d_nb} --rep repN --out results/tokenize_overlap_native_r57_run2/repN_<mode>.json  (4 reps: inline->prefetch->h2d_nb, h2d_nb->prefetch->inline, ...)  then  --aggregate <12 files> --prior results/tokenize_overlap_native_r57_run1.json --out results/tokenize_overlap_native_r57.json",
   "loop":tn["loop"],"protocol":tn["protocol"],"modes":tn["modes"],"profiler":False,"batch_hash_mismatch":tn["batch_hash_mismatch_run2"],
   "stats":S,"classification":{k:cls(v) for k,v in S.items()},"per_rep_median_ms":tn["per_rep_median_ms"],"valid_tokens_per_s_run2":T,
   "raw_inline_step_ms_by_step_index":[[p["inline_ms"] for p in tn["pairs"] if p["run"]=="run2" and p["mode"]=="prefetch_faithful" and p["step"]==i] for i in range(3)],
   "run1_note":"run 1 = first r57 run (loop without lookahead, prefetch variant loss_reimpl): 10/12 slower, median +16.6 ms; its file said '11/12' (text error, erratum recorded)",
   "verdict":sent("prefetch_faithful (run 2, exact entry loop)",pf)+"; "+sent("prefetch_loss_reimpl (run 1)",p1)+"; the round-56 gain does not exist on the entry loop. Prefetch CLOSED."},
 "item2b_h2d_nonblocking":{"file":"results/tokenize_overlap_native_r57.json (mode h2d_nb) + results/host_stage_timeline_r57.entry_h2d_nb.json","script":"scripts/tokenize_overlap_native.py, scripts/host_stage_timeline.py",
   "what":"move_features through pinned buffers + non_blocking=True (same bytes, same kernels, same order)",
   "stats":hb,"classification":cls(hb),"valid_tokens_per_s":T["h2d_nb"],
   "host_timeline_blocking_vs_nonblocking":{"blocking":mp(hl["entry"]),"nonblocking":mp(hl["entry_h2d_nb"]),"step_ms_blocking":hl["entry"]["step_ms"] if "step_ms" in hl["entry"] else [s["step_ms"] for s in hl["entry"]["steps"]],"step_ms_nonblocking":[s["step_ms"] for s in hl["entry_h2d_nb"]["steps"]]},
   "training_state":cands["h2d_nonblocking"]["verdict"],
   "verdict":sent("h2d_nb (run 2, exact entry loop)",hb)+" -> not adopted (no gain); kept as the evidence that the forward phase is device-bound"},
 "item3_forward_gap_markers":{"files":"results/gap_markers_r57.entry.json(+.trace.json.gz), results/gap_markers_r57.capture.json(+.trace.json.gz)","script":"scripts/gap_markers.py",
   "command":"python3 examples/retrieval/gap_markers.py --conf examples/retrieval/conf/run/bge_m3_ppu.yaml --loop {entry|capture} --out results/gap_markers_r57.<loop>.json",
   "method":ge["method"],"categories":ge["categories"],"markers_found":ge["marker_names_found"],"min_gap_ms":ge["min_gap_ms"],"time_unit":"ms",
   "entry_loop":LS_E,"capture_loop":LS_C,
   "r36_note":"the r36 trace itself cannot be marked retro-actively; the capture-loop run repeats its exact loop and window (steps 2-4) with the markers added. r36's largest gap (19.24 ms in s2) did not recur (largest forward gap here: %.2f ms)."%LS_C["gaps_ge_min_total"]["forward_window"]["largest_gap_ms"],
   "profiler_free_host_timeline":{"script":"scripts/host_stage_timeline.py","command":"python3 examples/retrieval/host_stage_timeline.py --conf examples/retrieval/conf/run/bge_m3_ppu.yaml --loop {entry|capture} [--h2d nonblocking] [--gc track|disabled] --out results/host_stage_timeline_r57.<tag>.json",
      "stage_definitions":hl["entry"]["stage_definitions"],"entry_3_steps":mp(hl["entry"]),"entry_12_steps":mp(hl["entry_12steps"]),"capture_loop_3_steps":mp(hl["capture"]),
      "tokenize_total_per_step_ms":hl["entry_12steps"]["tokenize_total_per_step_ms"],"tokenize_pct_of_step":hl["entry_12steps"]["tokenize_pct_of_step"],
      "step_ms_entry":[s["step_ms"] for s in hl["entry"]["steps"]],"step_ms_capture":[s["step_ms"] for s in hl["capture"]["steps"]],
      "per_group_entry_last_step":hl["entry"]["steps"][-1]["groups"]},
   "h2d_blocking_probe":hl["entry"]["h2d_probe"],
   "smi_utilization":{"file":"results/smi_utilization_r57.json (+ results/host_stage_timeline_r57.entry_12steps.json.smi.csv)","tool":smi["tool"],"workload":smi["workload"],
      "timed_window_s":smi["timed_window_s"],"n_samples":smi["n_samples"],"utilization_gpu":smi["utilization_gpu"],"utilization_memory":smi["utilization_memory"],"histogram":smi["histogram_utilization_gpu_10pct_bins"],"caveat":smi["caveat"]},
   "gc_hypothesis_test":{"files":"results/host_stage_timeline_r57.entry_gc_track.json, results/host_stage_timeline_r57.entry_gc_disabled.json",
      "collections_per_step_mean":hl["entry_gc_track"]["gc_per_step_mean"],"step_ms_gc_default":[s["step_ms"] for s in hl["entry"]["steps"]],"step_ms_gc_track":[s["step_ms"] for s in hl["entry_gc_track"]["steps"]],"step_ms_gc_disabled":[s["step_ms"] for s in hl["entry_gc_disabled"]["steps"]],
      "result":"REJECTED: Python GC is 5 collections / 0.4 ms per step (max pause 0.18 ms) and gc.disable() leaves the step time unchanged; the ~2-3.4 ms backward gaps of the profiled trace are not GC. They stay UNKNOWN."}},
 "item4_hardware_counters":{"status":"BLOCKED - unchanged","evidence":"results/hbm_blocked_r46.md (PCM held by pg-exporter, acu exit 101)",
   "not_answered":["HBM bytes read/written","HBM bandwidth utilization","matrix-unit activity"],
   "what_is_new":"SMI utilization sampling (100 ms, vendor-defined semantics, raw distribution only) is NOT a counter measurement of the above and is not used as one","hardware_status":"unverified"},
 "item5_records":{"status":"needs maintainer hosting - unchanged","records_pkl_sha256_prefix":"7d00baa3","committed":"72-item summary + 576 per-tensor hashes + scripts/recompute_dw_accuracy.py","missing":"accessible 6.9 GB artifact, split/rejoin instructions, clean-checkout recompute"},
 "dW_route_on_entry_loop":{"file":"results/route_entry_ab_r57.json (+ results/route_entry_ab_r57/*.json, results/route_entry_ab_r57_batch1_console/*.json)","script":"scripts/route_entry_ab.py",
   "command":"python3 examples/retrieval/route_entry_ab.py --conf examples/retrieval/conf/run/bge_m3_ppu.yaml --second {py_route|native} --run runNN --out results/route_entry_ab_r57/runNN_<second>.json  (16 processes interleaved)  then  --aggregate <28 files> --out results/route_entry_ab_r57.json",
   "estimate":rt["estimate"],"pooled":rt["pooled"],"diff_in_diff_by_batch":rt["diff_in_diff_by_batch"],"stats_by_batch_second":rt["stats_by_batch_second"],
   "process_level_sign_test":sg,"batch_hash_mismatch":rt["batch_hash_mismatch"],"valid_tokens_per_s":rt["valid_tokens_per_s"],
   "verdict":("diff-in-diff %+.2f ms, 95%% CI [%+.2f, %+.2f]; py_route phase2-phase1 median %+.1f ms, faster in %d/%d; native control median %+.1f ms; process-level means negative in %d/%d py_route vs %d/%d native processes (Fisher two-sided p=%.4f). "
              "Point estimate ~0.4%% of a step, batch 2 alone is not significant -> candidate only, NOT deployable.")%(dd["mean_ms"],lo,hi,rt["pooled"]["py_route"]["median_ms"],rt["pooled"]["py_route"]["n_faster"],rt["pooled"]["n_py_route"],rt["pooled"]["native"]["median_ms"],sg["py_route"]["negative_mean_diff"],sg["py_route"]["processes"],sg["native"]["negative_mean_diff"],sg["native"]["processes"],sg["fisher_two_sided_p"]),
   "training_state":cands["dw_route"]["verdict"]},
 "loop_fidelity_disclosure":{"note":"rounds <= 56 timed whole steps on the r36 capture loop (no lookahead, no isfinite(loss) sync). The ORIGINAL entry loop is slower by the amounts below; all r57 timing uses the entry loop.",
   "capture_loop_step_ms":[s["step_ms"] for s in hl["capture"]["steps"]],"entry_loop_step_ms":[s["step_ms"] for s in hl["entry"]["steps"]],
   "loss_sync_ms_per_step_entry":hl["entry"]["mean_per_step"]["loss_sync_ms"]},
 "answers":{
   "stable_gain":"NONE that is deployable. Prefetch: +27.7 ms (slower). H2D non-blocking: -1.8 ms (no significant difference). dW Python route: -9.0 ms [-17.5, -0.6] on 42+42 paired steps, bitwise-identical training state, candidate only (~0.4%).",
   "time_breakdown":("per step (entry loop, median step {med} ms of steps 2-4: {s0} / {s1} / {s2}): backward GEMM 772 ms device (38% of kernel time), backward attention 375 ms (18.5%), forward GEMM 330 ms (16.2%), forward attention 205 ms (10.1%) [r36 kernel times]; host over 12 steps: tokenize {tok} ms ({tokp}% of step), H2D block {h2d} ms (waiting for the device), loss sync {ls} ms (waiting for the device), backward launch {bl} ms (host held by the launch queue). Forward gaps in the profiled entry-loop trace: ~{gi} ms/step in gaps >= 2 ms, of which tokenizer ~{tk} ms/step; UNKNOWN ~{uf} ms/step (forward) + ~{ub} ms/step (backward, profiled only). Unprofiled device utilization (SMI): median {smi}%.").format(med=sorted(x["step_ms"] for x in hl["entry"]["steps"])[1],s0=hl["entry"]["steps"][0]["step_ms"],s1=hl["entry"]["steps"][1]["step_ms"],s2=hl["entry"]["steps"][2]["step_ms"],tok=T12["prepare_embedding_inputs_ms"],tokp=hl["entry_12steps"]["tokenize_pct_of_step"],h2d=T12["h2d_ms"],ls=T12["loss_sync_ms"],bl=T12["backward_launch_ms"],gi=round(ge["totals"]["idle_in_gaps_ge_min_ms"]/3,1),tk=round(ge["totals_gaps_ge_min_partition_ms"]["tokenizer"]/3,1),uf=round(LS_E["gaps_ge_min_total"]["forward_window"]["unknown_ms"]/3,1),ub=round(LS_E["gaps_ge_min_total"]["backward_window"]["unknown_ms"]/3,1),smi=smi["utilization_gpu"]["median"]),
   "counter_backed":"none for HBM / matrix units (PCM blocked). The only device-side measurement added in r57 is SMI utilization sampling (vendor-defined semantics, median 96%, raw distribution reported)."}}
json.dump(out,open(os.path.join(R,"summary_r57.json"),"w"),indent=1)
print("summary_r57.json written")
for k in ("entry","capture"):
    g=LS_E if k=="entry" else LS_C; print(k,"| busy%",g["device_busy_pct"],"| fwd gaps:",g["gaps_ge_min_total"]["forward_window"],"| bwd:",g["gaps_ge_min_total"]["backward_window"],"| largest",g["largest_gap_ms"])
print("timeline entry 12 steps:",mp(hl["entry_12steps"])); print("entry 3 steps:",mp(hl["entry"])); print("h2d_nb:",mp(hl["entry_h2d_nb"]))
print(out["dW_route_on_entry_loop"]["verdict"]); print(out["item2_prefetch_not_a_measurement_artefact"]["verdict"]); print(out["item2b_h2d_nonblocking"]["verdict"])
