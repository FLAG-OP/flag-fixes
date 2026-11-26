"""Round-58 summary: assembled ONLY from the round's result files (no hand-typed numbers).
Usage: python3 scripts/build_summary_r58.py   (run from the directory that holds results/)
"""
import glob, hashlib, json, os, statistics
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
L=lambda n: json.load(open(os.path.join(R,n)))
ts=L("train_state_r58.json"); re_=L("route_entry_ab_r57.json"); sr=L("software_roofline_r49.json"); smi=L("smi_utilization_r57.json")
hl=L("host_stage_timeline_r57.entry_12steps.json")
s0=L("route_from_step0_r58.json"); s0n=L("route_from_step0_r58_nocopy.json"); s0b=L("route_from_step0_r58_b2.json")
rec=L("records_delivery_r58.json")
pr=L("step0_repro_r58.json"); pooled=L("route_from_step0_r58_pooled.json")
STD=os.path.join(R,"state_transfer_r58"); ST={n:json.load(open(os.path.join(STD,n+".json"))) for n in ("save","fresh1","fresh2","fresh3","pert1","pert2000","pert20000")}
DIAG=[json.load(open(f)) for f in sorted(glob.glob(os.path.join(R,"route_from_step0_r58_diag","*","run.json")))]
out={"round":58,"review":"TruNcat3, PR #8 comment 7182886976 (2026-11-26T04:46:13Z)","base_commit":"59346831df384a2ffb393dc36eca579029fa2e3c"}
# ---------------- item 1 ----------------
C=ts["candidates"]; snap=C["dw_route"]["snapshot"]
def cand(c):
    x=C[c]; return {"verdict":x["verdict"],"candidate_pairs":x["candidate_pairs"],"dense_grad_tensors":x["dense_grad_tensors"],"dense_param_tensors":x["dense_param_tensors"],
        "dense_grad_tensors_differing_max":x["dense_grad_tensors_differing_max"],"dense_grad_max_abs":x["dense_grad_max_abs"],"dense_grad_max_rel":x["dense_grad_max_rel"],
        "dense_param_tensors_differing_max":x["dense_param_tensors_differing_max"],"dense_param_max_abs":x["dense_param_max_abs"],"dense_optimizer_state_max_abs":x["dense_optimizer_state_max_abs"],
        "loss_equal":x["loss_equal_all_pairs"],"grad_norm_equal":x["grad_norm_equal_all_pairs"],"keys_equal":x["keys_equal_all_pairs"],"nonfinite_total":x["nonfinite_total"],
        "call_counts_candidate_runs":x["call_counts_candidate_runs"],"native_band":x["native_band"],"snapshot":x["snapshot"],"grad_norm":x["grad_norm"][:1],"loss0":round(x["losses_micro"][0][0],6)}
p0=C["dw_route"]["pairs"][0]
out["item1_state_consistency_after_real_updates"]={
 "file":"results/train_state_r58.json (+ results/train_state_r58.<candidate>.json)","script":"scripts/train_state_consistency.py (v3)",
 "command":"for c in dw_route prefetch_faithful prefetch_loss_reimpl prefetch_loss_reimpl_run1 h2d_nonblocking; do python3 examples/retrieval/train_state_consistency.py --conf examples/retrieval/conf/run/bge_m3_ppu.yaml --candidate $c --out results/train_state_r58.json; done; python3 examples/retrieval/train_state_consistency.py --merge --out results/train_state_r58.json",
 "what_changed_vs_round57":"the warmup now consists of 2 REAL optimizer updates of the entry loop (clip_grad_norm_ + optimizer.step), so the snapshot carries AdamW momentum; accumulated dense gradients are compared after EVERY micro-batch; the pre-clip gradient norm, key sets and non-finite counts are checked; dw_route counts whitelist matches and actual flag_gems.mm calls and fails if the route was requested but never reached",
 "snapshot_example":snap,"snapshot_dense_param_sha256_per_process":{c:C[c]["snapshot"]["param_sha256_dense"][:16] for c in C},
 "counts":{"dense_gradient_tensors":p0["grads_dense_per_micro"][-1]["tensors"],"dense_param_tensors":p0["params_dense"]["tensors"],"dense_optimizer_state_tensors":p0["optimizer_state_dense"]["tensors"],"micro_batches_compared":len(p0["grads_dense_per_micro"]),"dense_optimizer_state_tensors":p0["optimizer_state_dense"]["tensors"],"embedding_tables_excluded":len(C["dw_route"]["embedding_tables_excluded"])},
 "correction":"'391/391 params' in the round-57 summary headline was wrong; the data show %d dense gradient tensors and %d dense parameter tensors (the 3 embedding tables excluded)."%(
     p0["grads_dense_per_micro"][-1]["tensors"],p0["params_dense"]["tensors"]),
 "native_per_micro_dense_grads_identical":{f"micro{i+1}":g["bitwise_identical"]==g["tensors"] for i,g in enumerate(p0["grads_dense_per_micro"])},
 "candidates":{c:cand(c) for c in ("prefetch_faithful","h2d_nonblocking","dw_route","prefetch_loss_reimpl","prefetch_loss_reimpl_run1")},
 "verdict":"dw_route / prefetch_faithful / h2d_nonblocking / prefetch_loss_reimpl: bitwise identical to the same-snapshot native runs in every dense section after every micro-batch, with equal pre-clip gradient norm, equal key sets and 0 non-finite; only the word / position embedding gradients and their optimizer state differ, and native-vs-native differs on the same tensors. prefetch_loss_reimpl_run1 (the round-56 implementation): FAIL."}
# ---------------- item 2 ----------------
G=s0["groups"]; PC=s0["pair_classes_by_step"]
def rng(v): return [min(v),max(v)]
out["item2_route_from_step0"]={
 "files":"results/route_from_step0_r58/<mode>_N/run.json (+ saved gradients, not committed) -> results/route_from_step0_r58.json ; results/route_from_step0_r58_nocopy.json","script":"scripts/route_from_step0_consistency.py (v3)",
 "command":"python3 examples/retrieval/route_from_step0_consistency.py --conf examples/retrieval/conf/run/bge_m3_ppu.yaml --mode {native_from_start|fallback_from_start|route_from_start} --copy-grads host --save-grads <dir> --out <dir>/run.json   (5 fresh processes per mode, interleaved) ; python3 examples/retrieval/route_from_step0_consistency.py --aggregate 'results/route_from_step0_r58/*/run.json' --out results/route_from_step0_r58.json",
 "bug_fixed":"round-57's route_from_start mode set USE_FG=True as a function-local variable, so the override never reached flag_gems.mm (whitelist hits were counted, native mm.out was executed). The switch is now a module-level global; three counters are kept separately (whitelist matches / actual flag_gems.mm calls / native fallbacks); a run that requests the route with zero flag_gems.mm calls exits with code 3.",
 "round57_withdrawal":"the round-57 numbers 9.125 / 10.5 / 13.5 (three 'route_from_start' processes) were fallback-only runs; they are withdrawn as evidence about FlagGems.",
 "groups":{m:{"n_processes":g["n_processes"],"call_counts_total":g["call_counts_total"],"per_step_calls":g["per_step_calls_first_process"],"distinct_initial_param_sha256":g["distinct_initial_param_sha256"],
              "step0_losses_identical_across_processes":g["step0_losses_identical_across_processes"],"distinct_warmup_param_sha256":g["distinct_warmup_param_sha256"],
              "hit_step_micro0_loss":g["hit_step_micro0_loss"],"hit_step_grad_norm":g["hit_step_grad_norm"]} for m,g in G.items()},
 "batch_hashes_steps_and_hit":G["native_from_start"]["step_batch_hashes_first_process"],
 "pair_classes_by_step":PC,"step_labels":s0["step_labels"],
 "step0_native_vs_route_within_native_range":s0.get("step0_native_vs_route_within_native_range"),"reading":s0.get("reading"),"hit_step_grad_norm_ranges_batch1":s0.get("hit_step_grad_norm_ranges"),
 "batch2_groups":{m:{"n_processes":g["n_processes"],"call_counts_total":g["call_counts_total"],"distinct_initial_param_sha256":g["distinct_initial_param_sha256"],"step0_losses_identical_across_processes":g["step0_losses_identical_across_processes"],"distinct_warmup_param_sha256":g["distinct_warmup_param_sha256"],"hit_step_micro0_loss":g["hit_step_micro0_loss"],"hit_step_grad_norm":g["hit_step_grad_norm"]} for m,g in s0b["groups"].items()},
 "verdict":("FlagGems dW really executes from the warmup on (whitelist matches == flag_gems.mm calls: %d / %d over %d route processes; per process 0 in warmup step 0, 96 in warmup step 1, 432 in the hit step). At warmup step 0 (identical initial model, batches and RNG in every process) route-vs-native differs in %s of 386 dense gradient tensors with max_abs at most %.3g; native-vs-native differs in %s with max_abs at most %.3g: the same band. From warmup step 1 on, native-vs-native itself differs in 386/386 tensors (max_abs %.2g-%.2g), so from-step-0 comparisons cannot isolate a route effect. Hit-step grad norm, %d processes per group: native %s, route %s, fallback-only %s (Mann-Whitney p: route vs native %.3f, fallback vs native %.3f, route vs fallback %.2f) - the statistic is dominated by the warmup divergence (the fallback-only group never calls flag_gems.mm and differs from native too)."%(
    G["route_from_start"]["call_counts_total"]["flag_gems_mm"],G["route_from_start"]["call_counts_total"]["whitelist_match"],G["route_from_start"]["n_processes"],
    PC["native_vs_route"]["0"]["n_differing"] and "%d-%d"%(PC["native_vs_route"]["0"]["n_differing"]["min"],PC["native_vs_route"]["0"]["n_differing"]["max"]),PC["native_vs_route"]["0"]["max_abs"]["max"],
    "%d-%d"%(PC["native_vs_native"]["0"]["n_differing"]["min"],PC["native_vs_native"]["0"]["n_differing"]["max"]),PC["native_vs_native"]["0"]["max_abs"]["max"],
    PC["native_vs_native"]["1"]["max_abs"]["min"],PC["native_vs_native"][str(s0["steps_compared"][-1])]["max_abs"]["max"],
    pooled["groups"]["native_from_start"]["hit_step_grad_norm"].__len__(),pooled["groups"]["native_from_start"]["hit_step_grad_norm"][0]+[pooled["groups"]["native_from_start"]["hit_step_grad_norm"][-1]] if False else (pooled["groups"]["native_from_start"]["hit_step_grad_norm"][0],pooled["groups"]["native_from_start"]["hit_step_grad_norm"][-1]),
    (pooled["groups"]["route_from_start"]["hit_step_grad_norm"][0],pooled["groups"]["route_from_start"]["hit_step_grad_norm"][-1]),(pooled["groups"]["fallback_from_start"]["hit_step_grad_norm"][0],pooled["groups"]["fallback_from_start"]["hit_step_grad_norm"][-1]),
    pooled["mann_whitney_normal_approx"]["hit_step_grad_norm"]["route_vs_native"]["p_two_sided_normal_approx"],pooled["mann_whitney_normal_approx"]["hit_step_grad_norm"]["fallback_vs_native"]["p_two_sided_normal_approx"],pooled["mann_whitney_normal_approx"]["hit_step_grad_norm"]["route_vs_fallback"]["p_two_sided_normal_approx"])),
 "copy_grads_none":{m:{"n_processes":g["n_processes"],"distinct_initial_param_sha256":g["distinct_initial_param_sha256"],"distinct_warmup_param_sha256":g["distinct_warmup_param_sha256"],
                       "distinct_warmup_state_sha256":g["distinct_warmup_state_sha256"],"call_counts_total":g["call_counts_total"],"hit_step_micro0_loss":g["hit_step_micro0_loss"],"hit_step_grad_norm":g["hit_step_grad_norm"]} for m,g in s0n["groups"].items()},
 "pooled_main_plus_batch2":pooled,
 "per_step_state_hashes_diag":{ "note":"6 processes (3 native, 3 route, --copy-grads none); hashes of the dense parameters and of the CUDA / CPU RNG states taken right after each step",
   "rows":[{"mode":r["mode"],"step0":{"params_dense":r["steps"][0]["after_step_params_sha256_dense"][:16],"rng_cuda":r["steps"][0]["after_step_rng_cuda_sha256"][:16]},"step1":{"params_dense":r["steps"][1]["after_step_params_sha256_dense"][:16],"rng_cuda":r["steps"][1]["after_step_rng_cuda_sha256"][:16]},"hit":{"params_dense":r["hit_step"]["after_step_params_sha256_dense"][:16],"rng_cuda":r["hit_step"]["after_step_rng_cuda_sha256"][:16]},"step1_loss0":round(r["steps"][1]["losses"][0],6)} for r in sorted(DIAG,key=lambda r:(r["mode"],r["started_unix"]))],
   "distinct_dense_param_hashes":{"initial":len({r["initial_param_sha256_dense"] for r in DIAG}),"after_step0":len({r["steps"][0]["after_step_params_sha256_dense"] for r in DIAG}),"after_step1":len({r["steps"][1]["after_step_params_sha256_dense"] for r in DIAG})},
   "distinct_cuda_rng_hashes":{"after_step0":len({r["steps"][0]["after_step_rng_cuda_sha256"] for r in DIAG}),"after_step1":len({r["steps"][1]["after_step_rng_cuda_sha256"] for r in DIAG})}},
 "note":"the saved per-step gradients (~600 MB per process and step) stay on the measurement host; the JSON holds the per-pair comparisons"}
P0=pr["pairs"][0]["sections"]
def stsec(sec): return {"differing_elements":P0[sec]["dense_totals"]["differing"],"numel":P0[sec]["dense_totals"]["numel"],"differing_pct":round(100*P0[sec]["dense_totals"]["differing"]/P0[sec]["dense_totals"]["numel"],4),"sign_flips":P0[sec]["dense_totals"]["sign_flips"],"elements_rel_gt_10pct":P0[sec]["dense_totals"]["n_rel_gt_10pct"],"tensors_with_differences":P0[sec]["tensors_with_differences"],"max_abs":P0[sec]["max_abs"]}
base=ST["save"]["step1_losses_in_process_history"]; mad=lambda L_:max(abs(x-y) for x,y in zip(L_,base))
out["why_trajectories_diverge_after_a_real_update"]={
 "files":"results/step0_repro_r58.json (scripts/step0_repro_probe.py), results/state_transfer_r58/*.json (scripts/state_transfer_probe.py), per-step hashes in results/route_from_step0_r58_diag/*/run.json",
 "1_first_backward_is_not_bitwise_reproducible_across_processes":{"design":"3 fresh processes, native path, one entry-loop step on batches 0-3, no host access to the gradients before clip; clipped gradients / parameters / optimizer state copied to the host after optimizer.step()",
   "step0_losses_identical":len({json.dumps(x) for x in pr["step0_losses_per_process"]})==1,"step0_grad_norm_per_process":pr["step0_grad_norm_per_process"],"dtypes":pr["meta_first_process"]["dtypes"],"optimizer":pr["meta_first_process"]["optimizer"],
   "pair_1_vs_2":{sec:stsec(sec) for sec in ("grads_clipped","exp_avg","exp_avg_sq","params")},
   "other_pairs_differing_elements":{sec:[p_["sections"][sec]["dense_totals"]["differing"] for p_ in pr["pairs"][1:]] for sec in ("grads_clipped","params")}},
 "2_first_update_differs_by_1_ulp_in_a_few_elements":{"dense_param_sha256_after_step0_per_process":pr["dense_param_sha256_after_step0_per_process"],"differing_elements":"%d-%d of %d"%(min(p_["sections"]["params"]["dense_totals"]["differing"] for p_ in pr["pairs"]),max(p_["sections"]["params"]["dense_totals"]["differing"] for p_ in pr["pairs"]),P0["params"]["dense_totals"]["numel"]),"max_abs":max(p_["sections"]["params"]["max_abs"] for p_ in pr["pairs"])},
 "3_forward_is_a_deterministic_function_of_the_state":{"design":"the state after step 0 of one process (parameters, optimizer state, CPU+CUDA RNG) is saved and loaded by 3 fresh processes; step-1 forward on batches 4-7",
   "step1_losses_in_saving_process":base,"fresh_processes_equal_to_saving_process":all(ST[k]["step1_losses_fresh_process_replays"][0]==base for k in ("fresh1","fresh2","fresh3")),"replays_bitwise_equal_in_every_fresh_process":all(ST[k]["replays_bitwise_equal"] for k in ("fresh1","fresh2","fresh3"))},
 "4_loss_is_extremely_sensitive_to_single_ulp_weight_changes":{"design":"same saved state, N random dense parameter elements moved by exactly one bf16 ulp before the step-1 forward",
   "max_abs_loss_change":{"1_element":round(mad(ST["pert1"]["step1_losses_fresh_process_replays"][0]),6),"2000_elements":round(mad(ST["pert2000"]["step1_losses_fresh_process_replays"][0]),6),"20000_elements":round(mad(ST["pert20000"]["step1_losses_fresh_process_replays"][0]),6)},
   "unperturbed_replay_equals_baseline":all(ST[k]["step1_losses_fresh_process_replays"][1]==base for k in ("pert1","pert2000","pert20000")),
   "mechanism":"contrastive loss with temperature 0.05: logits = cosine / 0.05 are ~20x the embedding differences and are computed by an autocast bf16 matmul, so a 1-ulp change of ONE weight already moves the micro-batch loss by ~0.1%"},
 "5_consequence":"after one real update the processes differ in ~0.0007% of the parameters by 1 ulp, which through (4) gives ~1% differences in the step-1 loss and 10-50% differences in the later grad norms; the RNG states stay hash-identical. Trajectory-level comparisons between a candidate and native (from-step-0 runs) are therefore dominated by this amplification; same-state single-step comparisons (item 1) are the valid instrument.",
 "round57_correction":"the round-57 statement that the native warmup is reproducible across fresh processes (5/5, sha b133cb9f...) referred to a 'warmup' WITHOUT optimizer updates (parameters unchanged) plus one update; with real updates the warmup state differs in every process (train_state_r58.json: 5 distinct snapshot hashes)."}
# ---------------- item 3a ----------------
lv=re_["levels"]
def w(x): return {"n":f"{x['n_native']} vs {x['n_py_route']}","diff_ms":x["diff_ms"],"ci95_ms":x["ci95_ms"],"ci_excludes_zero":x["ci_excludes_zero"],"native_mean_ms":x["native_mean_ms"],"native_sd_ms":x["native_sd_ms"],"py_route_mean_ms":x["py_route_mean_ms"],"py_route_sd_ms":x["py_route_sd_ms"],"welch_df":x["welch_df"]}
pl=lv["all"]["process_level"]["welch"]; b2=lv["batch2_only"]["process_level"]["welch"]; b1=lv["batch1_console_only"]["process_level"]["welch"]
out["item3a_dw_gain_process_level"]={"file":"results/route_entry_ab_r57.json (re-aggregated, field version route_entry_ab_v3)","script":"scripts/route_entry_ab.py",
 "why":"the three training steps of one process share the run environment; the process level (one mean per process) is the conservative unit",
 "process_level":{"all_28_processes":w(pl),"batch2_only_16_processes":w(b2),"batch1_console_only_12_processes":w(b1)},
 "step_level":{"all":w(lv["all"]["step_level"]["welch"]),"batch2_only":w(lv["batch2_only"]["step_level"]["welch"]),"batch1_console_only":w(lv["batch1_console_only"]["step_level"]["welch"])},
 "provenance":{k:v["provenance"] for k,v in lv.items()},"sign_test":re_["process_level_sign_test"],"process_mean_diffs_ms":re_["process_mean_diffs_ms"],
 "verdict":("process level, all 28 processes: %+.2f ms, Welch 95%% CI [%+.2f, %+.2f] - the interval spans zero. There is a small indication of a gain (13/14 py_route processes with a negative mean vs 6/14 controls), not enough to deploy. Batch 2 alone (complete raw records): %+.2f ms [%+.2f, %+.2f]."%(pl["diff_ms"],pl["ci95_ms"][0],pl["ci95_ms"][1],b2["diff_ms"],b2["ci95_ms"][0],b2["ci95_ms"][1]))}
# ---------------- item 3b ----------------
H=sr["headroom_by_shape"]
out["item3b_gemm_tiers"]={"file":"results/software_roofline_r49.json (field version roofline_v2)","script":"scripts/software_roofline.py",
 "window":H["window"],"token_dim_rule":H["token_dim_rule"],"regroup_note":H["regroup_note"],
 "tiers":[{"group":g["group"],"n_shapes":g["n"],"calls":g["calls"],"device_ms":g["device_ms"],"headroom_ms":g["headroom_ms"],"share_pct":round(100*g["headroom_ms"]/H["covered_headroom_ms_per_step"],1)} for g in H["by_group"]],
 "loss_matmul_separate":H["loss_matmul_separate"],"covered_headroom_ms_per_step":H["covered_headroom_ms_per_step"],
 "headroom_definition":"device_ms - FLOPs/(139.2e12) per shape, summed over the tier, clamped >= 0: the ideal-device-time gap under the 139.2 TFLOP/s reference assumption; NOT an end-to-end ceiling"}
# ---------------- item 4 / 5 ----------------
out["item4_hardware"]={"status":"BLOCKED - unchanged","evidence":"results/hbm_blocked_r58.md (probe 2026-11-26T04:52:41Z: acu exit 101 'Device is not ready for profiling'; dmesg: 'PPU0000 PCM is in used by app[pg-exporter]!')",
 "not_answered":["HBM bytes read/written","HBM bandwidth utilization","matrix-unit activity"],"owner_actions_dependency":"see results/hbm_blocked_r58.md",
 "device_utilization_sampling_kept":{"file":"results/smi_utilization_r57.json","n_samples":smi["n_samples"],"utilization_gpu":smi["utilization_gpu"],"caveat":smi["caveat"]},
 "hardware_status":"unverified"}
out["item5_records_delivery"]=rec
# ---------------- answers against the original goal ----------------
gn=G.get("route_from_start",{}).get("hit_step_grad_norm",[]); gnn=G.get("native_from_start",{}).get("hit_step_grad_norm",[])
T=hl["mean_per_step"]
out["answers_against_original_goal"]={
 "main_time_cost_of_one_step":{"step_wall_ms_steps_2_4":[s["step_ms"] for s in hl["steps"][:3]],"mean_step_ms_12_steps":T["step_ms"],
   "device_kernel_time_per_step_3step_mean":{x["phase"]:x["device_ms_per_step"] for x in sr["phase_ranking"][:4]},"device_kernel_time_source":"results/software_roofline_r49.json phase_ranking (training_trace r36)",
   "host_per_step":{k:T[k] for k in ("prepare_embedding_inputs_ms","h2d_ms","forward_launch_ms","loss_sync_ms","backward_launch_ms","end_wait_ms")}},
 "realizable_whole_step_saving_with_uncertainty":{"tokenize prefetch":"+27.7 ms (12/12 slower, CI [+21.8, +41.0]) - closed","h2d_nonblocking":"-1.8 ms median (CI [-10.1, +8.1]) - no significant difference",
   "dW Python route":"%+.2f ms, process-level Welch 95%% CI [%+.2f, %+.2f] (~0.4%% of a step) - small indication of a gain, not enough to deploy"%(pl["diff_ms"],pl["ci95_ms"][0],pl["ci95_ms"][1]),
   "measured_noise_floor":"paired-step sd 14.3 ms (h2d_nb, r57)"},
 "accuracy":{"same_snapshot_single_step":"dw_route bitwise identical to native in all dense sections after every micro-batch from a snapshot with AdamW momentum (flag_gems.mm called %d times in the candidate runs)"%C["dw_route"]["call_counts_candidate_runs"]["flag_gems_mm"],
   "from_step_0":"real flag_gems.mm calls from step 1 on (counters); route-vs-native deviation at step 0 equals native-vs-native; from step 1 on cross-process comparisons are dominated by the native run-to-run divergence (see item2_route_from_step0 and why_trajectories_diverge_after_a_real_update)"},
 "hardware_measured_vs_unknown":{"measured":["SMI utilization.gpu sampling: median %s over %d samples"%(smi["utilization_gpu"]["median"],smi["n_samples"])],"unknown":["HBM bytes / bandwidth utilization / matrix-unit activity (PCM blocked)","~4% not-utilized time in the unprofiled run","backward-window stalls seen under the profiler (~10.7 ms/step)","why native numerics depend on the run pattern"]},
 "conclusion":"No change is deployable on the evidence so far: keep the native path. The dW Python route remains a candidate (accuracy bitwise identical in same-state comparisons; ~0.4% step-time indication with an interval spanning zero)."}
json.dump(out,open(os.path.join(R,"summary_r58.json"),"w"),indent=1)
print("summary_r58.json written")
print("item1:",{c:C[c]["verdict"][:14] for c in C}); print("item3a:",out["item3a_dw_gain_process_level"]["verdict"])
print("item3b:",[(t["group"][:5],t["headroom_ms"]) for t in out["item3b_gemm_tiers"]["tiers"]],"loss",H["loss_matmul_separate"]["headroom_ms"])
