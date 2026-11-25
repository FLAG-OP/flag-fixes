"""Review bqj7-Bw item 4: explain device gaps with the EXISTING host+device swimlanes
(no PCM needed). Per step: window per MICRO-BATCH via CPU markers (forward_m0..m3,
backward_m0..m3), include device events crossing the boundary, clip then union.
Then find the largest device-idle gaps and reconcile them against CPU ops,
runtime launches, syncs and correlation ids available in the trace."""
import gzip, json, collections, bisect
R="results/"
trace=json.load(gzip.open(R+"training_trace_r36_trace.json.gz","rt")); evs=trace["traceEvents"]
markers=sorted([e for e in evs if e.get("name","").startswith("rf_s")],key=lambda e:e["ts"])
act=[e for e in evs if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
act.sort(key=lambda e:e["ts"])
first={}
for e in markers:
    st=e["name"].split("_")[1]; first.setdefault(st,e["ts"])
order=sorted(first.items(),key=lambda kv:kv[1]); allend=max(e["ts"]+e.get("dur",0) for e in act)
cpu_ops=[e for e in evs if e.get("cat")=="cpu_op"]
rt=[e for e in evs if e.get("cat") in ("cuda_runtime","cuda_driver")]
rt.sort(key=lambda e:e["ts"]); cpu_ops.sort(key=lambda e:e["ts"])
flows=[e for e in evs if e.get("cat")=="ac2g" and e.get("ph")=="s"]
corr2launch={}
for f in flows: corr2launch.setdefault(f["id"],f["ts"])
def union(iv):
    u=0.0; cs=ce=None
    for a,b in sorted(iv):
        if cs is None or a>ce:
            if cs is not None: u+=ce-cs
            cs,ce=a,b
        else: ce=max(ce,b)
    if cs is not None: u+=ce-cs
    return u
def gaps_in(t0,t1,acts):
    """idle intervals inside [t0,t1) not covered by any clipped device interval"""
    iv=sorted((max(e["ts"],t0),min(e["ts"]+e.get("dur",0),t1)) for e in acts if e["ts"]<t1 and e["ts"]+e.get("dur",0)>t0)
    out=[]; cur=t0
    for a,b in iv:
        if a>cur: out.append((cur,a))
        cur=max(cur,b)
    if cur<t1: out.append((cur,t1))
    return out
res={"method":"window per micro-batch from CPU rf markers (forward_m{i}/backward_m{i}); device events crossing a boundary are CLIPPED into it; union of clipped intervals; gaps = uncovered time inside the window. Host reconciliation uses the trace's own cpu_op / cuda_runtime / sync events and ac2g correlation ids. No PCM.",
     "field_version":"gap_v1","time_unit":"ms","steps":{}}
for i,(st,t0) in enumerate(order):
    t1=order[i+1][1] if i+1<len(order) else allend
    step_act=[e for e in act if e["ts"]<t1 and e["ts"]+e.get("dur",0)>t0]
    pm=[e for e in markers if e["name"].split("_")[1]==st]
    micro={}
    for e in pm:
        nm=e["name"]; a=e["ts"]; b=e["ts"]+e.get("dur",0)
        inwin=[x for x in step_act if x["ts"]<b and x["ts"]+x.get("dur",0)>a]
        bu=union([(max(x["ts"],a),min(x["ts"]+x.get("dur",0),b)) for x in inwin])
        g=gaps_in(a,b,inwin)
        micro[nm]={"window_ms":round((b-a)/1000,2),"busy_union_ms":round(bu/1000,2),"idle_ms":round((b-a-bu)/1000,2),
                   "n_gaps":len(g),"largest_gap_ms":round(max((y-x for x,y in g),default=0)/1000,2)}
    # whole-step: biggest idle gaps with host reconciliation
    sg=[(x,y) for x,y in gaps_in(t0,t1,step_act) if (y-x)>=2000]      # >= 2 ms
    sg.sort(key=lambda g:g[0]-g[1])
    top=[]
    for (ga,gb) in sg[:6]:
        dur=(gb-ga)/1000
        # host events inside the gap
        cin=[e for e in cpu_ops if ga<=e["ts"]<=gb]
        rin=[e for e in rt if ga<=e["ts"]<=gb]
        # events just BEFORE the gap (what the host was doing) and the kernel right AFTER
        before=[e for e in cpu_ops if e["ts"]<ga and e["ts"]+e.get("dur",0)>ga-15000]  # within 15 ms before
        after_k=[e for e in step_act if e["ts"]>=gb][:1]
        after_corr=after_k[0].get("args",{}).get("correlation") if after_k else None
        launch_ts=corr2launch.get(after_corr)
        top.append({"gap_start_us":round(ga,1),"gap_ms":round(dur,3),
                    "cpu_ops_in_gap":dict(collections.Counter(e["name"] for e in cin).most_common(4)),
                    "runtime_in_gap":dict(collections.Counter(e["name"] for e in rin).most_common(4)),
                    "host_ops_ending_before_gap":dict(collections.Counter(e["name"] for e in before).most_common(3)),
                    "next_device_kernel":(after_k[0]["name"][:60] if after_k else None),
                    "next_kernel_launch_lead_ms":(round((after_k[0]["ts"]-launch_ts)/1000,3) if (after_k and launch_ts) else None),
                    "attribution":("HOST-SIDE: cpu ops / runtime calls fill the gap -> submission gap (host-bound)" if (cin or rin) else "NO host events in gap -> dependency/stream wait or unattributed (UNKNOWN)")})
    res["steps"][st]={"micro_windows":micro,"n_idle_gaps_ge2ms":len(sg),"idle_in_gaps_ge2ms":round(sum(y-x for x,y in sg)/1000,2),
                      "top_gaps":top}
json.dump(res,open(R+"gap_attribution_r51.json","w"),indent=1)
for st,v in res["steps"].items():
    print(f"\n=== {st}: gaps>=2ms n={v['n_idle_gaps_ge2ms']} total {v['idle_in_gaps_ge2ms']} ms ===")
    for nm,w in v["micro_windows"].items(): print(f"  {nm:<18} win {w['window_ms']:>8} busy {w['busy_union_ms']:>8} idle {w['idle_ms']:>7} largest {w['largest_gap_ms']:>6}")
    for g in v["top_gaps"][:3]: print(f"  GAP {g['gap_ms']:>6} ms @ {g['gap_start_us']} | {g['attribution'][:70]} | next {g['next_device_kernel']}")
