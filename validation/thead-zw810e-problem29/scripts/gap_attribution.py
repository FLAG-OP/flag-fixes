"""Review BqnWy6w item 3: gap attribution with CPU-only markers and explicit
covered-vs-UNKNOWN accounting.
- CPU windows: ONLY cat=='user_annotation' rf_s markers (GPU-side markers of the
  same name are kept separately; the old script let them overwrite the CPU ones
  and reported backward windows as 0 ms).
- Device events crossing a window boundary are clipped into it; union of clips.
- For each gap >= 2 ms: CPU op / runtime / sync intervals are UNIONED (not
  counted) -> covered_ms, remainder = UNKNOWN.
- Explicit time markers for tokenize / prepare_embedding_inputs / H2D are taken
  from the trace's own cpu_op (tokenize op names) and gpu_memcpy events; a gap is
  attributed to a category only if that category's interval INTERSECTS the gap."""
import gzip, json, collections
R="results/"
evs=json.load(gzip.open(R+"training_trace_r36_trace.json.gz","rt"))["traceEvents"]
CPU=[e for e in evs if e.get("cat")=="user_annotation" and e.get("name","").startswith("rf_s")]
GPU=[e for e in evs if e.get("cat")=="gpu_user_annotation" and e.get("name","").startswith("rf_s")]
CPU.sort(key=lambda e:e["ts"]); GPU.sort(key=lambda e:e["ts"])
act=sorted([e for e in evs if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")],key=lambda e:e["ts"])
cpu_ops=sorted([e for e in evs if e.get("cat")=="cpu_op"],key=lambda e:e["ts"])
rt=sorted([e for e in evs if e.get("cat") in ("cuda_runtime","cuda_driver")],key=lambda e:e["ts"])
TOKENIZE_OPS=("aten::tokenize","Tokenizer","prepare_embedding","aten::_tokenizer","RustTokenizer","tokenize")
def is_tokenize(e):
    n=e.get("name","")
    return any(t in n for t in TOKENIZE_OPS) or "tokenizer" in n.lower()
def union(iv):
    u=0.0; cs=ce=None
    for a,b in sorted(iv):
        if cs is None or a>ce:
            if cs is not None: u+=ce-cs
            cs,ce=a,b
        else: ce=max(ce,b)
    if cs is not None: u+=ce-cs
    return u
def clip_union(items,t0,t1):
    return union([(max(e["ts"],t0),min(e["ts"]+e.get("dur",0),t1)) for e in items if e["ts"]<t1 and e["ts"]+e.get("dur",0)>t0])
def gaps(t0,t1,items):
    iv=sorted((max(e["ts"],t0),min(e["ts"]+e.get("dur",0),t1)) for e in items if e["ts"]<t1 and e["ts"]+e.get("dur",0)>t0)
    out=[]; cur=t0
    for a,b in iv:
        if a>cur: out.append((cur,a))
        cur=max(cur,b)
    if cur<t1: out.append((cur,t1))
    return out
first={}
for e in CPU:
    st=e["name"].split("_")[1]; first.setdefault(st,e["ts"])
order=sorted(first.items(),key=lambda kv:kv[1]); allend=max(e["ts"]+e.get("dur",0) for e in act)
res={"field_version":"gap_v2","time_unit":"ms",
 "method":"CPU windows from cat==user_annotation only (GPU markers kept separately); clipped union of kernel/memcpy/memset; per gap: CPU-op ∪ runtime ∪ sync intervals -> covered_ms, remainder UNKNOWN; tokenize attribution only when a tokenize cpu_op interval INTERSECTS the gap",
 "fix_note":"gap_v1 (round-52) mixed user_annotation and gpu_user_annotation by name; GPU markers overwrote CPU ones and backward windows showed 0 ms (real CPU value e.g. rf_s2_backward_m0 = 317.08 ms). Superseded.",
 "steps":{}}
for i,(st,t0) in enumerate(order):
    t1=order[i+1][1] if i+1<len(order) else allend
    win_act=[e for e in act if e["ts"]<t1 and e["ts"]+e.get("dur",0)>t0]
    win_cpu=[e for e in CPU if e["name"].split("_")[1]==st]
    micro={}
    for e in win_cpu:
        a=e["ts"]; b=e["ts"]+e.get("dur",0)
        bu=clip_union(win_act,a,b)
        g=gaps(a,b,win_act)
        micro[e["name"]]={"cpu_marker_ms":round((b-a)/1000,2),"device_busy_union_ms":round(bu/1000,2),
                          "idle_ms":round((b-a-bu)/1000,2),"n_gaps":len(g),"largest_gap_ms":round(max((y-x for x,y in g),default=0)/1000,2)}
    gpu_side={e["name"]:round(e.get("dur",0)/1000,2) for e in GPU if e["name"].split("_")[1]==st}
    sg=sorted([(x,y) for x,y in gaps(t0,t1,win_act) if (y-x)>=2000],key=lambda g:g[0]-g[1])
    top=[]
    for ga,gb in sg[:6]:
        dur=(gb-ga)/1000
        cin=[e for e in cpu_ops if e["ts"]<gb and e["ts"]+e.get("dur",0)>ga]
        rin=[e for e in rt if e["ts"]<gb and e["ts"]+e.get("dur",0)>ga]
        sync=[e for e in rin if "Synchronize" in e.get("name","") or "synchronize" in e.get("name","")]
        tokn=[e for e in cin if is_tokenize(e)]
        h2d=[e for e in win_act if e.get("cat")=="gpu_memcpy" and "HtoD" in e.get("name","") and e["ts"]<gb and e["ts"]+e.get("dur",0)>ga]
        cov=union([(e["ts"],e["ts"]+e.get("dur",0)) for e in cin+rin])
        cov_c=min(cov,(gb-ga))
        tok_cov=min(union([(e["ts"],e["ts"]+e.get("dur",0)) for e in tokn]),(gb-ga)) if tokn else 0.0
        top.append({"gap_start_us":round(ga,1),"gap_ms":round(dur,3),
                    "cpu_ops_in_gap":dict(collections.Counter(e["name"] for e in cin).most_common(4)),
                    "runtime_in_gap":dict(collections.Counter(e["name"] for e in rin).most_common(4)),
                    "n_sync_in_gap":len(sync),
                    "host_event_covered_ms":round(cov_c/1000,3),
                    "tokenize_covered_ms":round(tok_cov/1000,3),
                    "h2d_memcpys_intersecting_gap":len(h2d),
                    "unknown_ms":round(dur-cov_c/1000,3),
                    "attribution":{"tokenize":round(tok_cov/1000,3) if tokn else 0.0,
                                   "other_host":round((cov_c-tok_cov)/1000,3),
                                   "UNKNOWN":round(dur-cov_c/1000,3)}})
    res["steps"][st]={"cpu_micro_windows":micro,"gpu_side_marker_ms":gpu_side,
                      "n_gaps_ge2ms":len(sg),"idle_in_gaps_ge2ms":round(sum(y-x for x,y in sg)/1000,2),
                      "top_gaps":top}
json.dump(res,open(R+"gap_attribution_r53.json","w"),indent=1)
for st,v in res["steps"].items():
    print(f"\n=== {st}: gaps>=2ms n={v['n_gaps_ge2ms']} total {v['idle_in_gaps_ge2ms']} ms ===")
    for nm,w in v["cpu_micro_windows"].items(): print(f"  {nm:<22} cpu {w['cpu_marker_ms']:>7} busy {w['device_busy_union_ms']:>7} idle {w['idle_ms']:>6} largest {w['largest_gap_ms']:>5}")
    for g in v["top_gaps"][:3]: print(f"  GAP {g['gap_ms']:>6} ms | host covered {g['host_event_covered_ms']} (tokenize {g['tokenize_covered_ms']}) | UNKNOWN {g['unknown_ms']} | next H2D? {g['h2d_memcpys_intersecting_gap']}")
