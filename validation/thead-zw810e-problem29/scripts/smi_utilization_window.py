"""Profiler-free device-utilisation sample of the ORIGINAL entry loop.
Runs host_stage_timeline.py (--loop entry) for --rounds timed steps while sampling
`nvidia-smi --query-gpu=utilization.gpu,utilization.memory` every 100 ms in a
separate process. Nothing is added to the workload. Only samples inside the timed
window (first timed step start .. last timed step end) are reported.
The semantics of utilization.gpu on this device are vendor-defined and undocumented
here: the raw distribution is reported, no derived busy-time is claimed from it.
"""
import argparse, json, os, statistics, subprocess, sys, time
ap=argparse.ArgumentParser()
ap.add_argument("--conf",required=True); ap.add_argument("--rounds",type=int,default=12)
ap.add_argument("--interval-ms",type=int,default=100); ap.add_argument("--out",required=True); ap.add_argument("--timeline-out",required=True)
a=ap.parse_args()
here=os.path.dirname(os.path.abspath(__file__))
raw=a.timeline_out+".smi.csv"
smi_log=open(raw,"w")
sampler=subprocess.Popen(["nvidia-smi","--query-gpu=timestamp,utilization.gpu,utilization.memory","--format=csv,noheader,nounits",
                          f"-lms",str(a.interval_ms),"-i","0"],stdout=smi_log,stderr=subprocess.STDOUT)
time.sleep(1.0)
env=dict(os.environ); env.setdefault("CUDA_VISIBLE_DEVICES","0"); env.setdefault("FLAGSCALE_DEVICE","cuda")
tl=os.path.join(here,"host_stage_timeline.py")
r=subprocess.run([sys.executable,tl,"--conf",a.conf,"--loop","entry","--rounds",str(a.rounds),"--out",a.timeline_out],env=env,
                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
time.sleep(0.5); sampler.terminate(); sampler.wait(timeout=10); smi_log.close()
tlj=json.load(open(a.timeline_out)); t0,t1=tlj["timed_window_unix"]
S=[]
for line in open(raw):
    p=[x.strip() for x in line.split(",")]
    if len(p)<3: continue
    try: ts=time.mktime(time.strptime(p[0],"%Y/%m/%d %H:%M:%S.%f"))
    except ValueError: continue
    if t0<=ts<=t1: S.append((ts-t0,int(p[1]),int(p[2])))
u=[x[1] for x in S]; m=[x[2] for x in S]; u.sort()
def pct(v,q): return v[min(len(v)-1,int(q*len(v)))]
hist={k:sum(1 for x in u if lo<=x<lo+10) for k,(lo,_) in enumerate([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,90),(90,100),(100,101)])}
res={"field_version":"smi_utilization_v1","device":"PPU-ZW810E","tool":"nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv,noheader,nounits -lms %d -i 0"%a.interval_ms,
     "workload":"host_stage_timeline.py --loop entry (ORIGINAL entry loop, no profiler), %d timed steps"%a.rounded if False else "workload",
     "timed_steps":a.rounds,"timed_window_s":round(t1-t0,2),"step_ms":[s["step_ms"] for s in tlj["steps"]],
     "n_samples":len(S),"utilization_gpu":{"min":u[0],"p10":pct(u,0.10),"median":statistics.median(u),"mean":round(sum(u)/len(u),1),"p90":pct(u,0.90),"max":u[-1]},
     "utilization_memory":{"min":min(m),"median":statistics.median(m),"mean":round(sum(m)/len(m),1),"max":max(m)},
     "histogram_utilization_gpu_10pct_bins":{f"{lo}-{min(lo+10,100)}":hist[i] for i,(lo,_) in enumerate([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,90),(90,100),(100,101)])},
     "caveat":"utilization.gpu semantics are vendor-defined and not documented for this device; raw distribution only, not a busy-time measurement",
     "samples":[{"t_s":round(x[0],2),"util_gpu":x[1],"util_mem":x[2]} for x in S]}
res["workload"]="host_stage_timeline.py --loop entry (ORIGINAL entry loop, no profiler), %d timed steps after 2 warmup steps"%a.rounds
json.dump(res,open(a.out,"w"),indent=1)
print("samples",len(S),"| util gpu min/p10/median/mean/p90/max:",u[0],pct(u,.1),statistics.median(u),round(sum(u)/len(u),1),pct(u,.9),u[-1],"| mem median",statistics.median(m),"| hist",hist)
