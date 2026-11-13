"""Probe 3: (a) size-scaling of native vs FlagGems overhead; (b) selective
enable hit/miss evidence via profiler at perf shapes (bf16)."""
import json
import subprocess
import sys

CODE = r'''
import json, sys
import torch
import torch.nn.functional as F
mode = sys.argv[1]
out = {}
if mode != "native":
    import flag_gems
    if mode == "selective":
        flag_gems.only_enable(include=["gelu", "gelu_backward", "gelu_and_mul"])
    else:
        flag_gems.enable()

import statistics
def p50(fn, iters=15, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return round(statistics.median(sorted(s.elapsed_time(e) for s,e in ev)),4)

g=torch.Generator(device="cuda").manual_seed(3)
sizes={"1.2M":1228800,"12.25M":12288000,"98M":98304000}
scale={}
for nm,n in sizes.items():
    x=torch.randn(n,generator=g,device="cuda",dtype=torch.float32)
    scale[nm]={"gelu_fp32":p50(lambda:F.gelu(x)),"add_fp32":p50(lambda:x+x),
               "gelu_bf16":p50(lambda:F.gelu(x.to(torch.bfloat16))),
               "cast_fp16":p50(lambda:x.to(torch.float16))}
out["scaling"]=scale
out["mode"]=mode

# profiler hit evidence at perf shape (bf16 gelu)
xb=torch.randn(12288000,generator=g,device="cuda",dtype=torch.bfloat16)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    F.gelu(xb); torch.cuda.synchronize()
ks=[e.key for e in prof.key_averages() if e.device_type == torch.profiler.DeviceType.CUDA]
out["gelu_bf16_kernels"]=[k for k in ks if "gelu" in k.lower()][:5] or [k for k in ks][:3]
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    torch.max(xb.float(),dim=-1); torch.cuda.synchronize()
ks=[e.key for e in prof.key_averages() if e.device_type == torch.profiler.DeviceType.CUDA]
out["findmax_kernels"]=[k[:80] for k in ks][:4]
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    (xb+xb); torch.cuda.synchronize()
ks=[e.key for e in prof.key_averages() if e.device_type == torch.profiler.DeviceType.CUDA]
out["add_bf16_kernels"]=[k[:80] for k in ks][:4]
print(json.dumps(out))
'''

res = {}
for mode in ("native", "full", "selective"):
    p = subprocess.run(
        [sys.executable, "-c", CODE, mode],
        env={"PYTHONPATH": "/tmp/opencode/FlagGems-base/src", "CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/local/bin:/usr/bin:/bin", "PPU_SDK": "/usr/local/PPU_SDK", "HOME": "/root"},
        capture_output=True, text=True, timeout=900,
    )
    res[mode] = json.loads(p.stdout.strip().splitlines()[-1])

json.dump(res, open("results/probe3_scaling_hits.json", "w"), indent=1)
print(json.dumps(res, indent=1))
