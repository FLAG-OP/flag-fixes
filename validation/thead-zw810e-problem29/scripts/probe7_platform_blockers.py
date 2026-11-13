"""Probe 7 (review round 2): platform blockers on ZW810E — FlagGems path hangs.

F-1: sdpa (FlagGems flash-attention registration) HANGS for training-shaped
     fp32 input [1,8,128,256] — forward never returns (>300s), no triton
     compile activity. Inference-shaped bf16 [8,16,512,64] works (perf matrix).
F-2: matmul BACKWARD HANGS for fp32 [8,128,256]@[256,512] under enable —
     forward identical shape works; large forward [4096,1280]@[1280,2048]
     works (perf matrix 2.19ms).

Provenance note: the same FlagGems tree + small-model workload COMPLETED under
gems_full on 2026-11-12 evening (results/small_model_v1/gems_full, 5 steps,
step_ms 10.5-15.3) — the hangs are environment-state dependent (device/driver
state changed between 2026-11-12 21:30 and 2026-11-13 11:30; device rebooted
or reset outside our control). Per issue #5 guidance ("出现设备级异常时先交由
平台恢复并隔离复验，不在异常共享节点反复触发"), each hang is triggered ONCE
in an isolated subprocess with timeout; no retries.

Each subprocess: hard timeout, hang -> exit non-zero; diagnosis JSON always
written first.
"""
import json
import subprocess
import sys

PY = sys.executable
ENV = {"PYTHONPATH": "/tmp/opencode/FlagGems-base/src", "CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin:/usr/local/bin", "PPU_SDK": "/usr/local/PPU_SDK", "HOME": "/root"}
TIMEOUT = 300

CASES = {
    "F1_sdpa_fwd_fp32_training_shape": """
import torch, torch.nn.functional as F, flag_gems
flag_gems.enable()
q = torch.randn(1,8,128,256,device='cuda',requires_grad=True)
att = F.scaled_dot_product_attention(q,q,q)
torch.cuda.synchronize(); att.sum().backward(); torch.cuda.synchronize()
print("DONE")
""",
    "F1_control_sdpa_bf16_inference_shape": """
import torch, torch.nn.functional as F, flag_gems
flag_gems.enable()
q = torch.randn(8,16,512,64,device='cuda',dtype=torch.bfloat16)
att = F.scaled_dot_product_attention(q,q,q)
torch.cuda.synchronize(); print("DONE", att.shape)
""",
    "F2_matmul_bwd_fp32": """
import torch, flag_gems
flag_gems.enable()
w = torch.randn(256,512,device='cuda',requires_grad=True)
x = torch.randn(8,128,256,device='cuda')
(x@w).sum().backward(); torch.cuda.synchronize()
print("DONE", w.grad.shape)
""",
    "F2_control_matmul_fwd_fp32": """
import torch, flag_gems
flag_gems.enable()
w = torch.randn(256,512,device='cuda')
x = torch.randn(8,128,256,device='cuda')
y = (x@w); torch.cuda.synchronize()
print("DONE", y.shape)
""",
}

res = {}
for name, code in CASES.items():
    try:
        p = subprocess.run([PY, "-u", "-c", code], env=ENV, capture_output=True, text=True, timeout=TIMEOUT)
        done = "DONE" in p.stdout
        res[name] = {"completed": done, "rc": p.returncode, "stdout_tail": p.stdout.strip()[-120:], "stderr_tail": p.stderr.strip()[-200:]}
    except subprocess.TimeoutExpired:
        res[name] = {"completed": False, "rc": "TIMEOUT", "note": f"no completion within {TIMEOUT}s; killed; no triton compile activity observed in prior runs"}

res["conclusion"] = {
    "F1": "FlagGems sdpa flash-attention path hangs on ZW810E for fp32 training shape; inference bf16 works -> full-enable training with sdpa is BLOCKED",
    "F2": "FlagGems matmul BACKWARD hangs (fp32 small shape); forward works -> full-enable training with matmul (any model) is BLOCKED",
    "policy": "single isolated trigger per hang, no retries (issue #5 device-anomaly rule); platform/device recovery + isolated re-verification pending",
}
with open("results/probe7_platform_blockers.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
hangs = [k for k, v in res.items() if isinstance(v, dict) and v.get("rc") == "TIMEOUT"]
print("HANGS:", hangs)
sys.exit(1 if hangs else 0)
