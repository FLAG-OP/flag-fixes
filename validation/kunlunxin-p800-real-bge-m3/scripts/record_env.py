#!/usr/bin/env python3
"""Record the software/hardware environment for the evidence package."""
import json, subprocess, sys, importlib.metadata as md

def ver(pkg):
    try: return md.version(pkg)
    except Exception: return None

env = {}
env["python"] = sys.version.split()[0]
env["torch"] = ver("torch")
env["flag_gems_site"] = ver("flag_gems")
env["transformers"] = ver("transformers")
env["datasets"] = ver("datasets")
env["triton"] = ver("triton")
env["flagcx"] = ver("flagcx")
for k in ("GEMS_VENDOR","KLX_USE_AUTOTUNE","XPU_ENABLE_PROFILER_TRACING","XMLIR_MATMUL_FAST_MODE",
          "XMLIR_MATMUL_FAST_TUNE","XMLIR_ENABLE_LINEAR_FC_FUSION","HF_HUB_OFFLINE","TRITON_CACHE_DIR"):
    env["env_"+k] = __import__("os").environ.get(k)

import torch
env["torch_cuda_available"] = torch.cuda.is_available()
env["device_count"] = torch.cuda.device_count()
env["device_name"] = torch.cuda.get_device_name(0)
env["device_capability"] = torch.cuda.get_device_capability(0)
env["device_total_mem_GB"] = round(torch.cuda.get_device_properties(0).total_memory/1e9, 2)

try:
    env["xpu_smi"] = subprocess.check_output(
        "xpu-smi 2>&1 | head -12", shell=True, text=True)
except Exception as e:
    env["xpu_smi"] = str(e)
try:
    env["uname"] = subprocess.check_output("uname -a", shell=True, text=True).strip()
except Exception as e:
    env["uname"] = str(e)

json.dump(env, open("env_record.json","w"), indent=1)
print(json.dumps(env, indent=1))
