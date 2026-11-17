#!/usr/bin/env python3
"""Short diagnostic run on the REAL BGE-M3 entry: record every GELU call's
input/grad_output shape, dtype, stride and contiguity, plus the autocast state
and which kernel family serves the call (vendor vs FlagGems), so later
microbenches can use the actual tensors (review round-1: intermediate dim is
expected to be 4096, not 1024).

Diagnostics only -- no timing here. One process, 2 optimizer steps.
"""
import json, os, sys

import torch

sys.path.insert(0, "/workspace/FlagScale")
os.environ.setdefault("FLAGSCALE_DEVICE", "cuda")
os.environ.setdefault("GEMS_VENDOR", "kunlunxin")

from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
from flagscale.models.retrieval import build_retrieval_model

_records = []

_orig_gelu = torch.nn.functional.gelu

def _patched_gelu(x, approximate="none"):
    a = torch.is_autocast_enabled()
    _records.append({
        "shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", ""),
        "contiguous": x.is_contiguous(),
        "stride": list(x.stride())[:4],
        "autocast": bool(a),
        "approximate": str(approximate),
    })
    try:
        return _orig_gelu(x, approximate)
    except TypeError:
        return _orig_gelu(x)

_rank, _world, dev, platform = init_distributed(OmegaConf.create({"device_type": "cuda"}))
cfg = OmegaConf.load("/workspace/FlagScale/examples/retrieval/conf/train/bge_m3.yaml")
model, processor = build_retrieval_model("bge_m3", cfg.model, dev)
# patch AFTER model construction: GELUActivation binds self.act = nn.functional.gelu
# at __init__, so patch each module instance's bound function.
import torch.nn as nn  # noqa: E402
_patched = 0
for name, mod in model.named_modules():
    if hasattr(mod, "act") and mod.act is nn.functional.gelu:
        mod.act = _patched_gelu
        _patched += 1
print(f"[inventory] patched gelu modules: {_patched}", flush=True)
model.train()
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = build_optimizer(cfg.optimizer, trainable, allow_adafactor=False)
loaders = build_split_loaders(str(cfg.task), cfg.data, int(cfg.batch_size), pin_memory=True, seed=42)
stream = iter(loaders[0])
dtype_name = str(cfg.model.get("torch_dtype", "bf16"))

# shapes inside backbone modules (confirm intermediate size actually used)
ffn = None
for name, mod in model.named_modules():
    if "intermediate" in name and hasattr(mod, "weight"):
        ffn = (name, tuple(mod.weight.shape), str(mod.weight.dtype))
        break
cls = [ (n, type(m).__name__) for n, m in model.named_modules() if type(m).__name__.endswith("XLMRobertaIntermediate")][:2]

for step in range(6):
    optimizer.zero_grad(set_to_none=True)
    for _ in range(int(cfg.gradient_accumulation_steps)):
        batch = next(stream)
        with autocast_context(dev, dtype_name):
            loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
        (loss / int(cfg.gradient_accumulation_steps)).backward()
    optimizer.step()

uniq = {}
for r in _records:
    key = (tuple(r["shape"]), r["dtype"], r["contiguous"], r["autocast"])
    uniq[key] = uniq.get(key, 0) + 1

out = {
    "ffn_module": ffn,
    "intermediate_modules": cls,
    "total_gelu_calls": len(_records),
    "unique_shapes": [
        {"shape": list(k[0]), "dtype": k[1], "contiguous": k[2], "autocast": k[3], "calls": v}
        for k, v in sorted(uniq.items(), key=lambda kv: -kv[1])
    ],
    "autocast_enabled_any": any(r["autocast"] for r in _records),
    "approximate_values": sorted({r["approximate"] for r in _records}),
}
json.dump(out, open("/workspace/evidence/gelu_call_inventory.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("ffn_module", "intermediate_modules", "total_gelu_calls", "autocast_enabled_any", "approximate_values")}, indent=1))
print("unique shapes:")
for u in out["unique_shapes"][:8]:
    print(" ", u)
