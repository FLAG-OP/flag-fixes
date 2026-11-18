#!/usr/bin/env python3
"""Smoke: load real BGE-M3 weights on KunlunXIN P800, fwd+bwd one batch."""
import json, sys, time
import torch

sys.path.insert(0, "/workspace/FlagScale")
from omegaconf import OmegaConf
from flagscale.models.retrieval import build_retrieval_model

cfg = OmegaConf.load("/workspace/FlagScale/examples/retrieval/conf/train/bge_m3.yaml")
model, processor = build_retrieval_model("bge_m3", cfg.model, torch.device("cuda"))
model.train()
n_params = sum(p.numel() for p in model.parameters())
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"params={n_params/1e6:.1f}M trainable_tensors={len(trainable)} dtype={next(model.parameters()).dtype}")

texts = ["检索训练冒烟测试：这是一段中文文本样例。" * 3,
         "BGE-M3 是一个多语向量化模型，支持 dense/sparse/colbert 三种检索。" * 2] * 5
tok = model.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
t0 = time.perf_counter()
tok_gpu = {k: v.to("cuda") for k, v in tok.items()}
torch.cuda.synchronize(); t_h2d = (time.perf_counter()-t0)*1000

autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
t0 = time.perf_counter()
with autocast:
    out = model.backbone(**tok_gpu).last_hidden_state[:, 0]  # CLS pooling
    loss = out.float().pow(2).mean()
torch.cuda.synchronize(); t_fwd = (time.perf_counter()-t0)*1000
t0 = time.perf_counter()
loss.backward()
torch.cuda.synchronize(); t_bwd = (time.perf_counter()-t0)*1000
gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
print(f"h2d={t_h2d:.1f}ms fwd={t_fwd:.1f}ms bwd={t_bwd:.1f}ms loss={loss.item():.6f} gnorm={gnorm.item():.3f}")
g = trainable[0].grad
print("grad0", g.shape, g.dtype, "finite:", torch.isfinite(g).all().item())
out_json = {"params_M": round(n_params/1e6,1), "loss": loss.item(), "gnorm": gnorm.item(),
            "h2d_ms": t_h2d, "fwd_ms": t_fwd, "bwd_ms": t_bwd,
            "model_path": cfg.model.model_path}
json.dump(out_json, open("/workspace/evidence/model-smoke.json","w"), indent=1)
print("SMOKE OK")
