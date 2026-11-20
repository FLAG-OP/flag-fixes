"""Valid/padding token statistics for the r25 capture window (CPU-only,
deterministic: same seed 42 + loader as the device capture). Reproduces the
3 capture steps' batch composition and reports per-group token counts."""
import json
import torch  # noqa: CPU only

from transformers import AutoTokenizer
from omegaconf import OmegaConf
from flagscale.train.datasets.retrieval import build_split_loaders

cfg = OmegaConf.load("/tmp/opencode/FlagScale-ppu/examples/retrieval/conf/run/bge_m3_ppu.yaml")
tok = AutoTokenizer.from_pretrained(str(cfg.model.model_path))
pad_id = tok.pad_token_id
loaders = build_split_loaders(str(cfg.task), cfg.data, int(cfg.batch_size), pin_memory=False, seed=42)
stream = iter(loaders[0])

SKIP, CAPTURE = 2, 3
out = {"pad_token_id": pad_id, "steps": {}}
for step in range(SKIP + CAPTURE):
    micros = []
    for _ in range(int(cfg.gradient_accumulation_steps)):
        batch = next(stream)
        groups = {}
        for key in sorted(batch.keys()):
            enc = tok(batch[key], padding=True, truncation=True,
                      max_length=int(cfg.model.get("max_length", 8192)),
                      return_tensors=None)
            # round-28 fix: valid tokens from BINARY attention_mask (previous
            # sum(len(ids)) counted post-padding positions -> always 0% padding)
            mask = enc["attention_mask"]
            total = sum(len(m) for m in mask)
            valid = sum(sum(1 for x in m if x == 1) for m in mask)
            max_len = max(len(m) for m in mask)
            n_seq = len(mask)
            groups[key] = {"sequences": n_seq, "max_len": max_len,
                           "valid_tokens": valid, "padded_positions": total,
                           "padding_ratio": round(1 - valid / total, 4)}
        micros.append(groups)
    if step >= SKIP:
        out["steps"][f"capture_step_{step - SKIP}"] = micros
tot = {}
for name, micros in out["steps"].items():
    valid = sum(g["valid_tokens"] for m in micros for g in m.values())
    padded = sum(g["padded_positions"] for m in micros for g in m.values())
    tot[name] = {"valid_tokens": valid, "padded_positions": padded,
                 "padding_ratio": round(1 - valid / padded, 4)}
out["step_totals"] = tot
json.dump(out, open("/tmp/opencode/zw810e-problem29/results/token_stats_r25.json", "w"), indent=1)
for name, t in tot.items():
    print(f"{name}: valid={t['valid_tokens']} padded={t['padded_positions']} ratio={t['padding_ratio']}")
