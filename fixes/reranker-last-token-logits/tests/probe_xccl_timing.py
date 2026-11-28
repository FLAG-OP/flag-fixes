#!/usr/bin/env python3
"""Platform probe (diagnostic only): is the XCCL all-gather hang sensitive to host timing?

Exploratory characterisation of platform blocker F-P800-1 for the runtime layer.
It does not judge the fix.  Launch with torchrun on two cards:

    PYTHONPATH=<FlagScale tree> SKEW=aligned python -m torch.distributed.run \
        --standalone --nproc_per_node=2 tests/probe_xccl_timing.py

A 2-layer Qwen3 (hidden 4096, real vocabulary) is sharded with the real
``parallel_utils.apply_fsdp2`` and run for ``SKEW_STEPS`` optimizer steps on a
tiny batch; the host logs every step so a hang's position is visible.  Variant
(``SKEW``):

  plain     no artificial delay
  skew      rank 1 sleeps ``SKEW_DELAY`` s before every forward, rank 0 does not
  aligned   both ranks sleep ``SKEW_DELAY`` s before every forward
  barrier   rank 1 sleeps, then ``dist.barrier()``, then forward

A hang shows as the run never reaching ``CLEAN_EXIT`` (kill it from outside;
the vendor watchdog needs 600 s).  Result of the 2026-11-27 session:
``../results/f-p800-1-timing-probe-2026-11-27.json``.
"""
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reranker_common import REAL_CONFIG, StubTokenizer  # noqa: E402

RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
VARIANT = os.environ.get("SKEW", "plain")
DELAY = float(os.environ.get("SKEW_DELAY", "3"))
STEPS = int(os.environ.get("SKEW_STEPS", "4"))


def log(msg):
    # both ranks log: the interleaving in the output is part of the evidence
    print(f"[rank{RANK} {time.strftime('%T')}] {msg}", flush=True)


torch.cuda.set_device(LOCAL_RANK)
dist.init_process_group(backend="nccl", init_method="env://")
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from flagscale.models.retrieval.qwen3_reranker import DEFAULT_INSTRUCTION, Qwen3RerankerModel  # noqa: E402
from flagscale.train.utils import parallel as pu  # noqa: E402

config = AutoConfig.for_model("qwen3", hidden_size=4096, intermediate_size=12288, num_attention_heads=32,
                              num_key_value_heads=8, head_dim=128, vocab_size=REAL_CONFIG["vocab_size"],
                              max_position_embeddings=40960, rope_theta=1000000.0, rms_norm_eps=1e-6,
                              tie_word_embeddings=False, attention_bias=False, hidden_act="silu")
config.num_hidden_layers = int(os.environ.get("SKEW_LAYERS", "2"))
torch.manual_seed(0)
backbone = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
model = Qwen3RerankerModel(backbone, StubTokenizer(vocab=config.vocab_size), 256, 9693, 2152, False, DEFAULT_INSTRUCTION)
model.enable_gradient_checkpointing()
model = pu.apply_fsdp2(model, device_type="cuda", world_size=2, compute_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, log=log)
model = model.to(device=torch.device("cuda", LOCAL_RANK), dtype=torch.float32)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=1e-5)
device = torch.device("cuda", LOCAL_RANK)
tokens = torch.randint(0, config.vocab_size, (4, 128), device=device)
mask = torch.ones_like(tokens)
log(f"variant={VARIANT} ready")


def pre_forward():
    if VARIANT == "skew" and RANK == 1:
        time.sleep(DELAY)
    elif VARIANT == "aligned":
        time.sleep(DELAY)
    elif VARIANT == "barrier":
        if RANK == 1:
            time.sleep(DELAY)
        log("entering barrier")
        dist.barrier()
        log("barrier done")


for step in range(STEPS):
    opt.zero_grad(set_to_none=True)
    model.train()
    log(f"step {step}: before pre_forward")
    pre_forward()
    log(f"step {step}: issuing forward")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model({"input_ids": tokens, "attention_mask": mask})["scores"].float().sum()
    log(f"step {step}: forward issued, backward")
    out.backward()
    torch.cuda.synchronize()
    log(f"step {step}: backward done")
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    torch.cuda.synchronize()
    log(f"step {step}: optimizer step done")
log("ALL_STEPS_DONE")
dist.barrier()
dist.destroy_process_group()
log("CLEAN_EXIT")
