#!/usr/bin/env python3
"""Compose the real bge_m3 train config (same as run-p800-fsdp2.sh) and save it
for the torchrun invocation, then launch the UNMODIFIED trainer via torchrun.

The compose step mirrors the repo's own P800 script; only the paths and the
flaggems flags are parameterised by env. Output: config file path.
"""
import os
import sys
from hydra import compose, initialize_config_dir

repo = os.environ.get("FS_ROOT", "/workspace/FlagScale")
out = sys.argv[1]
include = os.environ.get("FLAGGEMS_INCLUDE", "")  # comma list; "" = not set

overrides = [
    "train=bge_m3",
    "train.parallel=ddp",
    "train.model.torch_dtype=bf16",
    "train.model.max_length=512",
    "train.model.model_path=/flagos-retrieval-models/bge-m3/bge-m3",
    "train.data.path=/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15",
    "train.data.validation_path=/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15",
    "train.data.test_path=/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15",
    "train.batch_size=10",
    "train.gradient_accumulation_steps=4",
    "train.epochs=1",
    "train.log_every=1",
    "train.data.shuffle_buffer=20000",
    "train.data.length_sort_window=256",
    "train.output_dir=" + out,
    "experiment.exp_dir=" + out,
    "experiment.runner.nproc_per_node=1",
    "experiment.envs.WANDB_MODE=disabled",
    "train.hardware.device_type=cuda",
    "train.hardware.autocast_device=cuda",
    "train.hardware.use_flaggems=" + os.environ.get("USE_FLAGGEMS_CFG", "false"),
]
if include:
    overrides.append("+train.hardware.flaggems_include=[" + ",".join(include.split(",")) + "]")
max_samples = os.environ.get("MAX_SAMPLES")
if max_samples:
    overrides.append("train.data.max_samples=" + max_samples)
with initialize_config_dir(config_dir=repo + "/examples/retrieval/conf", version_base=None):
    cfg = compose(config_name="train", overrides=overrides)
os.makedirs(out, exist_ok=True)
from omegaconf import OmegaConf
OmegaConf.save(cfg, out + "/config.yaml", resolve=True)
print("RUN_DIR=" + out)
print("USE_FLAGGEMS_CFG=" + os.environ.get("USE_FLAGGEMS_CFG", "false"))
