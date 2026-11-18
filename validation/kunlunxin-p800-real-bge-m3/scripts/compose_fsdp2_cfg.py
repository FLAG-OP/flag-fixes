#!/usr/bin/env python3
"""Compose the real bge_m3 train config for a 2-card FSDP2 run (issue #9 E /
problem.md §5.5 multi-card boundary). Mirrors run-p800-fsdp2.sh overrides."""
import os
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

repo = os.environ.get("FS_ROOT", "/workspace/FlagScale")
out = sys.argv[1]
include = os.environ.get("FLAGGEMS_INCLUDE", "")

overrides = [
    "train=bge_m3",
    "train.parallel=fsdp2",
    "train.master_dtype=" + os.environ.get("MASTER_DTYPE", "bf16"),
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
    "train.data.max_samples=" + os.environ.get("MAX_SAMPLES", "600"),
    "train.output_dir=" + out,
    "experiment.exp_dir=" + out,
    "experiment.runner.nproc_per_node=" + os.environ.get("NPROC", "2"),
    "experiment.envs.WANDB_MODE=disabled",
    "train.hardware.device_type=cuda",
    "train.hardware.autocast_device=cuda",
    "train.hardware.use_flaggems=" + os.environ.get("USE_FLAGGEMS_CFG", "false"),
    "train.save_every_epoch=false",
]
if include:
    overrides.append("+train.hardware.flaggems_include=[" + ",".join(include.split(",")) + "]")
with initialize_config_dir(config_dir=repo + "/examples/retrieval/conf", version_base=None):
    cfg = compose(config_name="train", overrides=overrides)
os.makedirs(out, exist_ok=True)
OmegaConf.save(cfg, out + "/config.yaml", resolve=True)
print("RUN_DIR=" + out)
