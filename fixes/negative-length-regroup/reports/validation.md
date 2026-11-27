# Validation status

## Completed locally

Command:

```bash
python3 fixes/negative-length-regroup/tests/test_negative_length_regroup.py
```

Observed output:

```text
PASS: default path unchanged; opt-in regrouping preserves loss and reduces tokenization calls (4 -> 3)
```

The test uses a deterministic mock encoder. It verifies that:

- the default path does not enter the regrouping branch;
- the opt-in path flattens the negative slots before tokenization;
- encoded candidates are restored to their original positions;
- the contrastive loss remains unchanged when encoder outputs are independent of
  padding and group composition.

## Not completed

- ZW810E real-entry wall-time A/B.
- Peak device memory measurement.
- Forward/backward/clip+optimizer segment attribution.
- Long-horizon loss, gradient, and updated-parameter comparison.
- Retrieval metric evaluation.

PCM/HBM counters remain separately blocked in PR8. This fix must remain
`deployable=false` and `hardware_status=unverified` until those A/B results are
available and reviewed.

## PPU smoke and microbenchmark

A minimal real-model check ran on PPU-ZW810E with `BAAI/bge-m3` and the T2Ranking
triplet-15 parquet shard. It used three rows and two negative slots:

```text
legacy loss  1.4054532
regroup loss 1.4048704
abs diff     5.83e-4
tokenize/encode calls 4 -> 3
```

The small loss delta is expected because different padding groups change the
floating-point execution; the pair association and labels remain unchanged.

A forward/backward microbenchmark used the same real model and a fixed random
sample of rows. It is not the original FlagScale training entry and does not
include clip+optimizer or multi-step state:

| rows | negatives | legacy median ms | regroup median ms | delta ms |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 3 | 320.84 | 324.23 | +3.38 |
| 30 | 3 | 790.97 | 783.22 | -7.74 |
| 60 | 3 | 1509.88 | 1503.83 | -6.04 |

This is a sensitivity check, not a deployment result. The candidate therefore
remains draft and unverified.

## Original-entry one-step A/B

The original `flagscale/train/train_retrieval.py` entry ran four interleaved
processes per arm on PPU-ZW810E. Each arm used `BAAI/bge-m3`, T2Ranking
triplet-15, `batch_size=10`, `gradient_accumulation_steps=4`, one epoch, and one
optimizer step. Checkpoint output was placed on `/dev/shm` to reduce I/O noise.

| arm | median elapsed | range | train loss | test loss |
| --- | ---: | ---: | ---: | ---: |
| legacy | 3.65 s | 3.6–3.9 s | 1.402771 | 2.243490 |
| regroup | 3.70 s | 3.7–3.7 s | 1.475995 | 2.221029 |

The difference is `+0.05 s`, inside the observed run-to-run variation. This is
only a one-step whole-entry signal and still does not satisfy the multi-process,
multi-step, retrieval-metric acceptance requested in PR8. The candidate remains
`deployable=false` and `hardware_status=unverified`.
