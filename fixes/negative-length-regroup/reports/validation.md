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
