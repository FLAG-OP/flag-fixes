# Tests

Run the standalone semantic regression:

```bash
python3 tests/test_negative_length_regroup.py
```

For a patched FlagScale checkout, set `PYTHONPATH` to that target tree. The test
falls back to the local source snapshot when `FLAGSCALE_TREE` is not set.
