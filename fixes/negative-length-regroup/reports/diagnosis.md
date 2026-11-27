# Diagnosis

The 2026-11-26 PR8 review asks for a same-batch negative regroup experiment.
The current BGE-M3 pipeline already length-sorts rows before tokenization, but
the hard-negative slots are still encoded independently. Therefore, a long
candidate in one slot can dominate that slot's padded length even when the other
two slots contain much shorter candidates.

The candidate flattens all hard negatives in one micro-batch, uses the real
tokenized `attention_mask` length for a stable sort, and splits them back into
the original number and size of groups. It then scatters each encoded candidate
back to its original index. This keeps the anchor/positive/negative association
and contrastive labels unchanged while only changing the group boundaries used
for the forward calls.

The implementation deliberately falls back to the original path when the adapter
does not expose tensor tokenization features or when negative lists are not
rectangular. This keeps other retrieval model adapters and future input formats
safe.
