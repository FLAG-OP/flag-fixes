# Negative token-length regrouping

- ID: `negative-length-regroup`
- Status: draft (not deployable)
- Owner: unassigned
- Upstream: [FLAG-OP/flag-fixes#7](https://github.com/FLAG-OP/flag-fixes/issues/7)
- Affected layer: FlagScale text-retrieval hard-negative encoding
- Backends: Thead ZW810E candidate; no hardware verification yet

Within one micro-batch, BGE-M3 currently encodes each named negative slot as a
separate fixed-size group. Long-tail candidates can therefore make every group
pay for its longest member. This fix adds an opt-in mode that tokenizes all
hard negatives once, sorts them by their real `attention_mask` length, and puts
them back into fixed-size groups before encoding. The resulting embeddings are
scattered back to the original `(query, negative slot)` positions, so the
contrastive loss sees the same pairs. The default path is unchanged.

Enable it with `FLAGSCALE_NEGATIVE_LENGTH_REGROUP=1`. Keep it unset (or set it
to `0`) for the original-order A arm. This change only changes how candidates
are grouped for forward calls; it does not alter the dataset order, loss labels,
optimizer, or gradient-accumulation boundary.

The device A/B is not complete. Do not enable this in production and do not
treat the local regression as a performance validation.
