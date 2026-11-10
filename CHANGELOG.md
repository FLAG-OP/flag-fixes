# Changelog

## 0.1.0

- Pin eight existing FLAG-OP FIX repositories as Git submodules under `third_party/`.
- Index nine fix entries with source commits, patch hashes, target snapshots and explicit validation status.
- Add `templates/fix/` and `flag-fixes new` for new fixes maintained in this repository.
- Add 1 local performance fix `gelu-kunlunxin-pow`: replace libdevice `pow(x, 2.0)` with multiplication in the KunlunXIN GELU family (P800/XPU3 microbenchmark 5.8x-16.5x, no accuracy regression; draft, hardware unverified).
- Provide catalog validation, exact source-tree patch planning, explicit apply, status, verification and LIFO rollback.
- Test failure recovery, drift/conflict refusal, idempotence and imported source consistency.
- Add CI for Python 3.10 and 3.12.

Imported fixes remain non-deployable drafts pending adaptation and platform validation. Hardware profiles, validated bundles and offline release assembly are not included in this version. See `docs/source-audit.md` for the initial source audit.
