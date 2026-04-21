# Changelog

## Current Working Tree

Documentation refresh after the refactor:

- moved markdown documentation under `docs/`
- moved LaTeX report drafts under `reports/`
- expanded [../README.md](../README.md) and added [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- added [before_after.md](before_after.md) with recorded benchmark results
- converted `SSIA`, `SSIA_collateral`, and `SSICA` into importable packages

## `2d147fc` `Add repair-vs-reauction heuristic`

Main behavior change:

- `SSIA`, `SSIA-Collateral`, and `SSICA` now try to repair invalidated paths before triggering full reauction

Key code changes:

- `tasks.py` adds shared `AssignmentSnapshot` metadata
- `SSIA/task_allocation.py` stores winner and runner-up bids and handles invalidated assignments
- `SSIA_collateral/task_allocation.py` does the same with frozen collateral-aware reward
- `SSICA/task_allocation.py` stores fallback metadata only for the active head task
- `SSIA/main.py`, `SSIA_collateral/main.py`, and `SSICA/main.py` now delegate blocked-path handling back to the auctioneer instead of immediately dropping work

Observed benchmark effect:

- naive unchanged
- SSIA changed minimally
- SSIA-Collateral and SSICA changed more, with mixed per-map results

See [before_after.md](before_after.md) for the exact table.

## `0c90b7f` `Snapshot simulator state before refactor`

This commit records the pre-refactor simulator state used for the baseline benchmark.

Use it when you need to:

- compare behavior before the repair heuristic
- recover the earlier control flow
- reproduce the baseline benchmark described in [before_after.md](before_after.md)
