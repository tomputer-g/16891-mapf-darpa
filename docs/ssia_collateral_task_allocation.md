# `SSIA_collateral/task_allocation.py`

## Role

Implements SSIA with a collateral exploration bonus.

## Main Behavior

- computes the same basic sequential auction structure as SSIA
- adds collateral reward for expected map revelation
- still coordinates ground agents with CBS

## Refactor Additions

- stores winner and runner-up metadata on tasks
- freezes the effective reward used at assignment time
- compares repaired retained quality against the stored alternative without recomputing collateral during the repair check

## Important Note

The frozen reward choice keeps the heuristic cheap and stable, but it also means the stored comparison is only an approximation once the world has changed.
