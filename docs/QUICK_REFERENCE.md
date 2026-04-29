# Quick Reference

## Docs Map

Start here for the short version, then use the detailed references as needed:

- [../README.md](../README.md): setup, runners, and top-level repo map
- [before_after.md](before_after.md): benchmark results for the refactor branch
- [FINAL_REPORT_CODE_DISCREPANCIES.md](FINAL_REPORT_CODE_DISCREPANCIES.md): report/code mismatches grouped by report section
- [README.md](README.md): per-module reference docs
- [CHANGELOG.md](CHANGELOG.md): branch and refactor history

## What This Repo Simulates

This project models partially observed grid exploration with heterogeneous agents.

- Drones move over free space and reveal map structure quickly.
- Ground vehicles handle coordinated navigation and triage work.
- The hidden world lives in `GroundTruthMap`.
- The shared discovered state lives in `KnownMap`.
- Tasks include exploration, building investigation, and triage.

## Live Entry Points

The four main simulation paths are:

- `main.py`: naive greedy baseline
- `SSIA/main.py`: sequential single-item auction
- `SSIA_collateral/main.py`: SSIA with collateral exploration bonus
- `SSICA/main.py`: queue-based concurrent auction

The batch runner scripts are:

- `run_naive.py`
- `run_ssia.py`
- `run_collateral.py`
- `run_ssica.py`

Each runner processes `generated/darpa1.txt` through `generated/darpa7.txt` and writes a root-level `*_results.txt` summary file.

## Current Refactor State

Branch `ranais/refactor` introduces a repair-vs-reauction heuristic for the auction-based allocators.

The current behavior is:

- store winner and runner-up assignment metadata on each active task
- when a committed path is invalidated, try local repair first
- compare repaired retained quality against the stored runner-up with a small slack threshold
- keep the assignment if retained quality is still competitive
- escalate to full reauction when repair fails, coordinated repair fails, or the stored alternative is clearly better
- preserve agents already dwelling on incomplete triage tasks during full reauction instead of resetting that in-progress work

This heuristic exists in:

- `SSIA/task_allocation.py`
- `SSIA_collateral/task_allocation.py`
- `SSICA/task_allocation.py`

It is intended to reduce assignment churn. It is not an optimality proof.

## Allocation Notes

Naive baseline:

- Uses `NaiveTaskAuctioneer` in `naive_task_allocation.py`
- Iterates pending tasks in queue order
- Chooses an eligible idle agent by `(task.priority, -ManhattanDistance)`
- Task priority is intrinsic to the task object, but the queue-order iteration limits how much that priority actually affects assignment order

Auction variants:

- `SSIA`: bid is reward shaped by travel and dwell cost
- `SSIA-Collateral`: same core structure, but with a collateral exploration bonus
- `SSICA`: queue-based look-ahead, but only the active head task now carries repair snapshot metadata

## Planning Notes

- Ground-agent coordination uses `CBS` in `planner.py`.
- Drones plan independently.
- `Agent.assign_task()` moves an agent into `REPLANNING`.
- `planner.CBS.step()` emits `STEP_COMPLETE`, `PATH_BLOCKED`, or completion events as agents advance.
- `PATH_BLOCKED` now moves the agent into `REPLANNING`, not `IDLE`, before the allocator decides whether to repair or reauction.

The important consequence is that `PATH_BLOCKED` is a path-level failure signal, not necessarily an immediate proof that the whole assignment set should be discarded.

Naive baseline note:

- `main.py` still uses a simpler blocked-path policy: release the task and return the agent to `IDLE`.
- The repair-vs-reauction heuristic only exists in `SSIA`, `SSIA_collateral`, and `SSICA`.

## Important Files

- [../agents.py](../agents.py): agent state machine, sensing, replanning, and stepping
- [../planner.py](../planner.py): low-level search and CBS coordination
- [../maps.py](../maps.py): hidden and known map representations
- [../sim_types.py](../sim_types.py): enums for observation, agent status, and event types
- [../tasks.py](../tasks.py): task definitions plus shared `AssignmentSnapshot`
- [../naive_task_allocation.py](../naive_task_allocation.py): greedy baseline allocator
- [../SSIA/task_allocation.py](../SSIA/task_allocation.py): repair-aware SSIA auctioneer
- [../SSIA_collateral/task_allocation.py](../SSIA_collateral/task_allocation.py): repair-aware collateral auctioneer
- [../SSICA/task_allocation.py](../SSICA/task_allocation.py): repair-aware SSICA queue allocator

## Recommended Starting Order

If you are picking this up cold, read in this order:

1. `../README.md`
2. `before_after.md`
3. `QUICK_REFERENCE.md`
4. `tasks.md`
5. `agents.md`
6. `planner.md`
7. the allocator and simulation-loop docs for the variant you care about
