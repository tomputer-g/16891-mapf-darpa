# `tasks.py`

## Role

Defines the task model used across all allocators.

## Main Types

- `Task`: base task object with target location, progress, completion state, priority, and assignment fields
- `ExplorationTask`: frontier-style movement task
- `TriageTask`: dwell-based task for investigation or triage work
- `AssignmentSnapshot`: winner and runner-up metadata used by the repair-vs-reauction heuristic
- `TaskAuctioneer`: older generic auctioneer implementation that is not the live naive baseline path

## Important Notes

- Task priority is intrinsic to the task object.
- The refactor branch adds `assignment_snapshot` to tasks so allocators can keep local fallback context when a path becomes invalid.
- Clearing or replacing tasks now needs to clear stale snapshot metadata as well.

## Common Pitfall

There are two auctioneer concepts in the repo:

- `NaiveTaskAuctioneer` in `naive_task_allocation.py`
- `TaskAuctioneer` in `tasks.py`

For the active naive simulation path, the former is the one that matters.
