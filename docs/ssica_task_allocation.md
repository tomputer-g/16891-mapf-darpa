# `SSICA/task_allocation.py`

## Role

Implements the queue-based concurrent auction allocator.

## Main Behavior

- bids on tasks while accounting for queued work
- maintains per-agent task queues
- activates tasks as earlier queued tasks complete

## Refactor Additions

- only the active head task carries fallback assignment metadata
- `refresh_active_snapshot(...)` updates the live comparison context when a queued task becomes active
- `handle_invalidated_assignment(...)` decides whether to repair the active task or trigger full reauction
- full reauction preserves only an active dwell task that is already in progress on-target; queued tail work is still cleared

## Why SSICA Is Harder

Later queued tasks depend on the state induced by earlier tasks. That makes fallback comparisons for deep queue entries much less stable than in SSIA. The current design intentionally limits the heuristic to the active head task.
