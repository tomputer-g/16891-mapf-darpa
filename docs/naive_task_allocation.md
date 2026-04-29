# `naive_task_allocation.py`

## Role

Implements the greedy baseline allocator used by `main.py`.

## Main Behavior

- creates exploration and triage tasks from the known map
- filters eligible agents by task type
- assigns queued pending tasks greedily to idle agents
- does not use winner/runner-up repair snapshots
- does not implement SSIA-style repair-vs-reauction logic

## Scoring Rule

The key scoring rule is effectively:

`(task.priority, -ManhattanDistance(agent, task.target_loc))`

## Important Caveat

Pending tasks are processed in queue order. The allocator does not globally sort pending tasks by priority before assignment. That means priority affects task classes, but not as strongly as a report reader might assume from the scoring tuple alone.

## Blocked-Path Note

When a committed path becomes invalid in the naive baseline, `main.py` simply releases the task and lets later auction rounds pick it up again. That path does not attempt local repair against a stored runner-up alternative.
