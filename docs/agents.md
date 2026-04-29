# `agents.py`

## Role

Defines the agent state machine, sensing behavior, task assignment hooks, and per-step motion interface.

## Main Types

- `Event`
- `Agent`
- `DroneAgent`
- `GroundAgent`

## Important State Transitions

- Assigning a task moves an agent to `REPLANNING`.
- Successful planning produces a committed path.
- Triage-style work can advance progress without physical movement.
- Planner stepping emits events such as `STEP_COMPLETE` and `PATH_BLOCKED`.
- `PATH_BLOCKED` now moves the agent into `REPLANNING` so the allocator can attempt repair before deciding on reauction.
- An agent can remain stationary on `task.target_loc` while `TriageTask.progress` continues to increase.

## Why This File Matters

If control flow seems wrong, the root cause is often here rather than in the auctioneer. The auctioneer decides ownership. The agent and planner decide whether a path actually exists and whether progress is happening.
