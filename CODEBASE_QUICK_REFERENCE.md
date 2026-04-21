# Codebase Quick Reference

## What This Repo Does

This project simulates multi-agent exploration on a partially known grid.

- Agents: drones and ground vehicles
- World model: hidden `GroundTruthMap` plus shared `KnownMap`
- Work items: exploration, building investigation, and triage tasks
- Allocation variants: naive greedy, SSIA, SSICA, and collateral-aware SSIA
- Path planning: `CBS` in `planner.py`

## Main Execution Flow

For the naive baseline, the main path is:

1. `main.py` loads a scenario with `load_new_scenario()`.
2. It creates a shared `KnownMap`, a `CBS` planner, and a `NaiveTaskAuctioneer`.
3. Agents observe the world and reveal cells into `KnownMap`.
4. The auctioneer creates tasks from frontiers, revealed objectives, and buildings.
5. Idle agents are assigned tasks.
6. Agents replan, move, observe again, and completed tasks are released.

The allocator actually used by the naive simulation is `NaiveTaskAuctioneer` from `naive_task_allocation.py`, instantiated in [main.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/main.py:128).

## Important Files

- [main.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/main.py:116): naive simulation loop
- [SSIA_main.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/SSIA_main.py:90): SSIA simulation loop
- [naive_task_allocation.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/naive_task_allocation.py:37): greedy baseline allocator actually used by `main.py`
- [tasks.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/tasks.py:24): task classes and a second `TaskAuctioneer` implementation
- [agents.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/agents.py:141): agent state, observation, replanning, stepping
- [planner.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/planner.py:232): CBS planner and per-agent stepping
- [maps.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/maps.py:21): hidden world and shared known map
- [sim_types.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/sim_types.py:13): enums for map state, agent type, status, and events

## Task Model

Task priority is an intrinsic field on `Task`, not something computed by the naive allocator at auction time.

- `Task.__init__` stores `self.priority` in [tasks.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/tasks.py:41)
- `ExplorationTask` default priority is `1.0` in [tasks.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/tasks.py:77)
- `TriageTask` default priority is `2.0` in [tasks.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/tasks.py:94)

In the naive baseline:

- frontier tasks are created as `ExplorationTask((nr, nc))`, so they use priority `1.0`
- revealed objectives are created as `TriageTask(loc)`, so they use priority `2.0`
- building investigations are also `TriageTask(...)`, so they also use priority `2.0`
- confirmed occupied-building triage tasks are `TriageTask(...)`, again priority `2.0`

Those task creation sites are in [naive_task_allocation.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/naive_task_allocation.py:75), [naive_task_allocation.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/naive_task_allocation.py:85), [naive_task_allocation.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/naive_task_allocation.py:109), and [naive_task_allocation.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/naive_task_allocation.py:136).

## How Naive Allocation Scores Agents

The naive allocator loops over pending tasks and chooses the eligible idle agent that maximizes:

`(task.priority, -ManhattanDistance(agent, task.target_loc))`

That scoring tuple is defined in [naive_task_allocation.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/naive_task_allocation.py:205).

Interpretation:

- First key: higher `task.priority` wins
- Tie-breaker: smaller Manhattan distance wins because the code uses negative distance
- Ground-only triage tasks exclude drones before scoring

Because `task.priority` is the same for all agents bidding on the same task, it mostly serves to rank task classes, not agents. The distance term is what decides which eligible agent gets a given task.

One subtle but important detail: the current code does not sort pending tasks by priority before assignment. It iterates `available = self.pending()` in queue order and then runs `max(...)` only over agents for the current task. That means:

- `priority` is constant inside each per-task `max(...)`
- task order is effectively insertion order, not priority order
- in practice, `priority` has little effect on which agent wins a given task, and no direct effect on which task is considered first

See [naive_task_allocation.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/naive_task_allocation.py:185) and [naive_task_allocation.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/naive_task_allocation.py:194).

## Map and Planning Notes

- `GroundTruthMap` is the hidden environment. Agents should not read it directly outside observation logic.
- `KnownMap` starts fully unknown and is updated through agent sensing.
- `Agent.assign_task()` sets an agent to `REPLANNING`, and `Agent.replan()` asks `CBS.plan()` for a path.
- `CBS.step()` advances committed paths one cell at a time and emits `STEP_COMPLETE` or `PATH_BLOCKED`.

Relevant code:

- [maps.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/maps.py:21)
- [maps.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/maps.py:47)
- [agents.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/agents.py:176)
- [agents.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/agents.py:191)
- [planner.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/planner.py:273)
- [planner.py](/home/ranai/MRSD/mapf/16891-mapf-darpa/planner.py:361)

## One Easy Confusion To Avoid

There are two auctioneer implementations in this repo:

- `NaiveTaskAuctioneer` in `naive_task_allocation.py`
- `TaskAuctioneer` in `tasks.py`

For the current naive run path in `main.py`, the first one is the live implementation.
