# `reports/final_report.tex` vs Current Code

This audit tracks the current code in the repo as of the refactored layout:

- `main.py` for the greedy baseline
- `SSIA/` for SSIA
- `SSIA_collateral/` for SSIA-Collateral
- `SSICA/` for SSICA

The list below prioritizes behavior and architecture mismatches over small
constant/value mismatches.

## Highest-Priority Discrepancies

### 1. The report still overstates "auction + CBS" as one unified planner for the whole team

- Code:
  - `planner.py` runs CBS only for the ground-agent subset passed in by the
    auctioneers.
  - Drones bid and replan independently through `plan_path(...)` or
    `CBS.plan(..., drone=True)` without joint collision resolution.
  - `SSIA/task_allocation.py`, `SSIA_collateral/task_allocation.py`, and
    `SSICA/task_allocation.py` still do a separate same-type next-move screen,
    so drone-drone conflicts are not fully ignored at the allocator level.
- Report:
  - The abstract, introduction, preliminaries, and CBS discussion read like
    the system uses one CBS-backed MAPF layer for the whole heterogeneous team,
    and also say drone-drone collisions are not considered.
- Why this matters:
  - The live system is better described as auction-based task allocation plus
    CBS coordination for active ground agents, with drones planned
    independently.

### 2. Building investigation is described as an occupancy-reveal step, but the code uses it as a mandatory dwell gate

- Code:
  - `GroundAgent.observe()` immediately writes `OCCUPIED_BUILDING` vs
    `BUILDING` into `KnownMap` within the ground agent's sensor radius.
  - All three allocators still create a 1-step ground-only investigation task
    for every discovered building by using `TriageTask(..., dwell_steps=1)` and
    setting `_is_investigation = True`.
  - `add_confirmed_building_triage(...)` waits for that investigation task to
    complete before creating the actual occupied-building triage task.
- Report:
  - The task-model text says the investigation step reveals whether the
    building contains an objective and refers to an `InvestTask`.
- Why this matters:
  - In the current code there is no separate `InvestTask`, and investigation is
    not the step that reveals occupancy. It is a workflow gate before
    building-triage task creation.

### 3. Obstacle/path handling is now repair-first and snapshot-based, not direct global reauction

- Code:
  - `handle_invalidated_assignment(...)` in all three auction allocators first
    tries local path repair.
  - The repaired retained assignment is then compared against the stored
    winner/runner-up snapshot on the task.
  - The assignment is kept unless repair fails, the retained score loses to the
    stored runner-up beyond slack, or coordinated ground CBS repair fails.
- Report:
  - Related Work, SSIA conflict-management text, and qualitative analysis still
    describe newly discovered obstacles or path infeasibility as direct global
    reauction triggers, or describe the logic as "A* first, then CBS if A*
    fails."
- Why this matters:
  - This is the biggest high-level behavior change relative to the older
    report narrative.

### 4. Global reauction preservation is narrower than the report says

- Code:
  - `SSIA/task_allocation.py` and `SSIA_collateral/task_allocation.py` preserve
    only incomplete on-target dwelling `TriageTask`s during full reauction.
    That includes investigation tasks because they are encoded as
    `TriageTask`s.
  - `SSICA/task_allocation.py` preserves only the active head dwell task and
    clears the queued tail.
- Report:
  - The SSIA text is broader, reading as if agents that have reached their goal
    locations generally keep their work, or as if reauction either resets or
    preserves all goal-reached work.
- Why this matters:
  - The live exception is specifically "already dwelling on the active task,"
    not a blanket reached-goal rule.

### 5. The report still describes one shared harness and one auction round per step

- Code:
  - `main.py` defers auctioning until after both movement microsteps.
  - `SSIA/main.py`, `SSIA_collateral/main.py`, and `SSICA/main.py` call
    `auctioneer.update(...)` after the initial observation, after each
    observation phase, and again after triage progress.
  - That means the SSIA-family allocators can create tasks, sweep
    collateral completions, repair assignments, and auction multiple times
    inside one outer simulation step.
- Report:
  - The Simulation Loop and Allocator Configurations sections say all methods
    use the same harness and imply a single auction round once per full step.
- Why this matters:
  - This is a structural runtime difference, not just an implementation detail.

### 6. Bid computation and CBS coordination are more decoupled than the report suggests

- Code:
  - SSIA-family bids and repair screens use `agents.plan_path(...)`, which is
    an optimistic single-agent path estimate on the current `KnownMap`.
  - Joint CBS is only run afterward as a coordination pass for currently
    navigating ground agents.
  - The naive baseline is even less uniform: newly assigned agents replan
    individually, and its `_cbs_replan_ground()` hook only touches already
    navigating ground agents when an auction round runs.
- Report:
  - Several sections make it sound like the same CBS-backed planner directly
    supplies the per-task path costs used by the allocators for all methods.
- Why this matters:
  - The live allocation logic uses optimistic independent costs first and only
    then tries to restore coordinated ground motion.

### 7. SSICA queue behavior is only partially described in the report

- Code:
  - Only the active queue head gets a refreshed `assignment_snapshot`.
  - Queued tail tasks keep cached path/cost/reward metadata, but not the same
    repair fallback snapshot semantics.
  - Full reauction clears the queued tail and keeps at most the active on-goal
    dwell task.
- Report:
  - The SSICA section describes queue-aware bidding but does not mention this
    head-vs-tail asymmetry.
- Why this matters:
  - It is important for understanding stale queued work and why SSICA clears
    more aggressively than SSIA during reauction.

## Secondary Discrepancies

### Triage reward constants still do not match the report

- Code:
  - `SSIA/task_allocation.py` uses `TriageTask: 3`.
  - `SSIA_collateral/task_allocation.py` uses `TriageTask: 8`.
  - `SSICA/task_allocation.py` uses `TriageTask: 8`.
- Report:
  - `reports/final_report.tex` still presents `R_{\text{triage}} = 5`.
- Why this matters:
  - Lower priority than the behavioral mismatches, but the paper still presents
    one shared value that the implementations do not share.

### Greedy baseline wording still overstates priority-driven task ordering

- Code:
  - `naive_task_allocation.py` iterates tasks in queue order.
  - `(priority, -distance)` is only used when choosing the agent for the
    current task.
- Report:
  - The qualitative-analysis text still says the greedy baseline assigns tasks
    purely by distance and priority tier.
- Why this matters:
  - Priority affects which agent wins the current task, but queue insertion
    order still controls which task is considered first.

### Visualization caption is outdated

- Code:
  - `visualizer.py` uses per-agent colors for both dashed path lines and task
    stars.
- Report:
  - The figure caption still says planned paths are dashed cyan lines and task
    targets are gold stars.

### Drone-collision language is still too absolute

- Code:
  - CBS excludes drones, but the SSIA-family allocators still check same-type
    next-move conflicts, which includes drone-drone conflicts.
- Report:
  - The agent-model text says drone-drone collisions are not considered.

## Checked Items That Currently Match

- The abstract and results-table completion-step values currently match:
  - `naive_results.txt`
  - `ssia_results.txt`
  - `ssia_collateral_results.txt`
  - `ssica_results.txt`
- The current root results are:
  - Greedy: `72.3`
  - SSIA: `43.7`
  - SSIA-Collateral: `39.7`
  - SSICA: `50.9`
