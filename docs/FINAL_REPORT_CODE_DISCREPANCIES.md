# `reports/final_report.tex` vs Code Discrepancies

This version is organized by report section. Within each item, I list `Code`
first and `Report` second.

I only kept current, high-confidence mismatches after the latest
`reports/final_report.tex` edits. The old "omnidirectional motion" issue is no
longer listed because the report now correctly describes 4-connected movement.

Package note: the runtime modules were later reorganized into `SSIA/`,
`SSIA_collateral/`, and `SSICA/`. The older flat filenames below are kept only
as historical labels for the discrepancy audit.

## Related Worksk

### Reauction trigger is still overstated

- Code:
  - `SSIA_main.py:80-87` releases only the blocked agent's current task when a path step hits a newly revealed obstacle.
  - `SSIA_task_allocation.py:533-548` and `SSIA_collateral_task_allocation.py:581-596` release only infeasible tasks and then continue with normal auctioning.
  - `SSIA_task_allocation.py:415-445` and `SSIA_collateral_task_allocation.py:463-493` reserve global reauction for same-type next-move conflicts or CBS failure.
- Report:
  - `final_report.tex:75` says the event-triggered global reauction mechanism handles newly discovered obstacles and motion infeasibility.
- Why this matters:
  - In the current SSIA-family code, ordinary obstacle discovery does not trigger a full global reauction.

## Proposed Method

### Task Types / SSIA Reward-Shaped Bid: triage reward still does not match the implementations

- Code:
  - `SSIA_task_allocation.py:34-38` uses `TriageTask: 3`.
  - `SSIA_collateral_task_allocation.py:34-38` uses `TriageTask: 8`.
  - `SSICA_task_allocation.py:30-34` uses `TriageTask: 8`.
- Report:
  - `final_report.tex:124-128` says `R_{\text{triage}} = 5`.
  - `final_report.tex:153-157` again uses `R_{\text{triage}} = 5` in the SSIA bid description.
- Why this matters:
  - The report still presents one shared triage reward value, but the live implementations do not share a single constant.

### SSIA / Global Reauction Trigger: path infeasibility is released, not preserved through individual replanning

- Code:
  - `SSIA_main.py:80-87` clears the blocked agent's assignment immediately on `PATH_BLOCKED`.
  - `SSIA_task_allocation.py:533-542` and `SSIA_collateral_task_allocation.py:581-590` release infeasible tasks and return the agent to `IDLE`.
- Report:
  - `final_report.tex:193` says path infeasibility is handled by individual agent replanning without triggering a global reauction.
- Why this matters:
  - The code does avoid a global reauction here, but it also does not preserve the original assignment via pure replanning.

### CBS / Drone Handling: report still says inter-drone collisions are impossible, code still checks them for reauction

- Code:
  - `SSIA_task_allocation.py:374-423`, `SSIA_collateral_task_allocation.py:422-471`, and `SSICA_task_allocation.py:568-617` group agents by `agent_type` and check same-type next-move conflicts, which includes drone-drone conflicts.
- Report:
  - `final_report.tex:286-287` says multiple drones can operate at distinct flight altitudes, eliminating the possibility of inter-drone collisions.
- Why this matters:
  - CBS excludes drones, but the higher-level reauction logic still treats drone-drone next-move conflicts as real conflicts.

### Simulation Loop / Allocator Configurations: the report still describes one auction cadence and one shared harness

- Code:
  - `main.py:194-221` defers naive auctioning until after both microsteps.
  - `SSIA_main.py:52-57`, `SSIA_main.py:95`, and `SSIA_main.py:157-160` call `auctioneer.update(...)` after microstep 1, after the drone microstep, and again after triage progress.
  - `SSIA_collateral_main.py:52-57`, `SSIA_collateral_main.py:95`, and `SSIA_collateral_main.py:157-160` do the same.
  - `SSICA_main.py:52-57`, `SSICA_main.py:110`, and `SSICA_main.py:172-175` do the same.
- Report:
  - `final_report.tex:320` says post-observation updates assign tasks to idle agents.
  - `final_report.tex:322` says the drone-microstep update happens without a new auction round.
  - `final_report.tex:324` says a single auction round runs once per step.
  - `final_report.tex:374` says all configurations use the same simulation harness.
- Why this matters:
  - The SSIA-family loops can auction multiple times per full step, while the naive loop defers auctioning until the end of the step.
  - This section is also internally inconsistent: step 3 already assigns tasks, then step 7 says auction runs once per step.

## Results

### Qualitative Analysis / Figure Caption: caption does not match the current renderer

- Code:
  - `visualizer.py:219-223` uses per-agent colors for both dashed path lines and task stars.
- Report:
  - `final_report.tex:387` says task targets are gold stars and planned paths are dashed cyan lines.
- Why this matters:
  - The caption describes an older visual style, not the figure produced by the current renderer.

### Qualitative Analysis: obstacle-triggered global reallocation is still overstated

- Code:
  - `SSIA_main.py:80-87` and `SSIA_task_allocation.py:533-548` release the blocked or infeasible task rather than globally redistributing all incomplete tasks.
  - Global reauction is limited to same-type next-move conflicts or CBS failure in `SSIA_task_allocation.py:415-445` and `SSIA_collateral_task_allocation.py:463-493`.
- Report:
  - `final_report.tex:394` says newly discovered obstacles trigger global reauction and redistribution of all incomplete tasks from the current agent states.
- Why this matters:
  - This repeats the same behavior mismatch from the method section in the results narrative.

### Qualitative Analysis / Effect of Reward Shaping: triage reward is still internally inconsistent and not aligned with SSIA

- Code:
  - `SSIA_task_allocation.py:34-38` uses `R_{\text{triage}} = 3` for SSIA.
  - `SSIA_collateral_task_allocation.py:34-38` and `SSICA_task_allocation.py:30-34` use `R_{\text{triage}} = 8` for SSIA-Collateral and SSICA.
- Report:
  - `final_report.tex:128` and `final_report.tex:155` say `R_{\text{triage}} = 5`.
  - `final_report.tex:395` and `final_report.tex:438` say `R = 8`.
- Why this matters:
  - The report is still internally inconsistent, and the SSIA method description still does not match the current SSIA code.

### Qualitative Analysis: greedy baseline wording still overstates the role of priority

- Code:
  - `naive_task_allocation.py:185-210` iterates tasks in queue order and only uses `(priority, -distance)` when choosing the agent for the current task.
- Report:
  - `final_report.tex:395` says the greedy baseline assigns tasks purely by "distance and priority tier."
- Why this matters:
  - Priority tier does not determine which task is considered first; queue insertion order still dominates task ordering.
