# Codebase Reference

This folder is the longer-form reference set for the simulator.

## Recommended Read Order

1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. [CHANGELOG.md](CHANGELOG.md)
3. [tasks.md](tasks.md)
4. [agents.md](agents.md)
5. [planner.md](planner.md)
6. the allocator doc for the variant you care about
7. the matching simulation-loop doc

## File Index

- [QUICK_REFERENCE.md](QUICK_REFERENCE.md): compact runtime map
- [CHANGELOG.md](CHANGELOG.md): branch and refactor history
- [CLAUDE.md](CLAUDE.md): local agent notes and conventions
- [before_after.md](before_after.md): recorded benchmark comparison
- [FINAL_REPORT_CODE_DISCREPANCIES.md](FINAL_REPORT_CODE_DISCREPANCIES.md): report/code mismatches
- [FINAL_REPORT_TRIM_NOTES.md](FINAL_REPORT_TRIM_NOTES.md): report editing notes
- [TODO.md](TODO.md): working notes captured during the refactor/report pass
- [generate_scenario.md](generate_scenario.md): scenario generator notes
- [maps.md](maps.md): hidden and known world model
- [sim_types.md](sim_types.md): shared enums
- [tasks.md](tasks.md): task model and assignment snapshot metadata
- [agents.md](agents.md): agent state machine
- [planner.md](planner.md): A* and CBS planning layer
- [main.md](main.md): naive simulation loop
- [naive_task_allocation.md](naive_task_allocation.md): greedy baseline allocation
- [ssia_main.md](ssia_main.md): SSIA runtime loop
- [ssia_task_allocation.md](ssia_task_allocation.md): SSIA auction and repair logic
- [ssia_collateral_main.md](ssia_collateral_main.md): collateral SSIA runtime loop
- [ssia_collateral_task_allocation.md](ssia_collateral_task_allocation.md): collateral SSIA auction and repair logic
- [ssica_main.md](ssica_main.md): SSICA runtime loop
- [ssica_task_allocation.md](ssica_task_allocation.md): SSICA queueing and repair logic

## Why This Exists

The top-level docs were too thin for resuming work after a pause. This folder is meant to answer:

- which file owns which behavior
- where the current repair-vs-reauction heuristic lives
- what assumptions each allocator makes
- which code paths are active versus legacy or misleading
- where the report-writing notes and benchmark history live

Historical note:

- `TaskAuctioneer` in `tasks.py` is kept as reference code only.
- The live naive baseline uses `NaiveTaskAuctioneer` in `naive_task_allocation.py`.
