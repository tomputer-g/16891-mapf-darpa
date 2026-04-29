# `SSIA/main.py`

## Role

Runs the SSIA simulation loop.

## Main Differences From `main.py`

- uses the SSIA auctioneer instead of the naive allocator
- interleaves observation, task updates, auctioning, replanning, and stepping with the SSIA-specific control logic
- now delegates blocked-path handling back to the auctioneer

## Refactor-Relevant Behavior

On `PATH_BLOCKED`, the loop no longer directly drops the task. It calls the auctioneer’s invalidated-assignment handler so the allocator can choose between:

- local repair and task retention
- full reauction

If full reauction happens, agents already dwelling on incomplete triage tasks are preserved by the allocator instead of being reset.
