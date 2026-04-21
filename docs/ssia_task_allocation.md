# `SSIA/task_allocation.py`

## Role

Implements the sequential single-item auction allocator.

## Main Behavior

- generates and updates tasks
- computes bids using reward divided by execution cost
- assigns tasks sequentially while respecting agent-type constraints
- triggers coordinated replanning for ground agents

## Refactor Additions

- bid objects now retain execution-cost detail
- winner and runner-up metadata are stored on assigned tasks
- `handle_invalidated_assignment(...)` decides whether to keep the task after local repair or trigger full reauction

## Decision Logic

The current heuristic is:

1. estimate repaired retained quality
2. compare it against the stored runner-up with slack
3. keep the assignment if the retained bid is still competitive
4. reauction if repair fails, retained quality is clearly worse, or CBS repair fails

## What To Be Careful About

This is heuristic control logic layered on top of an already suboptimal auction process. It should be described as churn reduction, not as a proof of better optimality.
