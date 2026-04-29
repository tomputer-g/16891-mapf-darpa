# `main.py`

## Role

Runs the naive baseline simulation.

## Key Responsibilities

- load the scenario
- create `KnownMap`, planner, and agents
- instantiate `NaiveTaskAuctioneer`
- observe, update tasks, assign idle agents, and step the simulation

## Important Note

This file does not use the generic `TaskAuctioneer` in `tasks.py`. Its live allocator is `NaiveTaskAuctioneer` from `naive_task_allocation.py`.

It also keeps the older naive blocked-path policy:

- on `PATH_BLOCKED`, release the current task
- clear the agent path
- return the agent to `IDLE`

The repair-vs-reauction heuristic is only implemented in the SSIA-family variants.

## When To Read It

Read this first if you want the simplest end-to-end runtime path before diving into the auction-based variants.
