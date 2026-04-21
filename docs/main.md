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

## When To Read It

Read this first if you want the simplest end-to-end runtime path before diving into the auction-based variants.
