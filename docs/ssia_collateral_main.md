# `SSIA_collateral/main.py`

## Role

Runs the collateral-aware SSIA simulation loop.

## Main Differences From `SSIA/main.py`

- uses the collateral-aware auctioneer
- keeps the same repair-vs-reauction handoff pattern when a path is blocked
- preserves already-dwelling triage agents if full reauction happens

## Why It Exists

This variant rewards exploration that is expected to reveal useful additional free space along the route to a task.
