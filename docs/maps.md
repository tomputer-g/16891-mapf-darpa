# `maps.py`

## Role

Defines the hidden environment and the shared discovered map.

## Main Types

- `GroundTruthMap`: the real map, including blocked cells, buildings, and objectives
- `KnownMap`: the team’s shared partial view of the world

## Important Ideas

- Agents should only learn from `GroundTruthMap` through sensing and observation updates.
- Planning and allocation should operate on `KnownMap`.
- Newly discovered blocked cells are what cause the free-space assumption to break and can invalidate committed paths.

## Why This File Matters

A lot of control logic depends on the distinction between “unknown” and “known blocked.” If that distinction is described loosely in the report or in future refactors, the replanning behavior becomes easy to misstate.
