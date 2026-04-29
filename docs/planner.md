# `planner.py`

## Role

Implements the low-level planners and the ground-agent conflict resolution layer.

## Main Types

- `Constraint`
- `Collision`
- `_AStarNode`
- `_CBSNode`
- `CBS`

## Important Behavior

- Ground agents are coordinated through CBS.
- Drones do not participate in the same CBS conflict model.
- `CBS.step()` advances committed paths and can emit `PATH_BLOCKED` when the optimistic free-space assumption breaks because the next committed cell is now known to be impassable.
- `PATH_BLOCKED` is not itself a robot-robot collision event. Same-type next-move conflict checks happen later in the auctioneer logic.

## Refactor-Relevant Note

The current heuristic treats `PATH_BLOCKED` as a path-level failure first. Auctioneers now get a chance to repair locally and only escalate to full reauction if the retained assignment no longer looks worthwhile or coordinated repair fails.
