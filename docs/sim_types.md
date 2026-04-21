# `sim_types.py`

## Role

Defines the enums shared across the simulator.

## Main Enums

- `ObservationState`
- `AgentType`
- `AgentStatus`
- `EventType`

## Why It Matters

The simulation loops, planner, and agents communicate through these enums. If a control path feels confusing, check the event and status values here before assuming the bug is deeper in the auction code.
