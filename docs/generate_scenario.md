# `generate_scenario.py`

## Role

Generates standalone scenario files for the simulator.

## What It Controls

- grid size
- number of drones
- number of ground vehicles
- number of free objectives
- number of buildings
- number of occupied buildings
- random seed

## Typical Use

```bash
python3 generate_scenario.py instances/generated.txt --rows 15 --cols 15 --drones 2 --ground 3 --seed 7
```

The repo’s benchmark runners do not call this script. They use the existing maps in `generated/`.

## Why It Matters

If later experiments need more stressful MAPF behavior, this is one of the first places to revisit because current benchmark maps are fairly small and only use two ground agents.
