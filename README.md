# 16891-mapf-darpa

## Generating a scenario

```bash
python3 generate_scenario.py [output_path] [options]
```

| Option | Default | Description |
|---|---|---|
| `output_path` | `instances/generated.txt` | Path to write the scenario file |
| `--rows R` | `15` | Map height in cells |
| `--cols C` | `15` | Map width in cells |
| `--drones D` | `2` | Number of drone agents |
| `--ground G` | `3` | Number of ground vehicle agents |
| `--objectives K` | `4` | Number of free-standing objectives (not inside buildings) |
| `--buildings M` | `4` | Total number of buildings |
| `--occupied B` | `2` | Buildings that contain an objective (`B ≤ M`) |
| `--seed S` | *(random)* | RNG seed for reproducibility |
```