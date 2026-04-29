# 16891 MAPF DARPA Simulator

This repo simulates heterogeneous multi-agent exploration and triage on partially observed grid maps.

The main algorithm variants are:

- `main.py`: naive greedy task allocation
- `SSIA/main.py`: sequential single-item auction with reward-shaped bids
- `SSIA_collateral/main.py`: SSIA plus collateral exploration reward
- `SSICA/main.py`: concurrent queue-based auction variant

Ground vehicles are coordinated with `CBS` in `planner.py`. Drones plan independently over shared discovered free space.

## Environment

The current working setup uses the `idl` conda environment.

Example commands:

```bash
conda run -n idl python run_naive.py
conda run -n idl python run_ssia.py
conda run -n idl python run_collateral.py
conda run -n idl python run_ssica.py
```

If Matplotlib cache permissions are an issue in a restricted environment, run with:

```bash
MPLCONFIGDIR=/tmp/mapf_mpl XDG_CACHE_HOME=/tmp conda run -n idl python run_ssia.py
```

## Benchmark Runners

The batch runners execute the existing seven DARPA-style maps in `generated/`:

- `run_naive.py`
- `run_ssia.py`
- `run_collateral.py`
- `run_ssica.py`

Each script writes a root-level `*_results.txt` file with `generated/darpa1.txt` through `generated/darpa7.txt`.

The refactor benchmark summary is tracked in [docs/before_after.md](docs/before_after.md).

## Scenario Generation

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
| `--objectives K` | `4` | Number of free-standing objectives |
| `--buildings M` | `4` | Total number of buildings |
| `--occupied B` | `2` | Buildings that contain an objective (`B ≤ M`) |
| `--seed S` | *(random)* | RNG seed for reproducibility |

More detail is in [docs/generate_scenario.md](docs/generate_scenario.md).

## Documentation Map

- [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md): top-level runtime map and current design notes
- [docs/before_after.md](docs/before_after.md): baseline vs refactor benchmark results
- [docs/FINAL_REPORT_CODE_DISCREPANCIES.md](docs/FINAL_REPORT_CODE_DISCREPANCIES.md): report/code mismatches organized by report section
- [docs/FINAL_REPORT_TRIM_NOTES.md](docs/FINAL_REPORT_TRIM_NOTES.md): report trimming notes plus repo companion links
- [docs/README.md](docs/README.md): detailed module reference set
- [reports/final_report.tex](reports/final_report.tex) and [reports/report.tex](reports/report.tex): paper drafts

## Current Refactor Note

Branch `ranais/refactor` adds a repair-vs-reauction heuristic to the auction-based allocators:

- store winner and runner-up assignment metadata on tasks
- attempt local path repair first when a committed path is invalidated
- only trigger a full reauction when repaired retained quality falls below the stored alternative by more than a slack threshold, or coordinated repair fails
- blocked-path events now move agents into `REPLANNING` while the allocator decides whether to keep or replace the assignment
- full reauction no longer clears agents already on-target and actively dwelling on incomplete triage tasks

This is a heuristic for reducing assignment churn. It is not presented as a provably optimal TAPF policy.
