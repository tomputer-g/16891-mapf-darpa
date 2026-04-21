# Before / After Refactor Benchmark

This file records the batch-run results for the repair-vs-reauction refactor on branch `ranais/refactor`.

## Scope

- Baseline snapshot commit: `0c90b7f` `Snapshot simulator state before refactor`
- Refactor commit: `2d147fc` `Add repair-vs-reauction heuristic`
- Environment: `conda run -n idl`
- Maps: `generated/darpa1.txt` through `generated/darpa7.txt`
- Runners: `run_naive.py`, `run_ssia.py`, `run_collateral.py`, `run_ssica.py`
- Max steps per run: `500`
- Metric: total simulation steps until completion

Negative deltas are improvements. Positive deltas are regressions.

## Naive

| Map | Before | After | Delta |
|---|---:|---:|---:|
| `darpa1` | 60 | 60 | 0 |
| `darpa2` | 66 | 66 | 0 |
| `darpa3` | 53 | 53 | 0 |
| `darpa4` | 56 | 56 | 0 |
| `darpa5` | 49 | 49 | 0 |
| `darpa6` | 107 | 107 | 0 |
| `darpa7` | 115 | 115 | 0 |
| `total` | 506 | 506 | 0 |

## SSIA

| Map | Before | After | Delta |
|---|---:|---:|---:|
| `darpa1` | 46 | 46 | 0 |
| `darpa2` | 37 | 37 | 0 |
| `darpa3` | 30 | 30 | 0 |
| `darpa4` | 42 | 42 | 0 |
| `darpa5` | 41 | 39 | -2 |
| `darpa6` | 62 | 62 | 0 |
| `darpa7` | 50 | 50 | 0 |
| `total` | 308 | 306 | -2 |

## SSIA-Collateral

| Map | Before | After | Delta |
|---|---:|---:|---:|
| `darpa1` | 26 | 26 | 0 |
| `darpa2` | 35 | 35 | 0 |
| `darpa3` | 27 | 27 | 0 |
| `darpa4` | 40 | 40 | 0 |
| `darpa5` | 38 | 35 | -3 |
| `darpa6` | 60 | 60 | 0 |
| `darpa7` | 48 | 55 | 7 |
| `total` | 274 | 278 | 4 |

## SSICA

| Map | Before | After | Delta |
|---|---:|---:|---:|
| `darpa1` | 32 | 37 | 5 |
| `darpa2` | 46 | 46 | 0 |
| `darpa3` | 36 | 44 | 8 |
| `darpa4` | 51 | 51 | 0 |
| `darpa5` | 44 | 44 | 0 |
| `darpa6` | 69 | 69 | 0 |
| `darpa7` | 78 | 62 | -16 |
| `total` | 356 | 353 | -3 |

## Takeaways

- The naive baseline is unchanged, which is expected because the refactor only touched the auction-based variants.
- `SSIA` changed very little on these maps and improved slightly on `darpa5`.
- `SSIA-Collateral` stayed close to baseline overall, but the effect was mixed rather than uniformly neutral.
- `SSICA` changed the most. It improved substantially on `darpa7`, regressed on `darpa1` and `darpa3`, and ended slightly better in aggregate.
- On this seven-map set, the refactor behaves like a control-policy heuristic rather than a dominant performance win.

## Reproduction Commands

```bash
conda run -n idl python run_naive.py
conda run -n idl python run_ssia.py
conda run -n idl python run_collateral.py
conda run -n idl python run_ssica.py
```

If needed in a restricted environment:

```bash
MPLCONFIGDIR=/tmp/mapf_mpl XDG_CACHE_HOME=/tmp conda run -n idl python run_ssia.py
```
