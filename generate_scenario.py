"""
generate_scenario.py — Random scenario generator for the DARPA exploration simulation.

Generates a scenario file in the format consumed by maps.load_new_scenario().

Usage
-----
    python3 generate_scenario.py [output_path]
        [--rows R] [--cols C]
        [--drones D] [--ground G]
        [--objectives K]
        [--buildings M] [--occupied B]
        [--seed S]

Defaults
--------
    output_path : instances/generated.txt
    rows, cols  : 15 × 15
    drones      : 2
    ground      : 3
    objectives  : 4
    buildings   : 4
    occupied    : 2   (must be ≤ buildings)
    seed        : random

Output format
-------------
    <R> <C>
    <R rows of space-separated '.' / '@' cells>
    <num_agents>
    <sr> <sc> <D|G>          ← one line per agent
    <num_objectives>
    <or> <oc>                ← one line per free-standing objective
    <num_buildings>
    <br> <bc> <0|1>          ← one line per building; 1 = occupied (has objective inside)
"""

import argparse
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Obstacle shape generators
# ---------------------------------------------------------------------------

def _shape_single(r: int, c: int) -> List[Tuple[int, int]]:
    return [(r, c)]


def _shape_hbar(r: int, c: int, w: int) -> List[Tuple[int, int]]:
    return [(r, c + i) for i in range(w)]


def _shape_vbar(r: int, c: int, h: int) -> List[Tuple[int, int]]:
    return [(r + i, c) for i in range(h)]


def _shape_block(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    return [(r + dr, c + dc) for dr in range(h) for dc in range(w)]


def _shape_lshape(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    """Vertical arm + horizontal foot."""
    cells = [(r + i, c) for i in range(h)]          # vertical
    cells += [(r + h - 1, c + j) for j in range(1, w)]  # horizontal foot
    return cells


def _random_shape(rng: random.Random, rows: int, cols: int) -> List[Tuple[int, int]]:
    """Return a random obstacle shape anchored at a random position."""
    # Leave a 1-cell border free on all sides.
    r = rng.randint(1, rows - 2)
    c = rng.randint(1, cols - 2)

    kind = rng.choice(["single", "hbar", "vbar", "block", "lshape"])
    max_h = min(4, rows - 1 - r)
    max_w = min(4, cols - 1 - c)

    if kind == "single" or max_h < 1 or max_w < 1:
        return _shape_single(r, c)
    if kind == "hbar":
        w = rng.randint(2, max(2, max_w))
        return _shape_hbar(r, c, w)
    if kind == "vbar":
        h = rng.randint(2, max(2, max_h))
        return _shape_vbar(r, c, h)
    if kind == "block":
        h = rng.randint(2, max(2, max_h))
        w = rng.randint(2, max(2, max_w))
        return _shape_block(r, c, h, w)
    # lshape
    h = rng.randint(2, max(2, max_h))
    w = rng.randint(2, max(2, max_w))
    return _shape_lshape(r, c, h, w)


# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------

def _free_cells(grid: List[List[bool]]) -> Set[Tuple[int, int]]:
    rows, cols = len(grid), len(grid[0])
    return {(r, c) for r in range(rows) for c in range(cols) if not grid[r][c]}


def _bfs_component(grid: List[List[bool]], start: Tuple[int, int]) -> Set[Tuple[int, int]]:
    rows, cols = len(grid), len(grid[0])
    visited: Set[Tuple[int, int]] = set()
    q: deque = deque([start])
    visited.add(start)
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not grid[nr][nc] and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return visited


def _is_connected(grid: List[List[bool]]) -> bool:
    free = _free_cells(grid)
    if not free:
        return False
    component = _bfs_component(grid, next(iter(free)))
    return component == free


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def _generate_grid(
    rows: int,
    cols: int,
    target_ratio: float,
    rng: random.Random,
    max_attempts: int = 200,
) -> List[List[bool]]:
    """
    Place random obstacle shapes until ~target_ratio of cells are blocked,
    then verify full connectivity.  Retry from scratch on failure.
    """
    target = int(rows * cols * target_ratio)
    for _ in range(max_attempts):
        grid = [[False] * cols for _ in range(rows)]
        obstacle_count = 0

        for _ in range(200):
            if obstacle_count >= target:
                break
            cells = _random_shape(rng, rows, cols)
            # Only add shapes that stay within bounds (border excluded)
            valid = all(1 <= r < rows - 1 and 1 <= c < cols - 1 for r, c in cells)
            if not valid:
                continue
            for r, c in cells:
                if not grid[r][c]:
                    grid[r][c] = True
                    obstacle_count += 1

        if _is_connected(grid):
            return grid

    raise RuntimeError(
        f"Could not generate a connected {rows}×{cols} grid after {max_attempts} attempts."
    )


# ---------------------------------------------------------------------------
# Entity placement
# ---------------------------------------------------------------------------

def _sample_cells(
    free: Set[Tuple[int, int]],
    n: int,
    rng: random.Random,
    exclude: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    available = sorted(free - exclude)
    if len(available) < n:
        raise RuntimeError(
            f"Not enough free cells to place {n} entities "
            f"(need {n}, have {len(available)} available)."
        )
    return rng.sample(available, n)


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------

def generate_scenario(
    rows: int = 15,
    cols: int = 15,
    num_drones: int = 2,
    num_ground: int = 3,
    num_objectives: int = 4,
    num_buildings: int = 4,
    num_occupied: int = 2,
    seed: int | None = None,
    obstacle_ratio: float = 0.15,
) -> str:
    """
    Generate a scenario and return it as a formatted string ready to write to disk.
    """
    if num_occupied > num_buildings:
        raise ValueError("num_occupied must be ≤ num_buildings")

    rng = random.Random(seed)
    num_agents = num_drones + num_ground

    # ---- obstacles --------------------------------------------------------
    grid = _generate_grid(rows, cols, obstacle_ratio, rng)
    free = _free_cells(grid)

    # ---- agents -----------------------------------------------------------
    occupied: Set[Tuple[int, int]] = set()
    agent_cells = _sample_cells(free, num_agents, rng, occupied)
    occupied.update(agent_cells)
    agent_starts = [
        (r, c, "D" if i < num_drones else "G")
        for i, (r, c) in enumerate(agent_cells)
    ]

    # ---- buildings --------------------------------------------------------
    building_cells = _sample_cells(free, num_buildings, rng, occupied)
    occupied.update(building_cells)
    buildings: Dict[Tuple[int, int], bool] = {
        (r, c): (i < num_occupied) for i, (r, c) in enumerate(building_cells)
    }

    # ---- free-standing objectives ----------------------------------------
    objective_cells = _sample_cells(free, num_objectives, rng, occupied)
    objectives: Set[Tuple[int, int]] = set(objective_cells)

    # ---- final connectivity check (all key cells reachable) ---------------
    key_cells = (
        {(r, c) for r, c, _ in agent_starts}
        | set(buildings.keys())
        | objectives
    )
    start = next(iter(key_cells))
    component = _bfs_component(grid, start)
    if not key_cells.issubset(component):
        raise RuntimeError(
            "Generated map has unreachable key cells. Try a different seed."
        )

    # ---- format output ----------------------------------------------------
    lines: List[str] = []
    lines.append(f"{rows} {cols}")
    for r in range(rows):
        lines.append(" ".join("@" if grid[r][c] else "." for c in range(cols)))

    lines.append(str(num_agents))
    for r, c, atype in agent_starts:
        lines.append(f"{r} {c} {atype}")

    lines.append(str(num_objectives))
    for r, c in sorted(objectives):
        lines.append(f"{r} {c}")

    lines.append(str(num_buildings))
    for (r, c), occupied_flag in sorted(buildings.items()):
        lines.append(f"{r} {c} {int(occupied_flag)}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a random DARPA exploration scenario file."
    )
    p.add_argument("output", nargs="?", default="instances/generated.txt",
                   help="Output file path (default: instances/generated.txt)")
    p.add_argument("--rows",      type=int, default=15)
    p.add_argument("--cols",      type=int, default=15)
    p.add_argument("--drones",    type=int, default=2,  dest="drones")
    p.add_argument("--ground",    type=int, default=3,  dest="ground")
    p.add_argument("--objectives",type=int, default=4,  dest="objectives")
    p.add_argument("--buildings", type=int, default=4,  dest="buildings")
    p.add_argument("--occupied",  type=int, default=2,  dest="occupied")
    p.add_argument("--seed",      type=int, default=None)
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    content = generate_scenario(
        rows=args.rows,
        cols=args.cols,
        num_drones=args.drones,
        num_ground=args.ground,
        num_objectives=args.objectives,
        num_buildings=args.buildings,
        num_occupied=args.occupied,
        seed=args.seed,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    print(f"Wrote {args.rows}×{args.cols} scenario → {out}")


if __name__ == "__main__":
    main()
