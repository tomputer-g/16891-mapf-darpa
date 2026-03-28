"""
Map representations for the DARPA exploration simulation.

  ObservationState   – per-cell visibility / content enum (defines ASCII symbols)
  AgentType          – drone vs. ground-vehicle enum (defines ASCII symbols)
  PATH_SYMBOL        – ASCII symbol for planned-path waypoints
  GroundTruthMap     – the hidden world (agents cannot read this directly)
  KnownMap           – per-agent belief state; starts fully UNKNOWN
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from sim_types import AgentType, ObservationState, PATH_SYMBOL  # noqa: F401


# ===========================================================================
# Map classes
# ===========================================================================

class GroundTruthMap:
    """The real world; never read by agents directly (only via Agent.observe()).

    grid[r][c] is True when the cell is an obstacle.
    buildings maps (r, c) → bool (True = occupied / has objective inside).
    """

    def __init__(
        self,
        grid: List[List[bool]],
        *,
        objectives:   Optional[Set[Tuple[int, int]]]          = None,
        buildings:    Optional[Dict[Tuple[int, int], bool]]   = None,
        agent_starts: Optional[List[Tuple[int, int, str]]]    = None,
    ) -> None:
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.objectives:   Set[Tuple[int, int]]          = objectives   or set()
        self.buildings:    Dict[Tuple[int, int], bool]   = buildings    or {}
        self.agent_starts: List[Tuple[int, int, str]]    = agent_starts or []

    def is_obstacle(self, loc: Tuple[int, int]) -> bool:
        return self.grid[loc[0]][loc[1]]


class KnownMap:
    """
    Global known map of the world that is being uncovered with exploration tasks. All agents use the same KnownMap.
    """

    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.state: List[List[ObservationState]] = [
            [ObservationState.UNKNOWN] * cols for _ in range(rows)
        ]

    #TODO Make this a 3x3 grid update, or variable grid update depending on the agent
    def update(self, loc: Tuple[int, int], obs: ObservationState) -> bool:
        """Write a new observation.  Returns True if the cell state changed."""
        r, c = loc
        if self.state[r][c] != obs:
            self.state[r][c] = obs
            return True
        return False

    def is_passable(self, loc: Tuple[int, int]) -> bool:
        """UNKNOWN cells are optimistically treated as passable."""
        return self.state[loc[0]][loc[1]] != ObservationState.OBSTACLE

    def to_obstacle_grid(self) -> List[List[bool]]:
        """Binary grid for A* (True = blocked)."""
        return [
            [self.state[r][c] == ObservationState.OBSTACLE for c in range(self.cols)]
            for r in range(self.rows)
        ]

    def print_map(
        self,
        agent_locs:   Optional[List[Tuple[int, int, AgentType]]] = None,
        task_targets: Optional[List[Tuple[int, int]]]             = None,
        path:         Optional[List[Tuple[int, int]]]             = None,
    ) -> None:
        """Print an ASCII snapshot of the known map.

        agent_locs entries are (row, col, AgentType).
        task_targets render as ObservationState.OBJECTIVE symbol ('o').
        path waypoints render as PATH_SYMBOL ('*'), excluding current position.
        """
        agent_map  = {(r, c): atype for r, c, atype in (agent_locs or [])}
        target_set = set(task_targets or [])
        path_set   = set(path[1:]) if path else set()   # skip current position
        for r in range(self.rows):
            row = ""
            for c in range(self.cols):
                loc = (r, c)
                if loc in agent_map:
                    row += agent_map[loc].symbol()
                elif loc in target_set:
                    row += ObservationState.OBJECTIVE.symbol()
                elif loc in path_set:
                    row += PATH_SYMBOL
                else:
                    row += self.state[r][c].symbol()
            print(row)


# ===========================================================================
# Scenario factory
# ===========================================================================

def load_scenario(path: str) -> Tuple[GroundTruthMap, List[Tuple[int, int]]]:
    """Deprecated — use load_new_scenario() instead.

    Reads the old HW-style format (4-int agent lines, no objectives/buildings).
    Returns (GroundTruthMap, agent_starts) for backward compatibility.
    """
    lines = Path(path).read_text().splitlines()
    it = (ln for ln in lines if ln.strip())   # skip blank lines

    rows, cols = map(int, next(it).split())

    grid: List[List[bool]] = []
    for _ in range(rows):
        tokens = next(it).split()
        # Support both compact ("@@@..") and space-separated ("@ @ @") formats.
        cells = list(tokens[0]) if len(tokens) == 1 else tokens
        grid.append([cell == '@' for cell in cells])

    num_agents = int(next(it))
    agent_starts: List[Tuple[int, int]] = []
    for _ in range(num_agents):
        sr, sc, *_ = map(int, next(it).split())
        agent_starts.append((sr, sc))

    return GroundTruthMap(grid), agent_starts


def load_new_scenario(path: str) -> GroundTruthMap:
    """Load a scenario file produced by generate_scenario.py.

    File format (for an R×C map with N agents, K objectives, M buildings):

        <R> <C>
        <R rows of space-separated '.' / '@' cells>
        <N>
        <sr> <sc> <D|G>      ← one line per agent
        <K>
        <or> <oc>            ← one line per free-standing objective
        <M>
        <br> <bc> <0|1>      ← one line per building; 1 = occupied

    Cell and agent symbols are defined by ObservationState.symbol() and
    AgentType.symbol() respectively.
    """
    lines = Path(path).read_text().splitlines()
    it = (ln for ln in lines if ln.strip())

    R, C = map(int, next(it).split())

    grid: List[List[bool]] = []
    for _ in range(R):
        tokens = next(it).split()
        cells = list(tokens[0]) if len(tokens) == 1 else tokens
        grid.append([cell == '@' for cell in cells])

    num_agents = int(next(it))
    agent_starts: List[Tuple[int, int, str]] = []
    for _ in range(num_agents):
        sr, sc, atype = next(it).split()
        agent_starts.append((int(sr), int(sc), atype))

    num_objectives = int(next(it))
    objectives: Set[Tuple[int, int]] = set()
    for _ in range(num_objectives):
        r, c = map(int, next(it).split())
        objectives.add((r, c))

    num_buildings = int(next(it))
    buildings: Dict[Tuple[int, int], bool] = {}
    for _ in range(num_buildings):
        r, c, occ = next(it).split()
        buildings[(int(r), int(c))] = bool(int(occ))

    return GroundTruthMap(
        grid,
        objectives=objectives,
        buildings=buildings,
        agent_starts=agent_starts,
    )
