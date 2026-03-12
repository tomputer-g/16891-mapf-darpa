"""
Map representations for the DARPA exploration simulation.

  ObservationState   – per-cell visibility enum
  GroundTruthMap     – the hidden world (agents cannot read this directly)
  KnownMap           – per-agent belief state; starts fully UNKNOWN
  build_default_scenario – factory for the default 10×10 test map
"""

from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple


# ===========================================================================
# Enums
# ===========================================================================

class ObservationState(Enum):
    """Visibility / content state of a single map cell."""
    UNKNOWN   = auto()   # never seen by any agent
    FREE      = auto()   # observed, passable
    OBSTACLE  = auto()   # observed, impassable
    OBJECTIVE = auto()   # observed, high-value objective present (see ObjectiveTask)

    # EXTEND: HAZARD, CONTESTED, EXPLORED_STALE, COMMS_RELAY, …

    def symbol(self) -> str:
        return {
            ObservationState.UNKNOWN:   '?',
            ObservationState.FREE:      '.',
            ObservationState.OBSTACLE:  '#',
            ObservationState.OBJECTIVE: '!',
        }[self]


# ===========================================================================
# Maps
# ===========================================================================

class GroundTruthMap:
    """
    The real world.  grid[r][c] is True when the cell is an obstacle.
    Agents interact with this only through Agent.observe().
    """

    def __init__(self, grid: List[List[bool]]) -> None:
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def is_obstacle(self, loc: Tuple[int, int]) -> bool:
        return self.grid[loc[0]][loc[1]]


class KnownMap:
    """
    Per-agent belief state.  Every cell starts as UNKNOWN.
    """

    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.state: List[List[ObservationState]] = [
            [ObservationState.UNKNOWN] * cols for _ in range(rows)
        ]

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
        agent_locs:   Optional[List[Tuple[int, int]]] = None,
        task_targets: Optional[List[Tuple[int, int]]] = None,
        path:         Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        agent_set  = set(agent_locs   or [])
        target_set = set(task_targets or [])
        path_set   = set(path[1:]) if path else set()   # skip current position
        for r in range(self.rows):
            row = ""
            for c in range(self.cols):
                loc = (r, c)
                if loc in agent_set:
                    row += 'A'
                elif loc in target_set:
                    row += 'T'
                elif loc in path_set:
                    row += '*'
                else:
                    row += self.state[r][c].symbol()
            print(row)


# ===========================================================================
# Scenario factory
# ===========================================================================

def build_default_scenario() -> Tuple[GroundTruthMap, List[Tuple[int, int]]]:
    """
    10 × 10 map with two obstacle clusters.  The full map is initially
    hidden from agents; they must explore to reveal it.

    Ground truth (row increases downward):

        . . . . . . . . . .    row 0
        . . . # . . . . . .    row 1  ← vertical wall at col 3, rows 1–3
        . . . # . . . . . .    row 2
        . . . # . . . . . .    row 3
        . . . . . . . . . .    row 4
        . . . . . . . . . .    row 5
        . . . . . . . # . .    row 6  ← side barrier at col 7, rows 6–7
        . . . . . . . # . .    row 7
        . . . . . . . . . .    row 8
        . . . . . . . . . .    row 9

    Returns the ground-truth map and a list of agent start positions.
    """
    rows, cols = 10, 10
    obstacle_locs = {(1, 3), (2, 3), (3, 3), (6, 7), (7, 7)}
    grid = [
        [((r, c) in obstacle_locs) for c in range(cols)]
        for r in range(rows)
    ]
    return GroundTruthMap(grid), [(0, 0)]


def load_scenario(path: str) -> Tuple[GroundTruthMap, List[Tuple[int, int]]]:
    """
    Load a scenario from a text file.

    File format
    -----------
    Line 1          : <rows> <cols>
    Lines 2..rows+1 : space-separated cells — '@' obstacle, '.' free
    Line rows+2     : <num_agents>
    Lines rows+3..  : <start_row> <start_col> <goal_row> <goal_col>
                      (goal coordinates are recorded but not used)

    Example (maps/exp0.txt):
        4 7
        @ @ @ @ @ @ @
        @ . . . . . @
        @ @ @ . @ @ @
        @ @ @ @ @ @ @
        2
        1 1 1 5
        1 2 1 4
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

