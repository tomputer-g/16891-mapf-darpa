"""
Agent model and pathfinding for the DARPA exploration simulation.

  Event       – event dataclass
  plan_path   – A* on a KnownMap
  Agent       – abstract base; subclass for each agent type
  DroneAgent  – aerial agent; observe() cannot distinguish occupied buildings
  GroundAgent – ground vehicle; observe() reveals full building occupancy
"""

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from maps import GroundTruthMap, KnownMap
from sim_types import AgentStatus, AgentType, EventType, ObservationState
from tasks import Task

if TYPE_CHECKING:
    from planner import CBS


# ===========================================================================
# Events
# ===========================================================================

@dataclass
class Event:
    """Discrete event produced during the simulation loop."""
    kind: EventType
    data: dict = field(default_factory=dict)


# ===========================================================================
# Pathfinding  (self-contained A* — no external dependency)
# ===========================================================================

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # N E S W <<Never Eat Soggy Waffles>>


def _move(loc: Tuple[int, int], d: int) -> Tuple[int, int]:
    return loc[0] + _DIRS[d][0], loc[1] + _DIRS[d][1]


def _in_bounds(grid: List[List[bool]], loc: Tuple[int, int]) -> bool:
    return 0 <= loc[0] < len(grid) and 0 <= loc[1] < len(grid[0])


def _dijkstra_heuristic(
    grid: List[List[bool]], goal: Tuple[int, int]
) -> Dict[Tuple[int, int], int]:
    """
    Backward Dijkstra from goal. UNKNOWN cells (grid value False) are
    treated as free, giving an admissible heuristic.
    """
    dist: Dict[Tuple[int, int], int] = {goal: 0}
    heap = [(0, goal)]
    while heap:
        d, loc = heapq.heappop(heap)
        if d > dist.get(loc, float('inf')):
            continue
        for direction in range(4):
            nb = _move(loc, direction)
            if not _in_bounds(grid, nb) or grid[nb[0]][nb[1]]:
                continue
            nd = d + 1
            if nd < dist.get(nb, float('inf')):
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb))
    return dist


def manhattan_distance(rows: int, cols: int, goal: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    dist: Dict[Tuple[int, int], int] = {}
    for r in range(rows):
        for c in range(cols):
            dist[(r, c)] = abs(r - goal[0]) + abs(c - goal[1])
    return dist


def plan_path(
    known_map: KnownMap,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    drone: bool = False,
) -> Optional[List[Tuple[int, int]]]:
    """
    A* on the known map. UNKNOWN cells are treated as FREE (optimistic).
    Ground agents avoid known obstacles. Drones may fly over them.
    Returns a list of (row, col) positions from start to goal, or None.
    """
    grid = known_map.to_obstacle_grid()
    if drone:
        h_val = manhattan_distance(known_map.rows, known_map.cols, goal)
    else:
        h_val = _dijkstra_heuristic(grid, goal)

    if start not in h_val:
        return None

    closed: Dict[Tuple[int, int], dict] = {}
    root = {'loc': start, 'g': 0, 'h': h_val.get(start, 0), 'parent': None}
    counter = 0
    heap: list = [(root['g'] + root['h'], root['h'], start, counter, root)]

    while heap:
        _, _, _, _, curr = heapq.heappop(heap)
        loc = curr['loc']
        if loc in closed:
            continue
        closed[loc] = curr

        if loc == goal:
            path, node = [], curr
            while node:
                path.append(node['loc'])
                node = node['parent']
            path.reverse()
            return path

        for d in range(4):
            nb = _move(loc, d)
            if not _in_bounds(grid, nb) or nb in closed:
                continue
            if (not drone) and (not known_map.is_passable(nb)):
                continue

            g = curr['g'] + 1
            h = h_val.get(nb, 0)
            counter += 1
            child = {'loc': nb, 'g': g, 'h': h, 'parent': curr}
            heapq.heappush(heap, (g + h, h, nb, counter, child))

    return None


# ===========================================================================
# Agent
# ===========================================================================

class Agent:
    """Abstract base for all agents.

    Sensor model : axis-aligned square footprint, side = 2*obs_radius + 1.
    Nav policy   : A* to current_task.target_loc on the current KnownMap.

    Subclasses must implement observe(); everything else is shared.

    EXTEND: battery model, comms range, payload, team_id, sensor cone, …
    """

    def __init__(
        self,
        agent_id:   int,
        planner:    'CBS',
        start:      Tuple[int, int],
        agent_type: AgentType,
        obs_radius: int = 1,
    ) -> None:
        self.id            = agent_id
        self.pos           = start
        self.agent_type    = agent_type
        self.obs_radius    = obs_radius
        self.status        = AgentStatus.IDLE
        self.current_task: Optional[Task] = None
        self._planner:     'CBS'          = planner

    @property
    def path(self) -> List[Tuple[int, int]]:
        return self._planner.get_path(self.id)

    @path.setter
    def path(self, value: List[Tuple[int, int]]) -> None:
        self._planner.set_path(self.id, value)

    def assign_task(self, task: Task) -> None:
        """Called by the auctioneer; puts the agent into REPLANNING state."""
        self.current_task = task
        self.path = []
        self.status = AgentStatus.REPLANNING

    # ------------------------------------------------------------------
    def observe(self, ground_truth: GroundTruthMap, known_map: KnownMap) -> None:
        """Reveal cells within obs_radius footprint; update KnownMap in-place.

        Subclasses override this to apply agent-specific sensor rules.
        EXTEND: line-of-sight, sensor cone, range-dependent noise, …
        """
        raise NotImplementedError(f"{type(self).__name__} must implement observe()")

    def replan(self, known_map: KnownMap) -> bool:
        if self.current_task is None:
            self.status = AgentStatus.IDLE
            return False

        paths = self._planner.plan(
            {self.id: self.pos},
            {self.id: self.current_task.target_loc},
            known_map,
        )
        if paths:
            self.path = paths[self.id]
            self.status = AgentStatus.NAVIGATING
            return True

        self.status = AgentStatus.IDLE
        return False

    def step(self, known_map: KnownMap) -> Optional[Event]:
        """Move one cell along the planned path."""
        if self.status in (AgentStatus.IDLE, AgentStatus.REPLANNING):
            return None
        if len(self.path) < 2:
            return None

        event = self._planner.step(self.id, known_map)
        if event is None:
            return None
        if event.kind == EventType.STEP_COMPLETE:
            self.pos = event.data['pos']
        elif event.kind == EventType.PATH_BLOCKED:
            self.status = AgentStatus.IDLE
        return event


# ===========================================================================
# Concrete agent subclasses
# ===========================================================================

class DroneAgent(Agent):
    """Aerial agent with a wider sensor footprint.

    Observe rules (cannot see inside buildings):
        obstacle      → OBSTACLE
        building      → BUILDING  (occupied or not — drones cannot distinguish)
        otherwise     → FREE
    """

    def __init__(
        self,
        agent_id:   int,
        planner:    'CBS',
        start:      Tuple[int, int],
        obs_radius: int = 2,
    ) -> None:
        super().__init__(agent_id, planner, start, AgentType.DRONE, obs_radius)

    @property
    def path(self) -> List[Tuple[int, int]]:
        return self._planner.get_path(self.id)

    @path.setter
    def path(self, value: List[Tuple[int, int]]) -> None:
        self._planner.set_path(self.id, value, drone=True)

    def observe(self, ground_truth: GroundTruthMap, known_map: KnownMap) -> None:
        r, c = self.pos
        for dr in range(-self.obs_radius, self.obs_radius + 1):
            for dc in range(-self.obs_radius, self.obs_radius + 1):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < ground_truth.rows and 0 <= nc < ground_truth.cols):
                    continue
                loc = (nr, nc)
                if ground_truth.is_obstacle(loc):
                    state = ObservationState.OBSTACLE
                elif loc in ground_truth.buildings:
                    state = ObservationState.BUILDING
                else:
                    state = ObservationState.FREE
                known_map.update(loc, state)

    def replan(self, known_map: KnownMap) -> bool:
        if self.current_task is None:
            self.status = AgentStatus.IDLE
            return False
        paths = self._planner.plan(
            {self.id: self.pos},
            {self.id: self.current_task.target_loc},
            known_map,
            drone=True,
        )
        if paths:
            self.path = paths[self.id]
            self.status = AgentStatus.NAVIGATING
            return True
        self.status = AgentStatus.IDLE
        return False

    def step(self, known_map: KnownMap) -> Optional[Event]:
        """Move one cell along the planned path; drones fly over obstacles."""
        if self.status in (AgentStatus.IDLE, AgentStatus.REPLANNING):
            return None
        if len(self.path) < 2:
            return None
        event = self._planner.step(self.id, known_map)
        if event is None:
            return None
        if event.kind == EventType.STEP_COMPLETE:
            self.pos = event.data['pos']
        return event


class GroundAgent(Agent):
    """Ground vehicle / quadruped with full building visibility.

    Observe rules (can enter and inspect buildings):
        obstacle               → OBSTACLE
        occupied building      → OCCUPIED_BUILDING
        unoccupied building    → BUILDING
        otherwise              → FREE
    """

    def __init__(
        self,
        agent_id:   int,
        planner:    'CBS',
        start:      Tuple[int, int],
        obs_radius: int = 1,
    ) -> None:
        super().__init__(agent_id, planner, start, AgentType.GROUND, obs_radius)

    def observe(self, ground_truth: GroundTruthMap, known_map: KnownMap) -> None:
        r, c = self.pos
        for dr in range(-self.obs_radius, self.obs_radius + 1):
            for dc in range(-self.obs_radius, self.obs_radius + 1):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < ground_truth.rows and 0 <= nc < ground_truth.cols):
                    continue
                loc = (nr, nc)
                if ground_truth.is_obstacle(loc):
                    state = ObservationState.OBSTACLE
                elif loc in ground_truth.buildings:
                    state = (ObservationState.OCCUPIED_BUILDING
                             if ground_truth.buildings[loc]
                             else ObservationState.BUILDING)
                else:
                    state = ObservationState.FREE
                known_map.update(loc, state)
