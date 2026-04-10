import heapq
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from maps import KnownMap
from agents import Event
from sim_types import EventType

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Loc  = Tuple[int, int]
Path = List[Loc]

# ---------------------------------------------------------------------------
# Public dataclasses (used by callers for bidding / collision inspection)
# ---------------------------------------------------------------------------

@dataclass
class Constraint:
    """Forbids agent from occupying locs at timestep.

    Vertex constraint: locs = [loc]
    Edge constraint:   locs = [loc_from, loc_to]
    """
    agent:    int
    locs:     List[Loc]
    timestep: int


@dataclass
class Collision:
    """First detected conflict between two agents' paths."""
    a1:       int
    a2:       int
    locs:     List[Loc]   # same encoding as Constraint.locs
    timestep: int


# ---------------------------------------------------------------------------
# Private dataclasses (internal to planner)
# ---------------------------------------------------------------------------

@dataclass
class _AStarNode:
    loc:    Loc
    g:      int
    h:      int
    t:      int
    parent: Optional['_AStarNode'] = field(default=None, compare=False, repr=False)


@dataclass
class _CBSNode:
    cost:        int
    constraints: List[Constraint]
    paths:       Dict[int, Path]
    collisions:  List[Collision]


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _move(loc: Loc, d: int) -> Loc:
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[d][0], loc[1] + directions[d][1]


def _in_map(my_map: List[List[bool]], loc: Loc) -> bool:
    return 0 <= loc[0] < len(my_map) and 0 <= loc[1] < len(my_map[0])


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
            if not _in_map(grid, nb) or grid[nb[0]][nb[1]]:
                continue
            nd = d + 1
            if nd < dist.get(nb, float('inf')):
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb))
    return dist


def _manhattan_heuristic(rows: int, cols: int, goal: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    dist: Dict[Tuple[int, int], int] = {}
    for r in range(rows):
        for c in range(cols):
            dist[(r, c)] = abs(r - goal[0]) + abs(c - goal[1])
    return dist


def _get_loc(path: Path, t: int) -> Loc:
    return path[min(t, len(path) - 1)]


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

def _build_constraint_table(
    constraints: List[Constraint], agent: int
) -> Tuple[Dict[int, List[Constraint]], int]:
    table: Dict[int, List[Constraint]] = {}
    max_t = 0
    for c in constraints:
        if c.agent == agent:
            max_t = max(max_t, c.timestep)
            table.setdefault(c.timestep, []).append(c)
    return table, max_t


def _is_constrained(
    curr_loc: Loc, next_loc: Loc, next_t: int,
    table: Dict[int, List[Constraint]],
) -> bool:
    if next_t not in table:
        return False
    for c in table[next_t]:
        if c.locs == [next_loc]:
            return True
        if c.locs == [curr_loc, next_loc] or c.locs == [next_loc, curr_loc]:
            return True
    return False


# ---------------------------------------------------------------------------
# Low-level space-time A*
# ---------------------------------------------------------------------------

def _a_star(
    my_map:      List[List[bool]],
    start:       Loc,
    goal:        Loc,
    h_values:    Dict[Loc, int],
    agent:       int,
    constraints: List[Constraint],
    drone:       bool = False,
    T:           int  = 1_000_000,
) -> Optional[Path]:
    table, max_ct = _build_constraint_table(constraints, agent)
    open_list: list = []
    closed: Dict[Tuple[Loc, int], _AStarNode] = {}
    counter = 0

    root = _AStarNode(loc=start, g=0, h=h_values.get(start, 0), t=0)
    heapq.heappush(open_list, (root.g + root.h, root.h, start, 0, counter, root))
    closed[(start, 0)] = root

    while open_list:
        _, _, _, _, _, curr = heapq.heappop(open_list)
        if curr.loc == goal:
            if not any(_is_constrained(curr.loc, curr.loc, s, table)
                       for s in range(curr.t, max_ct + 1)):
                path: Path = []
                node: Optional[_AStarNode] = curr
                while node:
                    path.append(node.loc)
                    node = node.parent
                path.reverse()
                return path
        if curr.t >= T:
            continue
        for d in range(4 if drone else 5):
            nb = _move(curr.loc, d)
            if not _in_map(my_map, nb):
                continue
            if not drone and my_map[nb[0]][nb[1]]:
                continue
            if _is_constrained(curr.loc, nb, curr.t + 1, table):
                continue
            child = _AStarNode(loc=nb, g=curr.g + 1, h=h_values.get(nb, 0),
                               t=curr.t + 1, parent=curr)
            key = (nb, curr.t + 1)
            if key not in closed or closed[key].g + closed[key].h > child.g + child.h:
                closed[key] = child
                counter += 1
                heapq.heappush(open_list, (child.g + child.h, child.h, nb, child.t, counter, child))

    return None

# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------

def _detect_first_collision(path1: Path, path2: Path) -> Optional[Collision]:
    max_t = max(len(path1), len(path2))
    for t in range(max_t):
        if _get_loc(path1, t) == _get_loc(path2, t):
            return Collision(a1=0, a2=0, locs=[_get_loc(path1, t)], timestep=t)
    for t in range(min(len(path1), len(path2)) - 1):
        if (_get_loc(path1, t) == _get_loc(path2, t + 1)
                and _get_loc(path1, t + 1) == _get_loc(path2, t)):
            return Collision(a1=0, a2=0,
                             locs=[_get_loc(path1, t), _get_loc(path1, t + 1)],
                             timestep=t + 1)
    return None


def _detect_all_collisions(paths: Dict[int, Path]) -> List[Collision]:
    collisions: List[Collision] = []
    ids = list(paths.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a1, a2 = ids[i], ids[j]
            c = _detect_first_collision(paths[a1], paths[a2])
            if c:
                c.a1, c.a2 = a1, a2
                collisions.append(c)
    return collisions


# ---------------------------------------------------------------------------
# CBS class
# ---------------------------------------------------------------------------

class CBS:
    """Conflict-Based Search coordinator for ground agents.

    Drone agents are excluded — they resolve altitude conflicts independently
    and never need to coordinate paths with ground vehicles.

    Typical usage
    -------------
        cbs = CBS(rows, cols)

        # Bid phase: compute cost without committing paths.
        paths = cbs.plan(starts, goals, known_map)  # returns Dict[int, Path] or None
        bid   = len(paths[agent_id]) - 1 if paths else float('inf')

        # Commit phase: winner explicitly registers its path.
        if paths:
            cbs.set_path(agent_id, paths[agent_id])

        # Navigation: step() and get_path() only see committed paths.
        event = cbs.step(agent_id, known_map)
        path  = cbs.get_path(agent_id)              # Path, compatible with visualizer

    Integration with GroundAgent.step()
    ------------------------------------
    The CBS instance would be shared across all ground agents (e.g. held by the
    simulator or auctioneer). To swap GroundAgent.step() over to CBS-planned
    motion, replace the body of GroundAgent.step() with:

        event = self._cbs.step(self.id, known_map)
        if event and event.kind == EventType.STEP_COMPLETE:
            self.pos  = event.data['pos']
            self.path = self._cbs.get_path(self.id)
        return event
    """

    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self._paths: Dict[int, Path] = {}
        self._drone_ids: Set[int] = set()

    def plan(
        self,
        starts:         Dict[int, Loc],
        goals:          Dict[int, Loc],
        known_map:      KnownMap,
        drone:          bool = False,
        max_expansions: int  = 5_000,
    ) -> Optional[Dict[int, Path]]:
        """Compute paths for the given agents without modifying state.

        drone=True  — Manhattan heuristic, obstacles ignored, no collision
                      resolution; each agent is planned independently.
        drone=False — Dijkstra heuristic, obstacle avoidance, CBS collision
                      resolution among all provided agents.

        Heuristics are computed fresh from each goal on the current known_map.
        Returns {agent_id: Path} if a solution is found, None otherwise.
        Call set_path() to commit a path for navigation.
        """
        my_map = known_map.to_obstacle_grid()
        agent_ids = list(starts.keys())

        if drone:
            h: Dict[int, Dict[Loc, int]] = {
                aid: _manhattan_heuristic(self.rows, self.cols, goals[aid])
                for aid in agent_ids
            }
            paths: Dict[int, Path] = {}
            for aid in agent_ids:
                p = _a_star(my_map, starts[aid], goals[aid], h[aid], aid, [], drone=True)
                if p is None:
                    return None
                paths[aid] = p
            return paths

        h = {aid: _dijkstra_heuristic(my_map, goals[aid]) for aid in agent_ids}

        root_paths: Dict[int, Path] = {}
        for aid in agent_ids:
            p = _a_star(my_map, starts[aid], goals[aid], h[aid], aid, [])
            if p is None:
                return None
            root_paths[aid] = p

        root = _CBSNode(
            cost=sum(len(p) - 1 for p in root_paths.values()),
            constraints=[],
            paths=root_paths,
            collisions=_detect_all_collisions(root_paths),
        )

        open_list: list = []
        counter = 0
        heapq.heappush(open_list, (root.cost, len(root.collisions), counter, root))

        while open_list and counter < max_expansions:
            _, _, _, P = heapq.heappop(open_list)
            collisions = _detect_all_collisions(P.paths)
            if not collisions:
                return P.paths

            col = collisions[0]
            for aid in [col.a1, col.a2]:
                Q = copy.deepcopy(P)
                Q.constraints.append(Constraint(agent=aid, locs=col.locs, timestep=col.timestep))
                p = _a_star(my_map, starts[aid], goals[aid], h[aid], aid, Q.constraints)
                if p is not None:
                    Q.paths[aid] = p
                    Q.collisions = _detect_all_collisions(Q.paths)
                    Q.cost = sum(len(path) - 1 for path in Q.paths.values())
                    counter += 1
                    heapq.heappush(open_list, (Q.cost, len(Q.collisions), counter, Q))

        return None

    def set_path(self, agent_id: int, path: Path, drone: bool = False) -> None:
        """Commit a path for agent_id; used after winning a bid or replanning.

        Drone paths are stored but excluded from collision detection in plan().
        """
        self._paths[agent_id] = path
        if drone:
            self._drone_ids.add(agent_id)

    def get_path(self, agent_id: int) -> Path:
        """Return the remaining committed path for agent_id."""
        return self._paths.get(agent_id, [])

    def step(self, agent_id: int, known_map: KnownMap) -> Optional[Event]:
        """Advance agent_id one step along its CBS-planned path.

        Returns STEP_COMPLETE with the new position, PATH_BLOCKED if the next
        cell turned out to be an obstacle, or None if no path is available.
        Drone agents bypass the passability check (they fly over obstacles).
        """
        path = self._paths.get(agent_id)
        if not path or len(path) < 2:
            return None
        next_pos = path[1]
        if agent_id not in self._drone_ids and not known_map.is_passable(next_pos):
            self._paths[agent_id] = []
            return Event(EventType.PATH_BLOCKED, {'agent': agent_id, 'blocked_at': next_pos})
        self._paths[agent_id] = path[1:]
        return Event(EventType.STEP_COMPLETE, {'agent': agent_id, 'pos': next_pos})
