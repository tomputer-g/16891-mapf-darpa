"""
DARPA Multi-Agent Exploration Simulation

The simulation terminates when the task queue is exhausted — there is no
fixed goal location.  Exploration tasks are generated dynamically as the
agent reveals frontier cells; each task is complete once the auctioneer
confirms an agent has observed its target cell.

Architecture
------------
  Task              – base class for all assignable work items
  ExplorationTask   – navigate to and observe an UNKNOWN frontier cell
  TaskAuctioneer    – owns the queue; assigns work via auction rounds
  ObservationState  – per-cell visibility enum
  GroundTruthMap    – the hidden world (agents cannot read this directly)
  KnownMap          – per-agent belief state; starts fully UNKNOWN
  Agent             – navigates to assigned tasks using A*; no fixed goal
  run_simulation()  – top-level event loop

Extension points are marked with  # EXTEND:  comments throughout.
"""

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


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


class AgentStatus(Enum):
    """Operational state of a single agent."""
    IDLE       = auto()   # no task currently assigned
    NAVIGATING = auto()   # following a planned path toward a task target
    REPLANNING = auto()   # needs to (re)compute a path

    # EXTEND: EXECUTING_TASK, WAITING_FOR_TEAM, DOCKING, CHARGING, …


class EventType(Enum):
    """Events exchanged in the simulation loop."""
    TASK_ASSIGNED = auto()   # auctioneer gave an agent a new task
    TASK_COMPLETE = auto()   # agent's current task is finished
    PATH_BLOCKED  = auto()   # next waypoint turned out to be an obstacle
    STEP_COMPLETE = auto()   # normal single-step advance

    # EXTEND: AGENT_JOINED, COMMS_RECEIVED, BATTERY_LOW, OBJECTIVE_FOUND, …


# ===========================================================================
# Tasks
# ===========================================================================

class Task(ABC):
    """
    Abstract base class for all units of work assigned to agents.

    Subclasses must implement:
      target_loc         – where the agent should navigate toward
      check_completion() – return True the first time the task is done

    EXTEND:
      ObjectiveTask    – collect a reward item at a known location
      SurveillanceTask – loiter at a waypoint for N timesteps
      RelayTask        – establish a comms relay between two positions
      RescueTask       – reach a survivor cell and wait for handoff
    """

    _id_counter: int = 0

    def __init__(self, priority: float = 1.0) -> None:
        Task._id_counter += 1
        self.task_id:      int           = Task._id_counter
        self.priority:     float         = priority
        self.completed:    bool          = False
        self.assigned_to:  Optional[int] = None   # agent id

    @property
    @abstractmethod
    def target_loc(self) -> Tuple[int, int]: ...

    @abstractmethod
    def check_completion(self, known_map: "KnownMap") -> bool:
        """
        Called every tick by the simulation loop.
        Returns True (and sets self.completed) the first time the task
        transitions to the done state.
        Subclasses may inspect known_map to evaluate sensor-based conditions.
        """

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"id={self.task_id}, loc={self.target_loc}, done={self.completed})")


class ExplorationTask(Task):
    """
    Frontier exploration task.

    Target    : a specific UNKNOWN cell adjacent to the known frontier.
    Completion: the target cell is no longer UNKNOWN in KnownMap, meaning
                some agent's sensor footprint has swept over it.  The agent
                does not need to stand on the cell — proximity sufficient to
                observe it is enough.
    """

    def __init__(self, loc: Tuple[int, int], priority: float = 1.0) -> None:
        super().__init__(priority)
        self._target_loc = loc

    @property
    def target_loc(self) -> Tuple[int, int]:
        return self._target_loc

    def check_completion(self, known_map: "KnownMap") -> bool:
        if not self.completed:
            r, c = self._target_loc
            if known_map.state[r][c] != ObservationState.UNKNOWN:
                self.completed = True
                return True
        return False


# ===========================================================================
# Task Auctioneer
# ===========================================================================

class TaskAuctioneer:
    """
    Owns the global task queue and runs assignment rounds.

    Single-agent stub
    -----------------
    auction() selects the highest-priority unassigned task for each idle
    agent, breaking ties by Manhattan distance (closer = preferred).

    EXTEND:
      - Compute per-agent bids (e.g. 1/distance, capability score).
      - Run a greedy or Vickrey sealed-bid auction.
      - Enforce team-size or capability constraints per task type.
      - Re-auction when a task times out or an agent drops out.
      - Bundle tasks into tours (TSP) for efficiency.
    """

    def __init__(self) -> None:
        self._tasks:      List[Task]            = []
        self._known_locs: Set[Tuple[int, int]]  = set()   # dedup ExplorationTasks

    # ------------------------------------------------------------------
    def register(self, task: Task) -> bool:
        """
        Enqueue a task.  ExplorationTasks for already-registered locations
        are silently dropped (prevents duplicate frontier tasks).
        Returns True if the task was actually added.
        """
        if isinstance(task, ExplorationTask):
            if task.target_loc in self._known_locs:
                return False
            self._known_locs.add(task.target_loc)
        self._tasks.append(task)
        return True

    def add_frontier_tasks(self, known_map: "KnownMap") -> int:
        """
        Scan the known map for frontier cells — UNKNOWN cells that are
        4-adjacent to at least one FREE cell — and register an
        ExplorationTask for each new one found.
        Returns the number of newly created tasks.

        EXTEND: weight priority by distance to nearest agent, information
                gain estimate, or strategic importance of the region.
        """
        new_count = 0
        for r in range(known_map.rows):
            for c in range(known_map.cols):
                if known_map.state[r][c] != ObservationState.FREE:
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < known_map.rows
                            and 0 <= nc < known_map.cols
                            and known_map.state[nr][nc] == ObservationState.UNKNOWN):
                        if self.register(ExplorationTask((nr, nc))):
                            new_count += 1
        return new_count

    # ------------------------------------------------------------------
    @property
    def all_complete(self) -> bool:
        """True only when at least one task exists and all are finished."""
        return bool(self._tasks) and all(t.completed for t in self._tasks)

    def pending(self) -> List[Task]:
        """Tasks that are neither completed nor currently assigned."""
        return [t for t in self._tasks
                if not t.completed and t.assigned_to is None]

    def sweep_completions(self, known_map: "KnownMap") -> int:
        """
        Call check_completion on every unfinished task.  Cells that were
        observed as collateral while the agent navigated to another target
        are marked done here, before the next auction, so they are never
        assigned unnecessarily.
        Returns the number of tasks newly marked complete.
        """
        count = 0
        for task in self._tasks:
            if not task.completed and task.check_completion(known_map):
                count += 1
        return count

    def stats(self) -> str:
        total    = len(self._tasks)
        done     = sum(1 for t in self._tasks if t.completed)
        assigned = sum(1 for t in self._tasks
                       if t.assigned_to is not None and not t.completed)
        waiting  = total - done - assigned
        return f"tasks total={total} done={done} assigned={assigned} waiting={waiting}"

    # ------------------------------------------------------------------
    def auction(self, agents: List["Agent"], known_map: "KnownMap") -> None:
        """
        Assign pending tasks to idle agents.

        Current policy (single-agent stub):
          For each agent that needs a task, pick the unassigned task that
          maximises (priority, –manhattan_distance).

        EXTEND: replace with multi-agent bidding, capability matching,
                market-based clearing, or combinatorial assignment.
        """
        available = self.pending()
        if not available:
            return

        for agent in agents:
            # Skip agents that already have an active task
            if agent.current_task is not None and not agent.current_task.completed:
                continue
            if not available:
                break

            def score(t: Task) -> Tuple[float, float]:
                r0, c0 = agent.pos
                r1, c1 = t.target_loc
                return (t.priority, -(abs(r1 - r0) + abs(c1 - c0)))

            winner = max(available, key=score)
            winner.assigned_to = agent.id
            available.remove(winner)
            agent.assign_task(winner)
            print(f"  [AUCTION] Task {winner.task_id} → Agent {agent.id}"
                  f"  target={winner.target_loc}")


# ===========================================================================
# Events
# ===========================================================================

@dataclass
class Event:
    """Discrete event produced during the simulation loop."""
    kind: EventType
    data: dict = field(default_factory=dict)


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
# Visualizer
# ===========================================================================

class SimulationVisualizer:
    """
    Live matplotlib display of the exploration simulation.

    Rendering layers (bottom → top)
    --------------------------------
    1. Ground truth  – always visible; shows real terrain under the fog.
    2. Knowledge     – opaque on observed cells (white=FREE, dark=OBSTACLE);
                       transparent on UNKNOWN so the GT shows through.
    3. Fog            – semi-transparent grey on UNKNOWN cells only.
    4. Path line      – planned route from agent to task target (cyan dashes).
    5. Task marker    – gold star on the current task target cell.
    6. Agent marker   – green circle on the agent's current position.

    The effect: unexplored cells reveal the underlying terrain through a grey
    mist; explored cells appear cleanly mapped.
    """

    # ── Palette ──────────────────────────────────────────────────────────
    _GT_FREE     = (0.85, 0.93, 0.85, 1.0)   # pale green  – GT free cell
    _GT_OBS      = (0.12, 0.12, 0.12, 1.0)   # near-black  – GT obstacle
    _KN_FREE     = (1.00, 1.00, 1.00, 1.0)   # white       – known free
    _KN_OBS      = (0.22, 0.22, 0.22, 1.0)   # charcoal    – known obstacle
    _FOG_COLOUR  = (0.55, 0.55, 0.55, 0.62)  # grey fog    – RGBA

    def __init__(self, ground_truth: "GroundTruthMap") -> None:
        self._gt   = ground_truth
        self._rows = ground_truth.rows
        self._cols = ground_truth.cols

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.patch.set_facecolor("#1a1a2e")  # type: ignore[union-attr]
        self.ax.set_facecolor("#1a1a2e")
        self.fig.canvas.manager.set_window_title("DARPA Exploration Sim")  # type: ignore[union-attr]

        # Axes: x = col index, y = row index (row 0 at top)
        self.ax.set_xlim(-0.5, self._cols - 0.5)
        self.ax.set_ylim(self._rows - 0.5, -0.5)   # inverted so row 0 is top
        self.ax.set_aspect("equal")

        # Subtle grid lines at cell boundaries
        self.ax.set_xticks(np.arange(-0.5, self._cols, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self._rows, 1), minor=True)
        self.ax.grid(which="minor", color="#2a2a4a", linewidth=0.6, zorder=0)
        self.ax.tick_params(which="both", bottom=False, left=False,
                            labelbottom=False, labelleft=False)

        # imshow extent maps pixel centres to cell centres
        ext = [-0.5, self._cols - 0.5, self._rows - 0.5, -0.5]

        # Layer 1: ground truth (static — built once)
        self._gt_img = self._make_gt_image()
        self.ax.imshow(self._gt_img, origin="upper", extent=ext,
                       interpolation="nearest", zorder=1)

        # Layer 2: knowledge overlay (dynamic — opaque for known cells)
        self._know_data = np.zeros((self._rows, self._cols, 4))
        self._know_im = self.ax.imshow(self._know_data, origin="upper",
                                       extent=ext, interpolation="nearest",
                                       zorder=2)

        # Layer 3: fog (dynamic — semi-transparent grey on UNKNOWN)
        self._fog_data = np.zeros((self._rows, self._cols, 4))
        self._fog_im = self.ax.imshow(self._fog_data, origin="upper",
                                      extent=ext, interpolation="nearest",
                                      zorder=3)

        # Path line
        self._path_line, = self.ax.plot(
            [], [], "--", color="#00cfff", linewidth=1.8,
            alpha=0.85, zorder=4, label="Planned path")

        # Task target star
        self._task_dot, = self.ax.plot(
            [], [], "*", color="#ffd700", markersize=18,
            markeredgecolor="#7a5200", markeredgewidth=0.8,
            zorder=5, label="Task target")

        # Agent circle
        self._agent_dot, = self.ax.plot(
            [], [], "o", color="#00ff88", markersize=13,
            markeredgecolor="black", markeredgewidth=1.0,
            zorder=6, label="Agent")

        self._title = self.ax.set_title(
            "Initialising…", color="white", fontsize=10, pad=8)

        self._build_legend()
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.01)

    # ── Image builders ───────────────────────────────────────────────────

    def _make_gt_image(self) -> np.ndarray:
        img = np.ones((self._rows, self._cols, 4))
        for r in range(self._rows):
            for c in range(self._cols):
                img[r, c] = (self._GT_OBS if self._gt.is_obstacle((r, c))
                             else self._GT_FREE)
        return img

    def _update_knowledge_layer(self, known_map: "KnownMap") -> None:
        for r in range(self._rows):
            for c in range(self._cols):
                s = known_map.state[r][c]
                if s == ObservationState.FREE:
                    self._know_data[r, c] = self._KN_FREE
                elif s == ObservationState.OBSTACLE:
                    self._know_data[r, c] = self._KN_OBS
                else:
                    self._know_data[r, c] = (0, 0, 0, 0)   # transparent → GT visible
        self._know_im.set_data(self._know_data)

    def _update_fog_layer(self, known_map: "KnownMap") -> None:
        for r in range(self._rows):
            for c in range(self._cols):
                self._fog_data[r, c] = (
                    self._FOG_COLOUR
                    if known_map.state[r][c] == ObservationState.UNKNOWN
                    else (0, 0, 0, 0)
                )
        self._fog_im.set_data(self._fog_data)

    # ── Legend ───────────────────────────────────────────────────────────

    def _build_legend(self) -> None:
        elements = [
            mpatches.Patch(facecolor=self._KN_FREE[:3],
                           edgecolor="#aaa", label="Free (mapped)"),
            mpatches.Patch(facecolor=self._KN_OBS[:3],
                           edgecolor="#aaa", label="Obstacle (mapped)"),
            mpatches.Patch(facecolor=self._GT_FREE[:3], alpha=0.55,
                           edgecolor="#888", label="Free (unknown)"),
            mpatches.Patch(facecolor=self._GT_OBS[:3],  alpha=0.55,
                           edgecolor="#888", label="Obstacle (unknown)"),
            Line2D([0], [0], color="#00cfff", linestyle="--",
                   linewidth=1.8, label="Planned path"),
            Line2D([0], [0], marker="*", color="#ffd700",
                   markeredgecolor="#7a5200", markersize=13,
                   linestyle="None", label="Task target"),
            Line2D([0], [0], marker="o", color="#00ff88",
                   markeredgecolor="black", markersize=10,
                   linestyle="None", label="Agent"),
        ]
        legend = self.ax.legend(
            handles=elements, loc="upper right", fontsize=7.5,
            facecolor="#222244", edgecolor="#555577", labelcolor="white")
        legend.set_zorder(10)

    # ── Public interface ─────────────────────────────────────────────────

    def update(
        self,
        known_map:  "KnownMap",
        agents:     List["Agent"],
        step:       int,
        task_stats: str,
    ) -> None:
        """Redraw the display for one simulation step."""
        self._update_knowledge_layer(known_map)
        self._update_fog_layer(known_map)

        # Path (first agent only; extend loop for multi-agent)
        agent = agents[0]
        if len(agent.path) > 1:
            self._path_line.set_data(
                [p[1] for p in agent.path],   # col → x
                [p[0] for p in agent.path])   # row → y
        else:
            self._path_line.set_data([], [])

        # Task target
        if agent.current_task and not agent.current_task.completed:
            tr, tc = agent.current_task.target_loc
            self._task_dot.set_data([tc], [tr])
        else:
            self._task_dot.set_data([], [])

        # Agent position
        self._agent_dot.set_data([agent.pos[1]], [agent.pos[0]])

        # Caption
        self._title.set_text(f"Step {step}   |   {task_stats}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def finalize(self, task_stats: str) -> None:
        """Mark simulation complete and block until the window is closed."""
        self._title.set_text(f"COMPLETE   |   {task_stats}")
        self.fig.canvas.draw()
        plt.ioff()
        plt.show()


# ===========================================================================
# Pathfinding  (self-contained A* — no external dependency)
# ===========================================================================

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # N E S W


def _move(loc: Tuple[int, int], d: int) -> Tuple[int, int]:
    return loc[0] + _DIRS[d][0], loc[1] + _DIRS[d][1]


def _in_bounds(grid: List[List[bool]], loc: Tuple[int, int]) -> bool:
    return 0 <= loc[0] < len(grid) and 0 <= loc[1] < len(grid[0])


def _dijkstra_heuristic(
    grid: List[List[bool]], goal: Tuple[int, int]
) -> Dict[Tuple[int, int], int]:
    """
    Backward Dijkstra from goal.  UNKNOWN cells (grid value False) are
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


def plan_path(
    known_map: KnownMap,
    start:     Tuple[int, int],
    goal:      Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    A* on the known map.  UNKNOWN cells are treated as FREE (optimistic).
    Returns a list of (row, col) positions from start to goal, or None.
    """
    grid  = known_map.to_obstacle_grid()
    h_val = _dijkstra_heuristic(grid, goal)
    if start not in h_val:
        return None   # goal unreachable on current known map

    closed: Dict[Tuple[int, int], dict] = {}
    root = {'loc': start, 'g': 0, 'h': h_val.get(start, 0), 'parent': None}
    counter = 0   # unique tiebreaker so dicts are never compared by heapq
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
            if (not _in_bounds(grid, nb)
                    or not known_map.is_passable(nb)
                    or nb in closed):
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
    """
    Autonomous agent driven entirely by Task objects; no fixed goal location.
    Finishes when the auctioneer has no more tasks to assign.

    Sensor model : axis-aligned square footprint, side = 2*obs_radius + 1.
    Nav policy   : A* to current_task.target_loc on the current KnownMap.

    EXTEND: battery model, comms range, payload, team_id, sensor cone, …
    """

    def __init__(
        self,
        agent_id:   int,
        start:      Tuple[int, int],
        obs_radius: int = 2,
    ) -> None:
        self.id            = agent_id
        self.pos           = start
        self.obs_radius    = obs_radius
        self.status        = AgentStatus.IDLE
        self.path:         List[Tuple[int, int]] = []
        self.current_task: Optional[Task]        = None

        # EXTEND: battery, comms_range, payload, team_id, …

    # ------------------------------------------------------------------
    def assign_task(self, task: Task) -> None:
        """Called by the auctioneer; puts the agent into REPLANNING state."""
        self.current_task = task
        self.path         = []
        self.status       = AgentStatus.REPLANNING

    # ------------------------------------------------------------------
    def observe(self, ground_truth: GroundTruthMap, known_map: KnownMap) -> None:
        """
        Reveal all cells within obs_radius; update KnownMap in-place.
        EXTEND: line-of-sight, sensor cone, range-dependent noise, …
        """
        r, c = self.pos
        for dr in range(-self.obs_radius, self.obs_radius + 1):
            for dc in range(-self.obs_radius, self.obs_radius + 1):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < ground_truth.rows and 0 <= nc < ground_truth.cols):
                    continue
                loc = (nr, nc)
                new_state = (ObservationState.OBSTACLE
                             if ground_truth.is_obstacle(loc)
                             else ObservationState.FREE)
                known_map.update(loc, new_state)

    # ------------------------------------------------------------------
    def replan(self, known_map: KnownMap) -> bool:
        """
        Compute a path to current_task.target_loc via A*.
        Returns True if a valid path was found.

        EXTEND: chain tasks (TSP ordering), apply CBS constraints,
                coordinate with teammates, honour dynamic deadlines, …
        """
        if self.current_task is None:
            self.status = AgentStatus.IDLE
            return False

        path = plan_path(known_map, self.pos, self.current_task.target_loc)
        if path:
            self.path   = path
            self.status = AgentStatus.NAVIGATING
            return True

        # Target unreachable on current known map — wait for next auction
        self.status = AgentStatus.IDLE
        return False

    # ------------------------------------------------------------------
    def step(self, known_map: KnownMap) -> Optional[Event]:
        """
        Move one cell along the planned path.
        Returns an Event or None.
        """
        if self.status in (AgentStatus.IDLE, AgentStatus.REPLANNING):
            return None
        if len(self.path) < 2:
            return None

        next_pos = self.path[1]
        if not known_map.is_passable(next_pos):
            self.path   = []
            self.status = AgentStatus.IDLE
            return Event(EventType.PATH_BLOCKED,
                         {'agent': self.id, 'blocked_at': next_pos})

        self.pos  = next_pos
        self.path = self.path[1:]
        return Event(EventType.STEP_COMPLETE, {'agent': self.id, 'pos': self.pos})


# ===========================================================================
# Scenario factory
# ===========================================================================

def build_default_scenario() -> Tuple[GroundTruthMap, int, int]:
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

    Agent starts at (0, 0).

    EXTEND: load from file, randomise obstacles, add multiple agents, …
    """
    rows, cols = 10, 10
    obstacle_locs = {(1, 3), (2, 3), (3, 3), (6, 7), (7, 7)}
    grid = [
        [((r, c) in obstacle_locs) for c in range(cols)]
        for r in range(rows)
    ]
    return GroundTruthMap(grid), rows, cols


# ===========================================================================
# Simulation event loop
# ===========================================================================

def run_simulation(max_steps: int = 200, verbose: bool = True) -> KnownMap:
    """
    Main event loop.  Terminates when the task queue is exhausted.

    Per-timestep order:
      1. Each agent observes surroundings            → KnownMap updated
      2. Auctioneer scans for new frontier cells     → ExplorationTasks added
      3. Active tasks checked for passive completion → agents may go IDLE
      4. Auctioneer runs an auction round            → IDLE agents get tasks
      5. Agents that need paths compute them         → status → NAVIGATING
      6. Agents step along their paths               → handle PATH_BLOCKED
      7. Termination check

    Returns the final KnownMap (useful for post-run analysis).

    EXTEND: shared KnownMap with comms, CBS conflict resolution,
            multiple agents with capability constraints, visualisation, …
    """
    ground_truth, rows, cols = build_default_scenario()
    known_map  = KnownMap(rows, cols)
    auctioneer = TaskAuctioneer()

    start  = (0, 0)
    agents = [Agent(agent_id=0, start=start, obs_radius=2)]

    print("=" * 52)
    print("  DARPA Exploration Simulation  (task-queue driven)")
    print(f"  Agent 0 starts at {start}   map {rows}×{cols}")
    print("=" * 52)

    vis = SimulationVisualizer(ground_truth)

    # ── Bootstrap ────────────────────────────────────────────────────────
    for agent in agents:
        agent.observe(ground_truth, known_map)
    auctioneer.add_frontier_tasks(known_map)
    auctioneer.auction(agents, known_map)
    for agent in agents:
        if agent.status == AgentStatus.REPLANNING:
            agent.replan(known_map)

    # ── Main loop ────────────────────────────────────────────────────────
    for step in range(max_steps):

        # Display ────────────────────────────────────────────────────────
        if verbose:
            statuses = "  ".join(
                f"A{a.id}@{a.pos}[{a.status.name[0]}]" for a in agents
            )
            print(f"\n─── Step {step:3d}  {statuses}  {auctioneer.stats()} ───")

        vis.update(known_map, agents, step, auctioneer.stats())

        # 1. Observe ─────────────────────────────────────────────────────
        for agent in agents:
            agent.observe(ground_truth, known_map)

        # 2. Discover new frontier tasks ─────────────────────────────────
        new_tasks = auctioneer.add_frontier_tasks(known_map)
        if new_tasks and verbose:
            print(f"  [FRONTIER] +{new_tasks} exploration task(s) queued")

        # 2b. Sweep all queued tasks for collateral observations ──────────
        # Cells passed through en-route to another target may already be
        # observed; mark them done now so they are never auctioned.
        swept = auctioneer.sweep_completions(known_map)
        if swept and verbose:
            print(f"  [SWEPT]   {swept} task(s) observed as collateral")

        # 3. Check whether any agent's current task was just completed ────
        # Use .completed directly — sweep_completions already called
        # check_completion (which only fires once) for all tasks.
        for agent in agents:
            if agent.current_task and agent.current_task.completed:
                print(f"  [TASK ✓] Agent {agent.id} finished task"
                      f" {agent.current_task.task_id}"
                      f"  (observed {agent.current_task.target_loc})")
                agent.current_task = None
                agent.path         = []
                agent.status       = AgentStatus.IDLE

        # 4. Auction ─────────────────────────────────────────────────────
        auctioneer.auction(agents, known_map)

        # 5. Replan ──────────────────────────────────────────────────────
        for agent in agents:
            if agent.status == AgentStatus.REPLANNING:
                if not agent.replan(known_map) and verbose:
                    print(f"  [WARN] Agent {agent.id} cannot reach task target yet")

        # 6. Step ────────────────────────────────────────────────────────
        for agent in agents:
            ev = agent.step(known_map)
            if ev is None:
                continue
            if ev.kind == EventType.PATH_BLOCKED:
                blocked_at = ev.data['blocked_at']
                print(f"  [BLOCKED] Agent {agent.id} — cell {blocked_at}"
                      f" is obstacle; releasing task back to queue")
                if agent.current_task:
                    agent.current_task.assigned_to = None
                    agent.current_task = None
                auctioneer.auction(agents, known_map)
                if agent.status == AgentStatus.REPLANNING:
                    agent.replan(known_map)

        # 7. Termination ─────────────────────────────────────────────────
        if auctioneer.all_complete and all(a.status == AgentStatus.IDLE
                                           for a in agents):
            vis.update(known_map, agents, step, auctioneer.stats())
            print(f"\n[DONE] All {auctioneer.stats()} — finished in {step + 1} steps.")
            break

    else:
        print(f"\n[TIMEOUT] Simulation ended after {max_steps} steps.")

    vis.finalize(auctioneer.stats())
    return known_map


# ===========================================================================
if __name__ == "__main__":
    run_simulation(max_steps=200, verbose=True)
