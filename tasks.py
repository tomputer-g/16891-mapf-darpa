"""
Tasks and task queue for the DARPA exploration simulation.

  Task            – abstract base class for all assignable work items
  ExplorationTask – navigate to and observe an UNKNOWN frontier cell
  TaskAuctioneer  – owns the queue; assigns work via auction rounds
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from sim_types import AgentType, ObservationState

if TYPE_CHECKING:
    from agents import Agent
    from maps import KnownMap


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

class TriageTask(Task):
    def __init__(self, loc: Tuple[int, int], priority: float = 2.0,
                 dwell_steps: int = 2, ground_only: bool = False) -> None:
        super().__init__(priority)
        self._target_loc = loc
        self.dwell_steps = dwell_steps
        self.progress = 0
        self.ground_only = ground_only

    @property
    def target_loc(self) -> Tuple[int, int]:
        return self._target_loc

    def check_completion(self, known_map: "KnownMap") -> bool:
        return self.completed


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
        self._triage_locs: Set[Tuple[int, int]] = set()   # dedup TriageTasks
        self._invest_locs: Set[Tuple[int, int]] = set()   # dedup building investigations

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
    
    def add_revealed_triage_tasks(self, known_map: "KnownMap", ground_truth) -> int:
        """Create tasks for revealed objectives and buildings.

        - Free-standing objectives get a TriageTask immediately.
        - Buildings get a ground-only investigation TriageTask so a ground
          robot is dispatched to check occupancy.  Actual triage tasks for
          occupied buildings are created later by
          add_confirmed_building_triage().
        Returns the number of newly created tasks.
        """
        new_count = 0
        # Free-standing objectives
        for loc in ground_truth.objectives:
            r, c = loc
            if known_map.state[r][c] == ObservationState.OBJECTIVE and loc not in self._triage_locs:
                task = TriageTask(loc)
                self._triage_locs.add(loc)
                self._tasks.append(task)
                new_count += 1
        # Buildings -- send ground robot to investigate (occupancy unknown)
        for loc in ground_truth.buildings:
            r, c = loc
            state = known_map.state[r][c]
            if state in (ObservationState.BUILDING, ObservationState.OCCUPIED_BUILDING) and loc not in self._invest_locs:
                task = TriageTask(loc, ground_only=True, dwell_steps=1)
                task._is_investigation = True
                self._invest_locs.add(loc)
                self._tasks.append(task)
                new_count += 1
        return new_count

    def add_confirmed_building_triage(self, known_map: "KnownMap") -> int:
        """After a ground robot has investigated a building and the known
        map shows OCCUPIED_BUILDING, create an actual triage task.
        Only promotes once the investigation task for that location is done.
        Returns the number of newly created tasks.
        """
        # Build set of locations whose investigation is complete
        done_investigations: Set[Tuple[int, int]] = set()
        for t in self._tasks:
            if (isinstance(t, TriageTask)
                    and getattr(t, '_is_investigation', False)
                    and t.completed):
                done_investigations.add(t.target_loc)

        new_count = 0
        for loc in list(self._invest_locs):
            if loc not in done_investigations:
                continue
            r, c = loc
            if known_map.state[r][c] == ObservationState.OCCUPIED_BUILDING and loc not in self._triage_locs:
                task = TriageTask(loc, ground_only=True)
                self._triage_locs.add(loc)
                self._tasks.append(task)
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

            # Filter tasks this agent is eligible for
            eligible = [t for t in available
                        if not (isinstance(t, TriageTask) and t.ground_only
                                and agent.agent_type != AgentType.GROUND)]
            if not eligible:
                continue

            def score(t: Task) -> Tuple[float, float]:
                r0, c0 = agent.pos
                r1, c1 = t.target_loc
                return (t.priority, -(abs(r1 - r0) + abs(c1 - c0)))

            winner = max(eligible, key=score)
            winner.assigned_to = agent.id
            available.remove(winner)
            agent.assign_task(winner)
            print(f"  [AUCTION] Task {winner.task_id} -> Agent {agent.id}"
                  f"  target={winner.target_loc}")
