"""
Auction and task allocation for the DARPA exploration simulation.

Greedy queue-ordered baseline allocator.

This module is the live naive baseline used by `main.py`.

- pending tasks are considered in queue order
- eligible idle agents are scored by `(task.priority, -ManhattanDistance)`
- ground-only triage tasks exclude drones
- blocked paths are handled in `main.py` by releasing the task back to the
  pending pool; this module does not implement the SSIA-family
  repair-vs-reauction heuristic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.agents import Agent, DroneAgent
from src.maps import KnownMap, ObservationState
from src.planner import CBS
from src.sim_types import AgentType, AgentStatus
from src.tasks import ExplorationTask, Task, TriageTask

# ===========================================================================
# Task Auctioneer
# ===========================================================================

class NaiveTaskAuctioneer:
    """
    Greedy baseline auctioneer used by `main.py`.

    EXTEND:
      - Compute per-agent bids (e.g. 1/distance, capability score).
      - Run a greedy or Vickrey sealed-bid auction.
      - Enforce team-size or capability constraints per task type.
      - Re-auction when a task times out or an agent drops out.
      - Bundle tasks into tours (TSP) for efficiency.
    """

    def __init__(self, cbs: Optional[CBS] = None) -> None:
        self._tasks:      List[Task]            = []
        self._known_locs: Set[Tuple[int, int]]  = set()   # dedup ExplorationTasks
        self._triage_locs: Set[Tuple[int, int]] = set()   # dedup TriageTasks
        self._invest_locs: Set[Tuple[int, int]] = set()   # dedup building investigations
        self._cbs: Optional[CBS] = cbs

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

    def add_revealed_triage_tasks(self, known_map: "KnownMap", ground_truth) -> int:
        """Create tasks for revealed objectives and buildings."""
        new_count = 0
        for loc in ground_truth.objectives:
            r, c = loc
            if known_map.state[r][c] == ObservationState.OBJECTIVE and loc not in self._triage_locs:
                task = TriageTask(loc)
                self._triage_locs.add(loc)
                self._tasks.append(task)
                new_count += 1
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
        """Create triage tasks for buildings confirmed occupied."""
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

        idle_agents = [a for a in agents
                       if a.current_task is None or a.current_task.completed]
        if not idle_agents:
            return

        for task in available:
            if not idle_agents:
                break

            # Filter out drones for ground_only tasks
            eligible = [a for a in idle_agents
                        if not (isinstance(task, TriageTask) and task.ground_only
                                and a.agent_type != AgentType.GROUND)]
            if not eligible:
                continue

            def score(a: "t", t: Task = task) -> Tuple[float, float]:
                r0, c0 = a.pos
                r1, c1 = t.target_loc
                return (t.priority, -(abs(r1 - r0) + abs(c1 - c0)))

            winner = max(eligible, key=score)
            task.assigned_to = winner.id
            idle_agents.remove(winner)
            winner.assign_task(task)
            print(f"  [AUCTION] Task {task.task_id} -> Agent {winner.id}"
                  f"  target={task.target_loc}")
        # CBS multi-agent replan for all navigating ground agents
        self._cbs_replan_ground(agents, known_map)

    def _cbs_replan_ground(self, agents: List["Agent"], known_map: "KnownMap") -> None:
        """Run CBS jointly for all navigating ground agents to resolve collisions."""
        if self._cbs is None:
            return

        ground_nav = [
            a for a in agents
            if not isinstance(a, DroneAgent)
            and a.status == AgentStatus.NAVIGATING
            and a.current_task is not None
        ]
        if len(ground_nav) < 2:
            return

        starts = {a.id: a.pos for a in ground_nav}
        goals = {a.id: a.current_task.target_loc for a in ground_nav}

        paths = self._cbs.plan(starts, goals, known_map, drone=False)
        if paths is None:
            return

        for a in ground_nav:
            if a.id in paths:
                a.path = paths[a.id]
                self._cbs.set_path(a.id, paths[a.id])
