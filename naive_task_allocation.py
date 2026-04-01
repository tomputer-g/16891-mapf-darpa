"""
Auction and task allocation for the DARPA exploration simulation.

Availability-Constrained Sequential Single-Item Auction with Global
Reauction Trigger
---------------------------------------------------------------
Tasks are auctioned one at a time. Only robots that are currently
available, meaning they do not hold an active incomplete task, may bid.
Each bid is the marginal motion cost to reach the task target on the
current known map. The lowest feasible bid wins.

Because the environment is only partially known, assignments are monitored
continuously. If a path becomes infeasible or a simple inter-robot motion
conflict is detected, a global reauction is triggered. All incomplete
non-executing work is released, agent-task assignments are cleared, and a
new auction round is run from the robots' current states.

This gives a lightweight event-driven allocator for dynamic exploration
without requiring a full multi-agent constraint tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from agents import Agent
from maps import KnownMap, ObservationState
from tasks import ExplorationTask, Task

# ===========================================================================
# Task Auctioneer
# ===========================================================================

class NaiveTaskAuctioneer:
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

        idle_agents = [a for a in agents
                       if a.current_task is None or a.current_task.completed]
        if not idle_agents:
            return

        for task in available:
            if not idle_agents:
                break

            def score(a: "Agent", t: Task = task) -> Tuple[float, float]:
                r0, c0 = a.pos
                r1, c1 = t.target_loc
                return (t.priority, -(abs(r1 - r0) + abs(c1 - c0)))

            winner = max(idle_agents, key=score)
            task.assigned_to = winner.id
            idle_agents.remove(winner)
            winner.assign_task(task)
            print(f"  [AUCTION] Task {task.task_id} → Agent {winner.id}"
                  f"  target={task.target_loc}")
