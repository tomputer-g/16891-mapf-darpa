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

from typing import List, Optional, Tuple

from src.agents import Agent, AgentStatus, DroneAgent
from src.maps import KnownMap, ObservationState
from src.planner import CBS
from src.sim_types import AgentType
from src.tasks import ExplorationTask, Task, TriageTask
from src.allocators.base_task_allocation import BaseTaskAuctioneer


class NaiveTaskAuctioneer(BaseTaskAuctioneer):
    """
    Greedy baseline auctioneer used by `main.py`.

    Inherits task registration, frontier/triage discovery, completion sweeping,
    and basic state accessors from BaseTaskAuctioneer.

    EXTEND:
      - Compute per-agent bids (e.g. 1/distance, capability score).
      - Run a greedy or Vickrey sealed-bid auction.
      - Enforce team-size or capability constraints per task type.
      - Re-auction when a task times out or an agent drops out.
      - Bundle tasks into tours (TSP) for efficiency.
    """

    # ------------------------------------------------------------------
    # add_revealed_triage_tasks: returns int (total new tasks)
    # ------------------------------------------------------------------
    def add_revealed_triage_tasks(self, known_map: KnownMap, ground_truth) -> int:  # type: ignore[override]
        """Create tasks for revealed objectives and buildings. Returns total new count."""
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

    def stats(self) -> str:
        total    = len(self._tasks)
        done     = sum(1 for t in self._tasks if t.completed)
        assigned = sum(1 for t in self._tasks
                       if t.assigned_to is not None and not t.completed)
        waiting  = total - done - assigned
        return f"tasks total={total} done={done} assigned={assigned} waiting={waiting}"

    # ------------------------------------------------------------------
    def auction(self, agents: List[Agent], known_map: KnownMap) -> None:
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

            eligible = [a for a in idle_agents
                        if not (isinstance(task, TriageTask) and task.ground_only
                                and a.agent_type != AgentType.GROUND)]
            if not eligible:
                continue

            def score(a: Agent, t: Task = task) -> Tuple[float, float]:
                r0, c0 = a.pos
                r1, c1 = t.target_loc
                return (t.priority, -(abs(r1 - r0) + abs(c1 - c0)))

            winner = max(eligible, key=score)
            task.assigned_to = winner.id
            idle_agents.remove(winner)
            winner.assign_task(task)
            print(f"  [AUCTION] Task {task.task_id} -> Agent {winner.id}"
                  f"  target={task.target_loc}")
        self._cbs_replan_ground(agents, known_map)

    def _cbs_replan_ground(self, agents: List[Agent], known_map: KnownMap) -> None:  # type: ignore[override]
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
