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

from agents import Agent, AgentStatus, Agent_Drone, plan_path
from maps import KnownMap, ObservationState
from tasks import ExplorationTask, Task


@dataclass
class Bid:
    """Single sealed bid for one task from one agent."""
    agent_id: int
    task_id: int
    cost: int
    path: List[Tuple[int, int]]


class SequentialSingleItemAuctioneer:
    """
    Availability-constrained sequential single-item auctioneer.
    """

    def __init__(self) -> None:
        self._tasks: List[Task] = []
        self._known_locs: Set[Tuple[int, int]] = set()
        self.reauction_count: int = 0

    # ------------------------------------------------------------------
    # Task bookkeeping
    # ------------------------------------------------------------------
    def register(self, task: Task) -> bool:
        """
        Add a task to the queue.

        Exploration tasks are deduplicated by target location so the same
        frontier cell is not repeatedly added.
        """
        if isinstance(task, ExplorationTask):
            if task.target_loc in self._known_locs:
                return False
            self._known_locs.add(task.target_loc)
        self._tasks.append(task)
        return True

    def add_frontier_tasks(self, known_map: KnownMap) -> int:
        """
        Generate one ExplorationTask for each newly discovered frontier cell.

        A frontier cell is an UNKNOWN cell that is 4-adjacent to at least one
        known FREE cell.
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

    @property
    def all_complete(self) -> bool:
        return bool(self._tasks) and all(t.completed for t in self._tasks)

    def pending(self) -> List[Task]:
        return [t for t in self._tasks
                if not t.completed and t.assigned_to is None]

    def incomplete(self) -> List[Task]:
        return [t for t in self._tasks if not t.completed]

    def sweep_completions(self, known_map: KnownMap) -> int:
        count = 0
        for task in self._tasks:
            if not task.completed and task.check_completion(known_map):
                count += 1
        return count

    def stats(self) -> str:
        total = len(self._tasks)
        done = sum(1 for t in self._tasks if t.completed)
        assigned = sum(1 for t in self._tasks
                       if t.assigned_to is not None and not t.completed)
        waiting = total - done - assigned
        return (
            f"tasks total={total} done={done} assigned={assigned} "
            f"waiting={waiting} reauctions={self.reauction_count}"
        )

    # ------------------------------------------------------------------
    # Bidding and assignment
    # ------------------------------------------------------------------
    def available_agents(self, agents: Iterable[Agent]) -> List[Agent]:
        """
        Agents may bid only when they are not committed to another incomplete
        task. This is the availability constraint.
        """
        available: List[Agent] = []
        for agent in agents:
            if agent.current_task is None:
                available.append(agent)
                continue
            if agent.current_task.completed:
                available.append(agent)
                continue
        return available

    def compute_bid(
        self,
        agent: Agent,
        task: Task,
        known_map: KnownMap,
    ) -> Optional[Bid]:
        """
        Bid equals marginal motion cost to reach the task from the agent's
        current state on the current known map.

        Drones use drone path planning so they can fly over obstacles.
        Ground agents use normal path planning.
        """
        use_drone_path = isinstance(agent, Agent_Drone)

        path = plan_path(
            known_map,
            agent.pos,
            task.target_loc,
            drone=use_drone_path,
        )
        if not path:
            return None

        cost = max(0, len(path) - 1)
        return Bid(
            agent_id=agent.id,
            task_id=task.task_id,
            cost=cost,
            path=path,
        )

    def auction(self, agents: List[Agent], known_map: KnownMap) -> None:
        """
        Run a sequential single-item auction.
        """
        available_tasks = self.pending()
        if not available_tasks:
            return

        available_agents = self.available_agents(agents)
        if not available_agents:
            return

        for task in available_tasks:
            if not available_agents:
                break

            bids: List[Bid] = []
            for agent in available_agents:
                bid = self.compute_bid(agent, task, known_map)
                if bid is not None:
                    bids.append(bid)

            if not bids:
                continue

            winner = min(bids, key=lambda b: (b.cost, b.agent_id))
            winning_agent = next(a for a in available_agents if a.id == winner.agent_id)

            task.assigned_to = winning_agent.id
            winning_agent.assign_task(task)
            winning_agent.path = winner.path
            winning_agent.status = AgentStatus.NAVIGATING
            available_agents.remove(winning_agent)

            print(
                f"  [AUCTION] Task {task.task_id} -> Agent {winning_agent.id} "
                f"target={task.target_loc} bid_cost={winner.cost}"
            )

    # ------------------------------------------------------------------
    # Reauction support
    # ------------------------------------------------------------------
    def _path_infeasible(self, agent: Agent, known_map: KnownMap) -> bool:
        """
        Detect whether the agent's current assignment is no longer feasible.

        Ground agents:
        - blocked if next planned cell is now a known obstacle
        - blocked if replanning fails

        Drones:
        - obstacles do not block flight
        - blocked only if replanning fails
        """
        if agent.current_task is None or agent.current_task.completed:
            return False

        use_drone_path = isinstance(agent, Agent_Drone)

        if (
            not use_drone_path
            and len(agent.path) >= 2
            and not known_map.is_passable(agent.path[1])
        ):
            return True

        replanned = plan_path(
            known_map,
            agent.pos,
            agent.current_task.target_loc,
            drone=use_drone_path,
        )
        return replanned is None

    def _next_move_conflict(self, agents: List[Agent]) -> bool:
        """
        Lightweight inter-robot conflict detection.

        Triggers when two robots intend to occupy the same next cell or swap
        cells on the next move.
        """
        next_pos: Dict[int, Tuple[int, int]] = {}
        curr_pos: Dict[int, Tuple[int, int]] = {a.id: a.pos for a in agents}

        active_agents = []
        for agent in agents:
            if agent.status != AgentStatus.NAVIGATING or len(agent.path) < 2:
                continue
            next_pos[agent.id] = agent.path[1]
            active_agents.append(agent)

        seen: Set[Tuple[int, int]] = set()
        for pos in next_pos.values():
            if pos in seen:
                return True
            seen.add(pos)

        for i in range(len(active_agents)):
            for j in range(i + 1, len(active_agents)):
                a = active_agents[i]
                b = active_agents[j]
                if next_pos[a.id] == curr_pos[b.id] and next_pos[b.id] == curr_pos[a.id]:
                    return True
        return False

    def should_trigger_reauction(self, agents: List[Agent], known_map: KnownMap) -> bool:
        """
        Check for global reauction conditions.
        """
        for agent in agents:
            if self._path_infeasible(agent, known_map):
                return True
        return self._next_move_conflict(agents)

    def trigger_global_reauction(self, agents: List[Agent], known_map: KnownMap) -> None:
        """
        Release all incomplete assignments and run a fresh auction round.
        """
        self.reauction_count += 1

        for task in self._tasks:
            if not task.completed:
                task.assigned_to = None

        for agent in agents:
            agent.current_task = None
            agent.path = []
            agent.status = AgentStatus.IDLE

        print(f"  [REAUCTION] Global reauction triggered #{self.reauction_count}")
        self.auction(agents, known_map)

    # ------------------------------------------------------------------
    # Convenience update hook for the main loop
    # ------------------------------------------------------------------
    def update(self, agents: List[Agent], known_map: KnownMap) -> None:
        """
        One allocator update.
        """
        self.add_frontier_tasks(known_map)
        self.sweep_completions(known_map)

        for agent in agents:
            if agent.current_task and agent.current_task.completed:
                agent.current_task = None
                agent.path = []
                agent.status = AgentStatus.IDLE

        if self.should_trigger_reauction(agents, known_map):
            self.trigger_global_reauction(agents, known_map)
        else:
            self.auction(agents, known_map)