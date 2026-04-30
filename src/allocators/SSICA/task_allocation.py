"""
Auction and task allocation for the DARPA exploration simulation.

Sequential Single-Item Concurrent Auction (SSICA)
---------------------------------------------------------------
All agents bid on every task, even if they already have tasks in their
queue.  Each agent maintains an ordered task queue.  The bid for a new
task equals the cumulative cost of all tasks already in the queue PLUS
the path cost from the last queued task's target to the new task's
target.  The highest-scoring bid wins; the task is appended to the
winner's queue.

Because the environment is only partially known, assignments are monitored
continuously. Only the active head task stores repair fallback metadata, so
an invalidated path first goes through a repair-vs-reauction check. Full
reauction preserves only an active on-target dwell task that is already in
progress; queued tail work is cleared.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from src.agents import Agent, AgentStatus, DroneAgent, plan_path
from src.maps import KnownMap, ObservationState
from src.planner import CBS
from src.sim_types import AgentType
from src.tasks import AssignmentSnapshot, ExplorationTask, TriageTask, Task
from src.allocators.base_task_allocation import (
    BaseSSIATaskAuctioneer, Bid, _TRIAGE_DWELL, _REPAIR_METRIC_SLACK,
)


_TASK_REWARD = {
    ExplorationTask: 1,
    TriageTask: 8,
}


class SequentialSingleItemAuctioneer(BaseSSIATaskAuctioneer):
    """
    Sequential single-item concurrent auctioneer with per-agent task queues.

    Every agent bids on every unassigned task.  Bid cost accounts for all
    tasks already in the agent's queue: cumulative queue cost + path from
    the last queued target to the new task target.

    Inherits task bookkeeping, execution-cost helpers, CBS replanning,
    conflict detection, and the repair heuristic from BaseSSIATaskAuctioneer.
    Overrides sweep_completions, handle_invalidated_assignment,
    trigger_global_reauction, and update to manage per-agent queues.
    """

    _task_reward = _TASK_REWARD

    def __init__(self, cbs: Optional[CBS] = None) -> None:
        super().__init__(cbs)
        # Per-agent task queue: agent_id -> ordered list of Tasks.
        #
        # SSICA is "concurrent" because agents are allowed to win more than one
        # task in the same auction round. That only works if each agent can
        # remember future commitments, so each agent owns a queue:
        # - queue[0] is the task it is executing now (or will execute next)
        # - queue[1:] are future tasks already promised to that agent
        self._agent_queues: Dict[int, List[Task]] = {}
        self._agent_queue_cost: Dict[int, float] = {}
        self._agent_queue_reward: Dict[int, float] = {}
        self._agent_queue_end: Dict[int, Optional[Tuple[int, int]]] = {}
        # Cells predicted to be observed by agents following their queued paths
        self._predicted_revealed: Set[Tuple[int, int]] = set()
        self._map_rows: int = 0
        self._map_cols: int = 0
        # Tracks number of known obstacle cells observed so far; growth
        # between updates triggers a global reauction.
        self._known_obstacle_count: int = 0

    # ------------------------------------------------------------------
    # sweep_completions: override to also purge completed tasks from queues
    # ------------------------------------------------------------------
    def sweep_completions(self, known_map: KnownMap) -> int:
        count = 0
        for task in self._tasks:
            if not task.completed and task.check_completion(known_map):
                count += 1
        if count:
            self._purge_completed_from_queues()
        return count

    def _purge_completed_from_queues(self) -> None:
        """Remove collaterally-completed tasks from agent queues and
        recalculate cached queue costs."""
        for aid in list(self._agent_queues):
            queue = self._agent_queues[aid]
            removed = [t for i, t in enumerate(queue) if i != 0 and t.completed]
            for t in removed:
                self._agent_queue_cost[aid] -= getattr(t, '_queued_marginal_cost', 0)
                self._agent_queue_reward[aid] -= getattr(t, '_queued_effective_reward', 0)
                t.assignment_snapshot = None
            self._agent_queues[aid] = [t for i, t in enumerate(queue)
                                        if i == 0 or not t.completed]
            remaining = self._agent_queues[aid]
            if remaining:
                self._agent_queue_end[aid] = remaining[-1].target_loc
            else:
                self._agent_queue_cost[aid] = 0.0
                self._agent_queue_reward[aid] = 0.0
                self._agent_queue_end[aid] = None

    # ------------------------------------------------------------------
    # Per-agent queue helpers
    # ------------------------------------------------------------------
    def _ensure_agent(self, agent: Agent) -> None:
        if agent.id not in self._agent_queues:
            self._agent_queues[agent.id] = []
            self._agent_queue_cost[agent.id] = 0.0
            self._agent_queue_reward[agent.id] = 0.0
            self._agent_queue_end[agent.id] = None

    def _queue_end_pos(self, agent: Agent) -> Tuple[int, int]:
        """Position from which the agent would start its next task.

        This is the key queue-aware idea in SSICA: when the agent bids on a new
        task, the bid is not computed from "where the robot is right now", but
        from "where the robot will be after finishing the tasks it already won."
        """
        end = self._agent_queue_end.get(agent.id)
        return end if end is not None else agent.pos

    def _append_to_queue(self, agent: Agent, task: Task,
                         path: List[Tuple[int, int]],
                         marginal_cost: float,
                         effective_reward: Optional[float] = None) -> None:
        """Append a task to an agent's queue and update cached costs."""
        self._ensure_agent(agent)
        queue = self._agent_queues[agent.id]
        queue.append(task)
        task._queued_marginal_cost = marginal_cost
        rew = effective_reward if effective_reward is not None else _TASK_REWARD.get(type(task), 1)
        task._queued_effective_reward = rew
        self._agent_queue_cost[agent.id] += marginal_cost
        self._agent_queue_reward[agent.id] += rew
        self._agent_queue_end[agent.id] = task.target_loc
        task.assigned_to = agent.id
        task.assignment_snapshot = None

    def advance_queue(self, agent: Agent) -> Optional[Task]:
        """Pop the completed front task and return the next one (or None)."""
        self._ensure_agent(agent)
        queue = self._agent_queues[agent.id]
        while queue and queue[0].completed:
            done = queue.pop(0)
            self._agent_queue_cost[agent.id] -= getattr(done, '_queued_marginal_cost', 0)
            self._agent_queue_reward[agent.id] -= getattr(done, '_queued_effective_reward', 0)
            done.assignment_snapshot = None
        if not queue:
            self._agent_queue_cost[agent.id] = 0.0
            self._agent_queue_reward[agent.id] = 0.0
            self._agent_queue_end[agent.id] = None
            return None
        return queue[0]

    def _clear_agent_queue(self, agent: Agent) -> None:
        self._ensure_agent(agent)
        for task in self._agent_queues[agent.id]:
            if not task.completed:
                task.assigned_to = None
            task.assignment_snapshot = None
        self._agent_queues[agent.id] = []
        self._agent_queue_cost[agent.id] = 0.0
        self._agent_queue_reward[agent.id] = 0.0
        self._agent_queue_end[agent.id] = None

    def _preserve_active_queue_head(self, agent: Agent) -> None:
        """Keep only the active dwell task when reauction clears the queue."""
        self._ensure_agent(agent)
        task = agent.current_task
        if task is None:
            self._clear_agent_queue(agent)
            return
        for queued in self._agent_queues[agent.id]:
            if queued is task:
                continue
            if not queued.completed:
                queued.assigned_to = None
            queued.assignment_snapshot = None
        self._agent_queues[agent.id] = [task]
        task.assigned_to = agent.id
        task.assignment_snapshot = None
        self._agent_queue_cost[agent.id] = getattr(task, '_queued_marginal_cost', 0.0)
        self._agent_queue_reward[agent.id] = getattr(
            task, '_queued_effective_reward', _TASK_REWARD.get(type(task), 1),
        )
        self._agent_queue_end[agent.id] = task.target_loc

    def clear_agent_queue(self, agent: Agent) -> None:
        self._clear_agent_queue(agent)

    # ------------------------------------------------------------------
    # Collateral exploration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _path_footprint(
        path: List[Tuple[int, int]],
        obs_radius: int,
        rows: int,
        cols: int,
    ) -> Set[Tuple[int, int]]:
        revealed: Set[Tuple[int, int]] = set()
        for r, c in path:
            for dr in range(-obs_radius, obs_radius + 1):
                for dc in range(-obs_radius, obs_radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        revealed.add((nr, nc))
        return revealed

    def _collateral_bonus(self, path: List[Tuple[int, int]], obs_radius: int) -> float:
        footprint = self._path_footprint(path, obs_radius, self._map_rows, self._map_cols)
        new_cells = footprint - self._predicted_revealed
        return len(new_cells) * 0.25

    def _mark_path_revealed(self, path: List[Tuple[int, int]], obs_radius: int) -> None:
        self._predicted_revealed |= self._path_footprint(
            path, obs_radius, self._map_rows, self._map_cols
        )

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------
    def _activation_effective_reward(
        self,
        agent: Agent,
        task: Task,
        path: List[Tuple[int, int]],
        known_map: KnownMap,
    ) -> float:
        reward = _TASK_REWARD.get(type(task), 1)
        footprint = self._path_footprint(path, agent.obs_radius, known_map.rows, known_map.cols)
        new_cells = sum(
            1 for r, c in footprint if known_map.state[r][c] == ObservationState.UNKNOWN
        )
        return reward + (0.25 * new_cells)

    def _compute_active_bid(self, agent: Agent, task: Task,
                            known_map: KnownMap) -> Optional[Bid]:
        use_drone_path = isinstance(agent, DroneAgent)
        path = plan_path(known_map, agent.pos, task.target_loc, drone=use_drone_path)
        if not path:
            return None
        execution_cost = self._execution_cost(agent, task, path)
        effective_reward = self._activation_effective_reward(agent, task, path, known_map)
        return Bid(
            agent_id=agent.id,
            task_id=task.task_id,
            score=effective_reward / execution_cost,
            path=path,
            execution_cost=execution_cost,
            effective_reward=effective_reward,
        )

    def compute_bid(self, agent: Agent, task: Task, known_map: KnownMap) -> Optional[Bid]:
        """
        Bid = marginal improvement in average utility when adding this task.

        score = (R_queue + r) / (C_queue + c) - R_queue / C_queue

        where r = task_reward + collateral_bonus, c = marginal_cost,
        R_queue / C_queue are the totals already queued (0 if empty).
        """
        self._ensure_agent(agent)
        use_drone_path = isinstance(agent, DroneAgent)
        start_pos = self._queue_end_pos(agent)
        path = plan_path(known_map, start_pos, task.target_loc, drone=use_drone_path)
        if not path:
            return None
        marginal_cost = self._execution_cost(agent, task, path)
        reward = _TASK_REWARD.get(type(task), 1)
        collateral = self._collateral_bonus(path, agent.obs_radius)
        r = reward + collateral
        C_queue = self._agent_queue_cost[agent.id]
        score = r / (C_queue + marginal_cost)
        return Bid(
            agent_id=agent.id,
            task_id=task.task_id,
            score=score,
            path=path,
            execution_cost=marginal_cost,
            effective_reward=r,
        )

    def refresh_active_snapshot(self, task: Task, agents: List[Agent],
                                known_map: KnownMap) -> None:
        """Refresh runner-up metadata for the current head task only."""
        bids: List[Bid] = []
        for agent in agents:
            if isinstance(task, TriageTask) and task.ground_only and agent.agent_type != AgentType.GROUND:
                continue
            bid = self._compute_active_bid(agent, task, known_map)
            if bid is not None:
                bids.append(bid)
        if not bids:
            task.assignment_snapshot = None
            return
        ordered = self._sorted_bids(bids)
        winner = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        task.assignment_snapshot = AssignmentSnapshot(
            winner_agent_id=winner.agent_id,
            winner_metric=winner.score,
            winner_execution_cost=winner.execution_cost,
            winner_effective_reward=winner.effective_reward,
            runner_up_agent_id=None if runner_up is None else runner_up.agent_id,
            runner_up_metric=float("-inf") if runner_up is None else runner_up.score,
            runner_up_execution_cost=0.0 if runner_up is None else runner_up.execution_cost,
            runner_up_effective_reward=0.0 if runner_up is None else runner_up.effective_reward,
            mode="activation",
        )

    # ------------------------------------------------------------------
    # Override: repair (refresh snapshot first when missing)
    # ------------------------------------------------------------------
    def handle_invalidated_assignment(
        self,
        agent: Agent,
        agents: List[Agent],
        known_map: KnownMap,
        verbose: bool = False,
        reason: str = "blocked",
    ) -> bool:
        task = agent.current_task
        if task is None or task.completed:
            return False

        if task.assignment_snapshot is None:
            self.refresh_active_snapshot(task, agents, known_map)

        snapshot = task.assignment_snapshot
        if snapshot is None:
            if verbose:
                print(f"  [REPAIR] Task {task.task_id} has no activation snapshot; reauctioning")
            self.trigger_global_reauction(agents, known_map)
            return True

        repair = self._repair_bid(agent, task, known_map,
                                  effective_reward=snapshot.winner_effective_reward)
        if repair is None:
            if verbose:
                print(f"  [REPAIR] Agent {agent.id} cannot repair task {task.task_id}; reauctioning")
            self.trigger_global_reauction(agents, known_map)
            return True

        if (snapshot.runner_up_agent_id is not None
                and snapshot.runner_up_metric > repair.score + _REPAIR_METRIC_SLACK):
            if verbose:
                print(f"  [REPAIR] Agent {agent.id} repaired task {task.task_id} "
                      f"score={repair.score:.3f} runner_up={snapshot.runner_up_metric:.3f}; reauctioning")
            self.trigger_global_reauction(agents, known_map)
            return True

        task.assigned_to = agent.id
        agent.current_task = task
        agent.path = repair.path
        agent.status = AgentStatus.NAVIGATING

        if not isinstance(agent, DroneAgent) and not self._cbs_replan_ground(agents, known_map):
            if verbose:
                print(f"  [REPAIR] Agent {agent.id} kept task {task.task_id} but CBS failed; reauctioning")
            self.trigger_global_reauction(agents, known_map)
            return True

        snapshot.winner_agent_id = agent.id
        snapshot.winner_metric = repair.score
        snapshot.winner_execution_cost = repair.execution_cost
        if verbose:
            print(f"  [REPAIR] Agent {agent.id} kept task {task.task_id} after {reason}; "
                  f"score={repair.score:.3f}")
        return False

    # ------------------------------------------------------------------
    # Override: trigger_global_reauction (uses queue management)
    # ------------------------------------------------------------------
    def trigger_global_reauction(self, agents: List[Agent], known_map: KnownMap) -> None:
        self.reauction_count += 1
        preserved_by_task: Dict[int, int] = {
            agent.current_task.task_id: agent.id
            for agent in agents
            if self._is_dwelling_agent(agent) and agent.current_task is not None
        }
        for task in self._tasks:
            if not task.completed:
                keeper_id = preserved_by_task.get(task.task_id)
                if keeper_id is not None:
                    task.assigned_to = keeper_id
                    task.assignment_snapshot = None
                    continue
                task.assigned_to = None
                task.assignment_snapshot = None
                if isinstance(task, TriageTask):
                    task.progress = 0
        for agent in agents:
            if self._is_dwelling_agent(agent):
                self._preserve_active_queue_head(agent)
                agent.path = []
                agent.status = AgentStatus.NAVIGATING
                continue
            self._clear_agent_queue(agent)
            agent.current_task = None
            agent.path = []
            agent.status = AgentStatus.IDLE
        print(f"  [REAUCTION] Global reauction triggered #{self.reauction_count}")
        self._in_reauction = True
        self.auction(agents, known_map)
        self._in_reauction = False

    # ------------------------------------------------------------------
    # Auction (all agents bid; won tasks appended to winner's queue)
    # ------------------------------------------------------------------
    def auction(self, agents: List[Agent], known_map: KnownMap) -> None:
        """
        Run a sequential single-item auction where ALL agents bid, even
        those already holding tasks.  Won tasks are appended to the
        winner's queue.  The first task in the queue is the active task.
        """
        pending = self.pending()
        if not pending:
            return

        self._map_rows = known_map.rows
        self._map_cols = known_map.cols
        self._predicted_revealed = set()
        for r in range(known_map.rows):
            for c in range(known_map.cols):
                if known_map.state[r][c] != ObservationState.UNKNOWN:
                    self._predicted_revealed.add((r, c))
        for agent in agents:
            self._ensure_agent(agent)
            for qtask in self._agent_queues[agent.id]:
                if not qtask.completed and hasattr(qtask, '_queued_path'):
                    self._mark_path_revealed(qtask._queued_path, agent.obs_radius)

        for agent in agents:
            self._ensure_agent(agent)

        available_tasks = sorted(
            pending,
            key=lambda t: self._auction_priority(t, agents),
            reverse=True,
        )
        activated_tasks: List[Task] = []

        for task in available_tasks:
            bids: List[Bid] = []
            for agent in agents:
                if (isinstance(task, TriageTask) and task.ground_only
                        and agent.agent_type != AgentType.GROUND):
                    continue
                bid = self.compute_bid(agent, task, known_map)
                if bid is not None:
                    bids.append(bid)

            if not bids:
                continue

            winner = max(bids, key=lambda b: (b.score, -b.agent_id))
            winning_agent = next(a for a in agents if a.id == winner.agent_id)

            effective_reward = winner.effective_reward
            self._append_to_queue(
                winning_agent, task, winner.path, winner.execution_cost, effective_reward
            )
            task._queued_path = winner.path
            self._mark_path_revealed(winner.path, winning_agent.obs_radius)

            if winning_agent.current_task is None or winning_agent.current_task.completed:
                winning_agent.assign_task(task)
                winning_agent.path = winner.path
                winning_agent.status = AgentStatus.NAVIGATING
                activated_tasks.append(task)

            print(
                f"  [AUCTION] Task {task.task_id} -> Agent {winning_agent.id} "
                f"target={task.target_loc} bid_score={winner.score:.2f} "
                f"queue_len={len(self._agent_queues[winning_agent.id])}"
            )

        if not self._in_reauction and not self._cbs_replan_ground(agents, known_map):
            print("  [CBS] Multi-agent replan failed; triggering reauction")
            self.trigger_global_reauction(agents, known_map)
            return

        for task in activated_tasks:
            self.refresh_active_snapshot(task, agents, known_map)

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

        use_drone_path = isinstance(agent, DroneAgent)

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

        Only checks conflicts between agents of the same type (ground-ground
        or drone-drone), since they operate at different altitudes.
        """
        next_pos: Dict[int, Tuple[int, int]] = {}
        curr_pos: Dict[int, Tuple[int, int]] = {a.id: a.pos for a in agents}

        active_agents = []
        for agent in agents:
            if agent.status != AgentStatus.NAVIGATING or len(agent.path) < 2:
                continue
            next_pos[agent.id] = agent.path[1]
            active_agents.append(agent)

        # Vertex conflicts: two same-type agents targeting the same cell
        from collections import defaultdict
        by_type: Dict[AgentType, List[Agent]] = defaultdict(list)
        for agent in active_agents:
            by_type[agent.agent_type].append(agent)

        for group in by_type.values():
            seen: Set[Tuple[int, int]] = set()
            for agent in group:
                pos = next_pos[agent.id]
                if pos in seen:
                    return True
                seen.add(pos)

        # Edge (swap) conflicts: only between same-type agents
        for group in by_type.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a = group[i]
                    b = group[j]
                    if next_pos[a.id] == curr_pos[b.id] and next_pos[b.id] == curr_pos[a.id]:
                        return True
        return False

    def should_trigger_reauction(self, agents: List[Agent], known_map: KnownMap) -> bool:
        """
        Check for global reauction conditions.

        Triggers on inter-robot motion conflicts (vertex or edge
        collisions between same-type agents). New high-value task
        discovery is handled inline in :meth:`update`. Path infeasibility
        is handled by normal replanning and auction.
        """
        return self._next_move_conflict(agents)

    def trigger_global_reauction(self, agents: List[Agent], known_map: KnownMap) -> None:
        """
        Release incomplete assignments, preserving only active on-target dwell
        work, then clear remaining queues and run a fresh auction round.
        """
        self.reauction_count += 1
        preserved_by_task: Dict[int, int] = {
            agent.current_task.task_id: agent.id
            for agent in agents
            if self._is_dwelling_agent(agent) and agent.current_task is not None
        }

        for task in self._tasks:
            if not task.completed:
                keeper_id = preserved_by_task.get(task.task_id)
                if keeper_id is not None:
                    task.assigned_to = keeper_id
                    task.assignment_snapshot = None
                    continue
                task.assigned_to = None
                task.assignment_snapshot = None
                if isinstance(task, TriageTask):
                    task.progress = 0

        for agent in agents:
            if self._is_dwelling_agent(agent):
                self._preserve_active_queue_head(agent)
                agent.path = []
                agent.status = AgentStatus.NAVIGATING
                continue
            self._clear_agent_queue(agent)
            agent.current_task = None
            agent.path = []
            agent.status = AgentStatus.IDLE

        print(f"  [REAUCTION] Global reauction triggered #{self.reauction_count}")
        self._in_reauction = True
        self.auction(agents, known_map)
        self._in_reauction = False

    def _cbs_replan_ground(self, agents: List[Agent], known_map: KnownMap) -> bool:
        """Run CBS jointly for all navigating ground agents to resolve collisions.

        Drone agents are excluded — they fly at a different altitude and don't
        collide with ground agents.

        Returns True if paths are collision-free, False if CBS failed.
        """
        if self._cbs is None:
            return True

        ground_nav = [
            a for a in agents
            if not isinstance(a, DroneAgent)
            and a.status == AgentStatus.NAVIGATING
            and a.current_task is not None
        ]
        if len(ground_nav) < 2:
            return True

        starts = {a.id: a.pos for a in ground_nav}
        goals = {a.id: a.current_task.target_loc for a in ground_nav}

        paths = self._cbs.plan(starts, goals, known_map, drone=False)
        if paths is None:
            return False

        for a in ground_nav:
            if a.id in paths:
                a.path = paths[a.id]
                self._cbs.set_path(a.id, paths[a.id])
        return True

    # ------------------------------------------------------------------
    # Convenience update hook for the main loop
    # Override: update (advances queue on task completion)
    # ------------------------------------------------------------------
    def update(self, agents: List[Agent], known_map: KnownMap,
               ground_truth=None, verbose: bool = False) -> None:
        new_explore = self.add_frontier_tasks(known_map)
        if new_explore and verbose:
            print(f"  [FRONTIER] +{new_explore} exploration task(s) queued")

        new_triage = 0
        new_bldg = 0
        if ground_truth is not None:
            new_triage, new_invest = self.add_revealed_triage_tasks(known_map, ground_truth)
            if (new_triage + new_invest) and verbose:
                print(f"  [TASKS]   +{new_triage} triage +{new_invest} investigation task(s) revealed")
            new_bldg = self.add_confirmed_building_triage(known_map)
            if new_bldg and verbose:
                print(f"  [TRIAGE]  +{new_bldg} occupied building triage task(s) confirmed")

        # Trigger reauction when high-value tasks appear: objective triage
        # or confirmed occupied building triage — but NOT low-value
        # investigation tasks (preliminary building checks).
        new_high_value_tasks = (new_triage + new_bldg) > 0

        # Detect newly observed obstacles since last update.
        obstacle_count = sum(
            1
            for r in range(known_map.rows)
            for c in range(known_map.cols)
            if known_map.state[r][c] == ObservationState.OBSTACLE
        )
        new_obstacles = obstacle_count - self._known_obstacle_count
        self._known_obstacle_count = obstacle_count
        if new_obstacles > 0 and verbose:
            print(f"  [OBSTACLES] +{new_obstacles} newly observed obstacle cell(s)")

        # 3. Sweep collateral completions
        swept = self.sweep_completions(known_map)
        if swept and verbose:
            print(f"  [SWEPT]   {swept} task(s) observed as collateral")

        for agent in agents:
            if agent.current_task and agent.current_task.completed:
                if verbose:
                    task = agent.current_task
                    if isinstance(task, TriageTask) and getattr(task, '_is_investigation', False):
                        print(f"  [INVESTIGATED] Agent {agent.id} checked building at {task.target_loc}")
                    elif isinstance(task, TriageTask):
                        print(f"  [TRIAGE DONE] Agent {agent.id} finished triage task "
                              f"{task.task_id} at {task.target_loc}")
                    else:
                        print(f"  [TASK DONE] Agent {agent.id} finished task"
                              f" {task.task_id}  (observed {task.target_loc})")
                agent.current_task.assignment_snapshot = None

                next_task = self.advance_queue(agent)
                if next_task is not None:
                    agent.current_task = next_task
                    agent.path = []
                    agent.status = AgentStatus.REPLANNING
                    self.refresh_active_snapshot(next_task, agents, known_map)
                    if verbose:
                        print(f"  [QUEUE] Agent {agent.id} advancing to next task "
                              f"{next_task.task_id} target={next_task.target_loc} "
                              f"remaining_queue={len(self._agent_queues[agent.id])}")
                else:
                    agent.current_task = None
                    agent.path = []
                    agent.status = AgentStatus.IDLE

        for agent in agents:
            if self._path_infeasible(agent, known_map):
                reauctioned = self.handle_invalidated_assignment(
                    agent, agents, known_map, verbose=verbose, reason="repair-screen"
                )
                if reauctioned:
                    return

        # 5. Reauction on conflict, new high-value task discovery, or newly
        #    observed obstacles; otherwise normal auction.
        if new_high_value_tasks and verbose:
            print(f"  [REAUCTION] Triggered by new high-value tasks "
                  f"(triage={new_triage}, building={new_bldg})")
        if new_obstacles > 0 and verbose:
            print(f"  [REAUCTION] Triggered by {new_obstacles} new obstacle cell(s)")
        if (
            new_high_value_tasks
            or new_obstacles > 0
            or self.should_trigger_reauction(agents, known_map)
        ):
            self.trigger_global_reauction(agents, known_map)
        else:
            self.auction(agents, known_map)
