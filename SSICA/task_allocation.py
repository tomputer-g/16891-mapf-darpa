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

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from agents import Agent, AgentStatus, DroneAgent, plan_path
from maps import KnownMap, ObservationState
from planner import CBS
from sim_types import AgentType
from tasks import AssignmentSnapshot, ExplorationTask, TriageTask, Task


# Reward values used in the bid formula: reward / (path_length + dwell_time)
_TASK_REWARD = {
    ExplorationTask: 1,
    TriageTask: 8,
}

# Dwell steps by agent type (must stay in sync with SSICA.main._TRIAGE_DWELL)
_TRIAGE_DWELL = {AgentType.GROUND: 2, AgentType.DRONE: 4}
_REPAIR_METRIC_SLACK = 0.01


@dataclass
class Bid:
    """Single sealed bid for one task from one agent."""
    agent_id: int
    task_id: int
    score: float
    path: List[Tuple[int, int]]
    execution_cost: float = 0.0
    effective_reward: float = 0.0


class SequentialSingleItemAuctioneer:
    """
    Sequential single-item concurrent auctioneer with per-agent task queues.

    Every agent bids on every unassigned task.  Bid cost accounts for all
    tasks already in the agent's queue: cumulative queue cost + path from
    the last queued target to the new task target.
    """

    def __init__(self, cbs: Optional[CBS] = None) -> None:
        self._tasks: List[Task] = []
        self._known_locs: Set[Tuple[int, int]] = set()
        self._triage_locs: Set[Tuple[int, int]] = set()
        self._invest_locs: Set[Tuple[int, int]] = set()
        self.reauction_count: int = 0
        self._cbs: Optional[CBS] = cbs
        self._in_reauction: bool = False
        # Per-agent task queue: agent_id -> ordered list of Tasks.
        #
        # SSICA is "concurrent" because agents are allowed to win more than one
        # task in the same auction round. That only works if each agent can
        # remember future commitments, so each agent owns a queue:
        # - queue[0] is the task it is executing now (or will execute next)
        # - queue[1:] are future tasks already promised to that agent
        #
        # Without this queue, SSICA would collapse back into SSIA-style
        # "one active task per available agent" behavior.
        self._agent_queues: Dict[int, List[Task]] = {}
        # Cached cumulative cost per agent (sum of path+dwell for queued tasks)
        self._agent_queue_cost: Dict[int, float] = {}
        # Cached cumulative reward per agent (sum of rewards for queued tasks)
        self._agent_queue_reward: Dict[int, float] = {}
        # Cached "end location" — target of the last task in the queue
        self._agent_queue_end: Dict[int, Optional[Tuple[int, int]]] = {}
        # Cells predicted to be observed by agents following their queued paths
        self._predicted_revealed: Set[Tuple[int, int]] = set()
        # Map dimensions (set on first auction call)
        self._map_rows: int = 0
        self._map_cols: int = 0

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

    def add_revealed_triage_tasks(self, known_map: KnownMap, ground_truth) -> Tuple[int, int]:
        """Create tasks for revealed objectives and buildings.
        Returns (triage_count, investigation_count)."""
        triage_count = 0
        invest_count = 0
        for loc in ground_truth.objectives:
            r, c = loc
            if known_map.state[r][c] == ObservationState.OBJECTIVE and loc not in self._triage_locs:
                task = TriageTask(loc)
                self._triage_locs.add(loc)
                self._tasks.append(task)
                triage_count += 1
        for loc in ground_truth.buildings:
            r, c = loc
            state = known_map.state[r][c]
            if state in (ObservationState.BUILDING, ObservationState.OCCUPIED_BUILDING) and loc not in self._invest_locs:
                task = TriageTask(loc, ground_only=True, dwell_steps=1)
                task._is_investigation = True
                self._invest_locs.add(loc)
                self._tasks.append(task)
                invest_count += 1
        return triage_count, invest_count

    def add_confirmed_building_triage(self, known_map: KnownMap) -> int:
        """Create triage tasks for buildings confirmed occupied after investigation."""
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
        # Purge completed tasks from the middle of agent queues
        if count:
            self._purge_completed_from_queues()
        return count

    def _purge_completed_from_queues(self) -> None:
        """Remove collaterally-completed tasks from agent queues and
        recalculate cached queue costs."""
        for aid in list(self._agent_queues):
            queue = self._agent_queues[aid]
            # Subtract costs of completed non-front tasks before removing
            removed = [
                t for i, t in enumerate(queue)
                if i != 0 and t.completed
            ]
            for t in removed:
                self._agent_queue_cost[aid] -= getattr(t, '_queued_marginal_cost', 0)
                self._agent_queue_reward[aid] -= getattr(t, '_queued_effective_reward', 0)
                t.assignment_snapshot = None
            self._agent_queues[aid] = [
                t for i, t in enumerate(queue)
                if i == 0 or not t.completed
            ]
            remaining = self._agent_queues[aid]
            if remaining:
                self._agent_queue_end[aid] = remaining[-1].target_loc
            else:
                self._agent_queue_cost[aid] = 0.0
                self._agent_queue_reward[aid] = 0.0
                self._agent_queue_end[aid] = None

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
    # Per-agent queue helpers
    # ------------------------------------------------------------------
    def _ensure_agent(self, agent: Agent) -> None:
        """Lazily initialize queue state for an agent."""
        if agent.id not in self._agent_queues:
            self._agent_queues[agent.id] = []
            self._agent_queue_cost[agent.id] = 0.0
            self._agent_queue_reward[agent.id] = 0.0
            self._agent_queue_end[agent.id] = None

    def _queue_end_pos(self, agent: Agent) -> Tuple[int, int]:
        """Return the position from which the agent would start its next task.

        If the agent has queued tasks, this is the target of the last task.
        Otherwise it is the agent's current position.

        This is the key queue-aware idea in SSICA: when the agent bids on a new
        task, the bid is not computed from "where the robot is right now", but
        from "where the robot will be after finishing the tasks it already won."
        """
        end = self._agent_queue_end.get(agent.id)
        return end if end is not None else agent.pos

    def _append_to_queue(self, agent: Agent, task: Task, path: List[Tuple[int, int]],
                         marginal_cost: float, effective_reward: float = None) -> None:
        """Append a task to an agent's queue and update cached costs.

        The queue lets SSICA commit future work early. Once a task is appended,
        later bids from the same agent become less attractive because their
        cost is evaluated after all already-queued work.
        """
        self._ensure_agent(agent)
        queue = self._agent_queues[agent.id]
        queue.append(task)
        # Store per-task costs for accurate subtraction later
        task._queued_marginal_cost = marginal_cost
        rew = effective_reward if effective_reward is not None else _TASK_REWARD.get(type(task), 1)
        task._queued_effective_reward = rew
        self._agent_queue_cost[agent.id] += marginal_cost
        self._agent_queue_reward[agent.id] += rew
        self._agent_queue_end[agent.id] = task.target_loc
        task.assigned_to = agent.id
        task.assignment_snapshot = None

    def advance_queue(self, agent: Agent) -> Optional[Task]:
        """Pop the completed front task and assign the next one.

        Returns the new current task (or None if the queue is empty).

        This is how queued commitments become active execution: when the front
        task finishes, the next queued task moves into the current-task slot.
        """
        self._ensure_agent(agent)
        queue = self._agent_queues[agent.id]

        # Remove completed tasks from the front, subtracting their costs
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

        next_task = queue[0]
        return next_task

    def _clear_agent_queue(self, agent: Agent) -> None:
        """Wipe the agent's entire queue."""
        self._ensure_agent(agent)
        for task in self._agent_queues[agent.id]:
            if not task.completed:
                task.assigned_to = None
            task.assignment_snapshot = None
        self._agent_queues[agent.id] = []
        self._agent_queue_cost[agent.id] = 0.0
        self._agent_queue_reward[agent.id] = 0.0
        self._agent_queue_end[agent.id] = None

    @staticmethod
    def _is_dwelling_agent(agent: Agent) -> bool:
        task = agent.current_task
        return (
            isinstance(task, TriageTask)
            and not task.completed
            and agent.pos == task.target_loc
        )

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
            task,
            '_queued_effective_reward',
            _TASK_REWARD.get(type(task), 1),
        )
        self._agent_queue_end[agent.id] = task.target_loc

    def clear_agent_queue(self, agent: Agent) -> None:
        """Public interface to clear an agent's task queue."""
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
        """Return all cells that would be observed by an agent walking *path*."""
        revealed: Set[Tuple[int, int]] = set()
        for r, c in path:
            for dr in range(-obs_radius, obs_radius + 1):
                for dc in range(-obs_radius, obs_radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        revealed.add((nr, nc))
        return revealed

    def _collateral_bonus(
        self,
        path: List[Tuple[int, int]],
        obs_radius: int,
    ) -> float:
        """Count all new cells that would be revealed by *path* but are
        NOT already in _predicted_revealed.  Each new cell adds +0.5."""
        footprint = self._path_footprint(
            path, obs_radius, self._map_rows, self._map_cols
        )
        new_cells = footprint - self._predicted_revealed
        return len(new_cells) * 0.25

    def _mark_path_revealed(
        self,
        path: List[Tuple[int, int]],
        obs_radius: int,
    ) -> None:
        """Add the observation footprint of *path* to _predicted_revealed."""
        self._predicted_revealed |= self._path_footprint(
            path, obs_radius, self._map_rows, self._map_cols
        )

    @staticmethod
    def _sorted_bids(bids: List[Bid]) -> List[Bid]:
        return sorted(bids, key=lambda b: (b.score, -b.agent_id), reverse=True)

    def _task_dwell(self, agent: Agent, task: Task) -> int:
        if not isinstance(task, TriageTask):
            return 0
        if getattr(task, '_is_investigation', False):
            return 1
        return _TRIAGE_DWELL.get(agent.agent_type, task.dwell_steps)

    def _execution_cost(
        self,
        agent: Agent,
        task: Task,
        path: List[Tuple[int, int]],
    ) -> float:
        path_len = max(1, len(path) - 1)
        if isinstance(agent, DroneAgent):
            path_len = path_len / 2
        return path_len + self._task_dwell(agent, task)

    def _activation_effective_reward(
        self,
        agent: Agent,
        task: Task,
        path: List[Tuple[int, int]],
        known_map: KnownMap,
    ) -> float:
        reward = _TASK_REWARD.get(type(task), 1)
        footprint = self._path_footprint(
            path,
            agent.obs_radius,
            known_map.rows,
            known_map.cols,
        )
        new_cells = sum(
            1
            for r, c in footprint
            if known_map.state[r][c] == ObservationState.UNKNOWN
        )
        return reward + (0.25 * new_cells)

    def _compute_active_bid(
        self,
        agent: Agent,
        task: Task,
        known_map: KnownMap,
    ) -> Optional[Bid]:
        use_drone_path = isinstance(agent, DroneAgent)
        path = plan_path(
            known_map,
            agent.pos,
            task.target_loc,
            drone=use_drone_path,
        )
        if not path:
            return None

        execution_cost = self._execution_cost(agent, task, path)
        effective_reward = self._activation_effective_reward(
            agent,
            task,
            path,
            known_map,
        )
        return Bid(
            agent_id=agent.id,
            task_id=task.task_id,
            score=effective_reward / execution_cost,
            path=path,
            execution_cost=execution_cost,
            effective_reward=effective_reward,
        )

    def refresh_active_snapshot(
        self,
        task: Task,
        agents: List[Agent],
        known_map: KnownMap,
    ) -> None:
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

    def _repair_bid(
        self,
        agent: Agent,
        task: Task,
        known_map: KnownMap,
        effective_reward: float,
    ) -> Optional[Bid]:
        use_drone_path = isinstance(agent, DroneAgent)
        path = plan_path(
            known_map,
            agent.pos,
            task.target_loc,
            drone=use_drone_path,
        )
        if not path:
            return None

        execution_cost = self._execution_cost(agent, task, path)
        return Bid(
            agent_id=agent.id,
            task_id=task.task_id,
            score=effective_reward / execution_cost,
            path=path,
            execution_cost=execution_cost,
            effective_reward=effective_reward,
        )

    def handle_invalidated_assignment(
        self,
        agent: Agent,
        agents: List[Agent],
        known_map: KnownMap,
        verbose: bool = False,
        reason: str = "blocked",
    ) -> bool:
        """Repair the active task locally or clear queues and reauction."""
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

        repair = self._repair_bid(
            agent,
            task,
            known_map,
            effective_reward=snapshot.winner_effective_reward,
        )
        if repair is None:
            if verbose:
                print(f"  [REPAIR] Agent {agent.id} cannot repair task {task.task_id}; reauctioning")
            self.trigger_global_reauction(agents, known_map)
            return True

        if (
            snapshot.runner_up_agent_id is not None
            and snapshot.runner_up_metric > repair.score + _REPAIR_METRIC_SLACK
        ):
            if verbose:
                print(
                    f"  [REPAIR] Agent {agent.id} repaired task {task.task_id} "
                    f"score={repair.score:.3f} runner_up={snapshot.runner_up_metric:.3f}; reauctioning"
                )
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
            print(
                f"  [REPAIR] Agent {agent.id} kept task {task.task_id} after {reason}; "
                f"score={repair.score:.3f}"
            )
        return False

    # ------------------------------------------------------------------
    # Bidding and assignment
    # ------------------------------------------------------------------
    def compute_bid(
        self,
        agent: Agent,
        task: Task,
        known_map: KnownMap,
    ) -> Optional[Bid]:
        """
        Bid = marginal improvement in average utility when adding this task.

        score = (R_queue + r) / (C_queue + c) - R_queue / C_queue

        where r = task_reward + collateral_bonus, c = marginal_cost,
        R_queue = sum of rewards already queued, C_queue = sum of costs queued.
        If the queue is empty, the second term is 0.
        """
        self._ensure_agent(agent)

        use_drone_path = isinstance(agent, DroneAgent)
        start_pos = self._queue_end_pos(agent)

        path = plan_path(
            known_map,
            start_pos,
            task.target_loc,
            drone=use_drone_path,
        )
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

    @staticmethod
    def _auction_priority(task: Task, agents: List[Agent]) -> float:
        """Score used to decide the order tasks are offered in the auction.

        priority = reward - manhattan_distance_to_closest_eligible_agent

        High-reward tasks close to at least one agent are offered first.
        """
        reward = _TASK_REWARD.get(type(task), 1)
        tr, tc = task.target_loc
        min_dist = float('inf')
        for agent in agents:
            if (isinstance(task, TriageTask) and task.ground_only
                    and agent.agent_type != AgentType.GROUND):
                continue
            dist = abs(agent.pos[0] - tr) + abs(agent.pos[1] - tc)
            if dist < min_dist:
                min_dist = dist
        if min_dist == float('inf'):
            min_dist = 0
        return reward - min_dist

    def auction(self, agents: List[Agent], known_map: KnownMap) -> None:
        """
        Run a sequential single-item auction where ALL agents bid, even
        those already holding tasks.  Won tasks are appended to the
        winner's queue.  The first task in the queue is the active task.
        """
        pending = self.pending()
        if not pending:
            return

        # Cache map dimensions and reset predicted revealed for this round
        self._map_rows = known_map.rows
        self._map_cols = known_map.cols
        self._predicted_revealed = set()
        # Seed with cells already known (not UNKNOWN)
        for r in range(known_map.rows):
            for c in range(known_map.cols):
                if known_map.state[r][c] != ObservationState.UNKNOWN:
                    self._predicted_revealed.add((r, c))
        # Add footprints from existing agent queue paths
        for agent in agents:
            self._ensure_agent(agent)
            for qtask in self._agent_queues[agent.id]:
                if not qtask.completed and hasattr(qtask, '_queued_path'):
                    self._mark_path_revealed(qtask._queued_path, agent.obs_radius)

        # Ensure all agents are tracked
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
                # Filter ground_only tasks from non-ground agents
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
                winning_agent,
                task,
                winner.path,
                winner.execution_cost,
                effective_reward,
            )

            # Store path on task for footprint tracking and update predicted map
            task._queued_path = winner.path
            self._mark_path_revealed(winner.path, winning_agent.obs_radius)

            # If this is the agent's first/only task, make it the active one
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

        # CBS multi-agent replan for all navigating ground agents
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

        Only triggers on inter-robot motion conflicts (vertex or edge
        collisions between same-type agents). Path infeasibility and
        new task discovery are handled by normal replanning and auction.
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
    # ------------------------------------------------------------------
    def update(self, agents: List[Agent], known_map: KnownMap,
               ground_truth=None, verbose: bool = False) -> None:
        """
        One allocator update: task creation -> sweep -> release -> reauction/auction.
        """
        # 1. Add exploration frontier tasks
        new_explore = self.add_frontier_tasks(known_map)
        if new_explore and verbose:
            print(f"  [FRONTIER] +{new_explore} exploration task(s) queued")

        # 2. Add revealed triage / investigation tasks
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

        # 3. Sweep collateral completions
        swept = self.sweep_completions(known_map)
        if swept and verbose:
            print(f"  [SWEPT]   {swept} task(s) observed as collateral")

        # 4. Release agents whose current task is finished — advance queue
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

                # Advance to next task in queue
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

        # 4b. Repair infeasible active tasks locally before clearing queues.
        for agent in agents:
            if self._path_infeasible(agent, known_map):
                reauctioned = self.handle_invalidated_assignment(
                    agent,
                    agents,
                    known_map,
                    verbose=verbose,
                    reason="repair-screen",
                )
                if reauctioned:
                    return

        # 5. Reauction on conflict, otherwise normal auction
        if self.should_trigger_reauction(agents, known_map):
            self.trigger_global_reauction(agents, known_map)
        else:
            self.auction(agents, known_map)
