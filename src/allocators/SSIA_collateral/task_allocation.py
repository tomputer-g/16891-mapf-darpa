"""
Auction and task allocation for the DARPA exploration simulation.

Collateral-aware sequential single-item auction with
repair-vs-reauction control.

This variant keeps the SSIA structure but shapes bids with a collateral
exploration bonus. Like SSIA, it stores winner and runner-up metadata on
tasks, attempts local repair first on invalidated paths, and only escalates
to global reauction when repair fails or becomes clearly unattractive.

During full reauction, agents already dwelling on incomplete triage tasks are
preserved instead of having that in-progress work reset.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

from src.agents import Agent, AgentStatus, DroneAgent, plan_path
from src.maps import KnownMap, ObservationState
from src.planner import CBS
from src.sim_types import AgentType
from src.tasks import ExplorationTask, TriageTask, Task
from src.allocators.base_task_allocation import (
    BaseSSIATaskAuctioneer, Bid, _TRIAGE_DWELL,
)


_TASK_REWARD = {
    ExplorationTask: 1,
    TriageTask: 8,
}


class SequentialSingleItemAuctioneer(BaseSSIATaskAuctioneer):
    """
    Availability-constrained sequential single-item auctioneer with
    collateral exploration bonuses.

    Inherits task bookkeeping, execution-cost helpers, CBS replanning,
    repair-vs-reauction heuristic, conflict detection, and the standard
    update loop from BaseSSIATaskAuctioneer.  Overrides compute_bid and
    auction to incorporate per-path collateral bonuses.
    """

    _task_reward = _TASK_REWARD

    def __init__(self, cbs: Optional[CBS] = None) -> None:
        super().__init__(cbs)
        # Cells predicted to be observed by agents following their winning paths
        self._predicted_revealed: Set[Tuple[int, int]] = set()
        self._map_rows: int = 0
        self._map_cols: int = 0

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
        """Count new cells revealed by path not yet in _predicted_revealed."""
        footprint = self._path_footprint(path, obs_radius, self._map_rows, self._map_cols)
        new_cells = footprint - self._predicted_revealed
        return len(new_cells) * 0.25

    def _mark_path_revealed(self, path: List[Tuple[int, int]], obs_radius: int) -> None:
        self._predicted_revealed |= self._path_footprint(
            path, obs_radius, self._map_rows, self._map_cols
        )

    # ------------------------------------------------------------------
    # Bidding (with collateral bonus)
    # ------------------------------------------------------------------
    def compute_bid(self, agent: Agent, task: Task, known_map: KnownMap) -> Optional[Bid]:
        """Bid = (reward + collateral_bonus) / (path_length + dwell_time)."""
        use_drone_path = isinstance(agent, DroneAgent)
        path = plan_path(known_map, agent.pos, task.target_loc, drone=use_drone_path)
        if not path:
            return None
        execution_cost = self._execution_cost(agent, task, path)
        reward = _TASK_REWARD.get(type(task), 1)
        collateral = self._collateral_bonus(path, agent.obs_radius)
        score = (reward + collateral) / execution_cost
        return Bid(
            agent_id=agent.id,
            task_id=task.task_id,
            score=score,
            path=path,
            execution_cost=execution_cost,
            effective_reward=reward + collateral,
        )

    # ------------------------------------------------------------------
    # Auction (seeds collateral state before each round)
    # ------------------------------------------------------------------
    def auction(self, agents: List[Agent], known_map: KnownMap) -> None:
        """
        Run a sequential single-item auction.

        Tasks are offered in order of reward - manhattan_dist_to_closest_agent
        so nearby high-reward tasks are assigned first.
        """
        pending = self.pending()
        if not pending:
            return

        available_agents = list(self.available_agents(agents))
        if not available_agents:
            return

        self._map_rows = known_map.rows
        self._map_cols = known_map.cols
        self._predicted_revealed = set()
        for r in range(known_map.rows):
            for c in range(known_map.cols):
                if known_map.state[r][c] != ObservationState.UNKNOWN:
                    self._predicted_revealed.add((r, c))

        busy_agents = [a for a in agents if a not in available_agents]
        for agent in busy_agents:
            if agent.path and len(agent.path) > 1:
                self._mark_path_revealed(agent.path, agent.obs_radius)

        available_tasks = sorted(
            pending,
            key=lambda t: self._auction_priority(t, available_agents),
            reverse=True,
        )

        for task in available_tasks:
            if not available_agents:
                break

            bids: List[Bid] = []
            for agent in available_agents:
                if (isinstance(task, TriageTask) and task.ground_only
                        and agent.agent_type != AgentType.GROUND):
                    continue
                bid = self.compute_bid(agent, task, known_map)
                if bid is not None:
                    bids.append(bid)

            if not bids:
                continue

            winner = max(bids, key=lambda b: (b.score, -b.agent_id))
            winning_agent = next(a for a in available_agents if a.id == winner.agent_id)

            task.assigned_to = winning_agent.id
            self._store_assignment_snapshot(task, bids)
            winning_agent.assign_task(task)
            winning_agent.path = winner.path
            winning_agent.status = AgentStatus.NAVIGATING
            available_agents.remove(winning_agent)

            self._mark_path_revealed(winner.path, winning_agent.obs_radius)

            print(
                f"  [AUCTION] Task {task.task_id} -> Agent {winning_agent.id} "
                f"target={task.target_loc} bid_score={winner.score:.2f}"
            )

        if not self._in_reauction and not self._cbs_replan_ground(agents, known_map):
            print("  [CBS] Multi-agent replan failed; triggering reauction")
            self.trigger_global_reauction(agents, known_map)
