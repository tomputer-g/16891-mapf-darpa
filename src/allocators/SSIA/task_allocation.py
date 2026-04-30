"""
Auction and task allocation for the DARPA exploration simulation.

Availability-constrained sequential single-item auction with
repair-vs-reauction control.

Tasks are auctioned one at a time. The allocator stores winner and
runner-up metadata on assigned tasks so that a blocked or infeasible path
can be handled locally first:

- attempt cheap local repair
- compare repaired retained quality against the stored runner-up
- preserve the assignment if it is still competitive
- escalate to global reauction only when repair fails, the retained option
  loses beyond slack, or coordinated CBS repair fails

During full reauction, agents already dwelling on incomplete triage tasks are
preserved instead of having that in-progress work reset.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from src.agents import Agent, AgentStatus, DroneAgent, plan_path
from src.maps import KnownMap
from src.planner import CBS
from src.sim_types import AgentType
from src.tasks import ExplorationTask, TriageTask, Task
from src.allocators.base_task_allocation import (
    BaseSSIATaskAuctioneer, Bid, _TRIAGE_DWELL,
)


_TASK_REWARD = {
    ExplorationTask: 1,
    TriageTask: 3,
}


class SequentialSingleItemAuctioneer(BaseSSIATaskAuctioneer):
    """
    Availability-constrained sequential single-item auctioneer.

    Inherits task bookkeeping, execution-cost helpers, CBS replanning,
    repair-vs-reauction heuristic, conflict detection, and the standard
    update loop from BaseSSIATaskAuctioneer.  Only the bid formula and
    auction loop are specific to SSIA.
    """

    _task_reward = _TASK_REWARD

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------
    def compute_bid(
        self,
        agent: Agent,
        task: Task,
        known_map: KnownMap,
    ) -> Optional[Bid]:
        """Bid = reward / (path_length + dwell_time)."""
        use_drone_path = isinstance(agent, DroneAgent)
        path = plan_path(known_map, agent.pos, task.target_loc, drone=use_drone_path)
        if not path:
            return None
        execution_cost = self._execution_cost(agent, task, path)
        reward = _TASK_REWARD.get(type(task), 1)
        return Bid(
            agent_id=agent.id,
            task_id=task.task_id,
            score=reward / execution_cost,
            path=path,
            execution_cost=execution_cost,
            effective_reward=reward,
        )

    # ------------------------------------------------------------------
    # Auction
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

            print(
                f"  [AUCTION] Task {task.task_id} -> Agent {winning_agent.id} "
                f"target={task.target_loc} bid_score={winner.score:.2f}"
            )

        if not self._in_reauction and not self._cbs_replan_ground(agents, known_map):
            print("  [CBS] Multi-agent replan failed; triggering reauction")
            self.trigger_global_reauction(agents, known_map)
