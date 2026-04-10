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

from agents import Agent, AgentStatus, DroneAgent, plan_path
from maps import KnownMap, ObservationState
from planner import CBS
from sim_types import AgentType
from tasks import ExplorationTask, TriageTask, Task


# Reward values used in the bid formula: reward / (path_length + dwell_time)
_TASK_REWARD = {
    ExplorationTask: 1,
    TriageTask: 5,
}

# Dwell steps by agent type (must stay in sync with SSIA_main._TRIAGE_DWELL)
_TRIAGE_DWELL = {AgentType.GROUND: 2, AgentType.DRONE: 4}


@dataclass
class Bid:
    """Single sealed bid for one task from one agent."""
    agent_id: int
    task_id: int
    score: float
    path: List[Tuple[int, int]]


class SequentialSingleItemAuctioneer:
    """
    Availability-constrained sequential single-item auctioneer.
    """

    def __init__(self, cbs: Optional[CBS] = None) -> None:
        self._tasks: List[Task] = []
        self._known_locs: Set[Tuple[int, int]] = set()
        self._triage_locs: Set[Tuple[int, int]] = set()
        self._invest_locs: Set[Tuple[int, int]] = set()
        self.reauction_count: int = 0
        self._cbs: Optional[CBS] = cbs
        self._in_reauction: bool = False

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
        use_drone_path = isinstance(agent, DroneAgent)

        path = plan_path(
            known_map,
            agent.pos,
            task.target_loc,
            drone=use_drone_path,
        )
        if not path:
            return None

        path_len = max(1, len(path) - 1)
        # Drones move twice per step, so effective travel time is halved
        if use_drone_path:
            path_len = path_len / 2
        reward = _TASK_REWARD.get(type(task), 1)

        # Dwell time: triage tasks require dwelling at the target
        if isinstance(task, TriageTask):
            if getattr(task, '_is_investigation', False):
                dwell = 1
            else:
                dwell = _TRIAGE_DWELL.get(agent.agent_type, task.dwell_steps)
        else:
            dwell = 0

        score = reward / (path_len + dwell)
        return Bid(
            agent_id=agent.id,
            task_id=task.task_id,
            score=score,
            path=path,
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
            winning_agent = next(a for a in available_agents if a.id == winner.agent_id)

            task.assigned_to = winning_agent.id
            winning_agent.assign_task(task)
            winning_agent.path = winner.path
            winning_agent.status = AgentStatus.NAVIGATING
            available_agents.remove(winning_agent)

            print(
                f"  [AUCTION] Task {task.task_id} -> Agent {winning_agent.id} "
                f"target={task.target_loc} bid_score={winner.score:.2f}"
            )

        # CBS multi-agent replan for all navigating ground agents
        if not self._in_reauction and not self._cbs_replan_ground(agents, known_map):
            print("  [CBS] Multi-agent replan failed; triggering reauction")
            self.trigger_global_reauction(agents, known_map)

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
        Release all incomplete assignments and run a fresh auction round.
        """
        self.reauction_count += 1

        for task in self._tasks:
            if not task.completed:
                task.assigned_to = None
                # Reset dwell progress so the new winner starts fresh
                if isinstance(task, TriageTask):
                    task.progress = 0

        for agent in agents:
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

        # 4. Release agents whose current task is finished
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
                agent.current_task = None
                agent.path = []
                agent.status = AgentStatus.IDLE

        # 4b. Release agents whose path became infeasible (no global reauction)
        for agent in agents:
            if self._path_infeasible(agent, known_map):
                if verbose:
                    print(f"  [INFEASIBLE] Agent {agent.id} path to "
                          f"{agent.current_task.target_loc} blocked; releasing task")
                agent.current_task.assigned_to = None
                agent.current_task = None
                agent.path = []
                agent.status = AgentStatus.IDLE

        # 5. Reauction on conflict, otherwise normal auction
        if self.should_trigger_reauction(agents, known_map):
            self.trigger_global_reauction(agents, known_map)
        else:
            self.auction(agents, known_map)