"""
DARPA Multi-Agent Exploration Simulation — top-level event loop.
"""

import argparse

from src.agents import DroneAgent, GroundAgent
from src.maps import KnownMap, load_new_scenario
from src.planner import CBS
from src.sim_types import AgentStatus, AgentType, EventType
from src.tasks import TriageTask
from src.allocators.Naive.naive_task_allocation import NaiveTaskAuctioneer
from src.allocators.base_main import SimulationHarness
from src.visualizer import SimulationVisualizer


class NaiveHarness(SimulationHarness):
    """
    Simulation harness for the Naive greedy baseline.

    Overrides run_simulation because the Naive loop differs from the
    SSIA-family template: auction is deferred until after both microsteps,
    and blocked paths are released directly (no repair-vs-reauction).
    Inherits _update_triage_progress from SimulationHarness.
    """

    name = "DARPA Exploration Simulation  (task-queue driven)"

    def _make_auctioneer(self, planner: CBS) -> NaiveTaskAuctioneer:
        return NaiveTaskAuctioneer(cbs=planner)

    # ------------------------------------------------------------------
    # Naive-specific observation update (inline, with deferred auction)
    # ------------------------------------------------------------------
    def _naive_post_observation_updates(
        self, agents, auctioneer, known_map, ground_truth, verbose: bool,
        run_auction: bool = True,
    ) -> None:
        new_explore = auctioneer.add_frontier_tasks(known_map)
        if new_explore and verbose:
            print(f"  [FRONTIER] +{new_explore} exploration task(s) queued")

        new_triage = auctioneer.add_revealed_triage_tasks(known_map, ground_truth)
        if new_triage and verbose:
            print(f"  [TASKS]   +{new_triage} task(s) revealed")

        new_bldg_triage = auctioneer.add_confirmed_building_triage(known_map)
        if new_bldg_triage and verbose:
            print(f"  [TRIAGE]  +{new_bldg_triage} occupied building triage task(s) confirmed")

        swept = auctioneer.sweep_completions(known_map)
        if swept and verbose:
            print(f"  [SWEPT]   {swept} task(s) observed as collateral")

        for agent in agents:
            if agent.current_task and agent.current_task.completed:
                if verbose:
                    task = agent.current_task
                    if isinstance(task, TriageTask) and getattr(task, '_is_investigation', False):
                        print(
                            f"  [INVESTIGATED] Agent {agent.id} checked building "
                            f"at {task.target_loc}"
                        )
                    elif isinstance(task, TriageTask):
                        print(
                            f"  [TRIAGE DONE] Agent {agent.id} finished triage task "
                            f"{task.task_id} at {task.target_loc}"
                        )
                    else:
                        print(
                            f"  [TASK DONE] Agent {agent.id} finished task "
                            f"{task.task_id} at {task.target_loc}"
                        )
                agent.current_task = None
                agent.path = []
                agent.status = AgentStatus.IDLE

        if run_auction:
            auctioneer.auction(agents, known_map)

        for agent in agents:
            if agent.status == AgentStatus.REPLANNING:
                if not agent.replan(known_map) and verbose:
                    print(f"  [WARN] Agent {agent.id} cannot reach task target yet")

    # ------------------------------------------------------------------
    # Full override: Naive-specific run loop
    # ------------------------------------------------------------------
    def run_simulation(
        self,
        path: str = "generated/darpa1.txt",
        max_steps: int = 200,
        verbose: bool = True,
        use_vis: bool = True,
    ) -> KnownMap:
        ground_truth = load_new_scenario(path)
        rows, cols = ground_truth.rows, ground_truth.cols
        known_map = KnownMap(rows, cols)
        planner = CBS(rows, cols)
        auctioneer = self._make_auctioneer(planner)

        agents = []
        for i, (sr, sc, atype) in enumerate(ground_truth.agent_starts):
            if atype == AgentType.DRONE.value:
                agents.append(DroneAgent(agent_id=i, start=(sr, sc), planner=planner))
            else:
                agents.append(GroundAgent(agent_id=i, start=(sr, sc), planner=planner))

        print("=" * 52)
        print(f"  {self.name}")
        print(f"  {len(agents)} agent(s)   map {rows}x{cols}"
              + (f"   [{path}]" if path else ""))
        print("=" * 52)

        vis = SimulationVisualizer(ground_truth) if use_vis else None

        for agent in agents:
            agent.observe(ground_truth, known_map)
        self._naive_post_observation_updates(agents, auctioneer, known_map, ground_truth, verbose)

        for step in range(max_steps):
            if verbose:
                statuses = "  ".join(
                    f"A{a.id}@{a.pos}[{a.status.name[0]}]" for a in agents
                )
                print(f"\n--- Step {step:3d}  {statuses}  {auctioneer.stats()} ---")

            if vis:
                vis.update(known_map, agents, step, auctioneer.stats())

            for microstep in range(2):
                moved_any = False

                for agent in agents:
                    if microstep == 1 and not isinstance(agent, DroneAgent):
                        continue

                    ev = agent.step(known_map)
                    if ev is None:
                        continue

                    moved_any = True

                    if ev.kind == EventType.PATH_BLOCKED:
                        blocked_at = ev.data['blocked_at']
                        if verbose:
                            print(f"  [BLOCKED] Agent {agent.id} - cell {blocked_at}"
                                  f" is obstacle; releasing task back to queue")
                        if agent.current_task:
                            agent.current_task.assigned_to = None
                            agent.current_task = None
                        agent.path = []
                        agent.status = AgentStatus.IDLE

                if not moved_any:
                    continue

                for agent in agents:
                    agent.observe(ground_truth, known_map)

                self._naive_post_observation_updates(
                    agents, auctioneer, known_map, ground_truth, verbose,
                    run_auction=False,
                )

            self._update_triage_progress(agents, verbose)

            for agent in agents:
                if agent.current_task and agent.current_task.completed:
                    if verbose:
                        task = agent.current_task
                        if isinstance(task, TriageTask) and getattr(task, '_is_investigation', False):
                            print(
                                f"  [INVESTIGATED] Agent {agent.id} checked building "
                                f"at {task.target_loc}"
                            )
                        elif isinstance(task, TriageTask):
                            print(
                                f"  [TRIAGE DONE] Agent {agent.id} finished triage task "
                                f"{task.task_id} at {task.target_loc}"
                            )
                    agent.current_task = None
                    agent.path = []
                    agent.status = AgentStatus.IDLE

            auctioneer.auction(agents, known_map)

            for agent in agents:
                if agent.status == AgentStatus.REPLANNING:
                    if not agent.replan(known_map) and verbose:
                        print(f"  [WARN] Agent {agent.id} cannot reach task target yet")

            if auctioneer.all_complete and all(a.status == AgentStatus.IDLE for a in agents):
                if vis:
                    vis.update(known_map, agents, step, auctioneer.stats())
                print(f"\n[DONE] All {auctioneer.stats()} -- finished in {step + 1} steps.")
                break

        else:
            print(f"\n[TIMEOUT] Simulation ended after {max_steps} steps.")

        if vis:
            vis.finalize(auctioneer.stats())
        return known_map


def run_simulation(
    path: str = "generated/darpa1.txt",
    max_steps: int = 200,
    verbose: bool = True,
    use_vis: bool = True,
) -> KnownMap:
    return NaiveHarness().run_simulation(path, max_steps, verbose, use_vis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARPA exploration simulation")
    parser.add_argument("path", nargs="?", default="generated/darpa1.txt",
                        help="scenario file to load (default: generated/darpa1.txt)")
    parser.add_argument("--steps", type=int, default=200, metavar="N",
                        help="maximum simulation steps (default: 200)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-step output")
    parser.add_argument("--no-vis", action="store_true",
                        help="disable matplotlib visualization")
    args = parser.parse_args()
    run_simulation(path=args.path, max_steps=args.steps, verbose=not args.quiet,
                   use_vis=not args.no_vis)
