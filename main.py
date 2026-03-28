"""
DARPA Multi-Agent Exploration Simulation — top-level event loop.
"""

import argparse
from typing import Optional

from agents import DroneAgent, GroundAgent
from maps import KnownMap, load_new_scenario
from planner import CBS
from sim_types import AgentStatus, AgentType, EventType
from naive_task_allocation import NaiveTaskAuctioneer
from visualizer import SimulationVisualizer


# ===========================================================================
# Simulation event loop
# ===========================================================================

def _post_observation_updates(agents, auctioneer, known_map, verbose: bool) -> None:
    """Common bookkeeping after agents have observed the map."""
    new_tasks = auctioneer.add_frontier_tasks(known_map)
    if new_tasks and verbose:
        print(f"  [FRONTIER] +{new_tasks} exploration task(s) queued")

    swept = auctioneer.sweep_completions(known_map)
    if swept and verbose:
        print(f"  [SWEPT]   {swept} task(s) observed as collateral")

    for agent in agents:
        if agent.current_task and agent.current_task.completed:
            if verbose:
                print(f"  [TASK ✓] Agent {agent.id} finished task"
                      f" {agent.current_task.task_id}"
                      f"  (observed {agent.current_task.target_loc})")
            agent.current_task = None
            agent.path = []
            agent.status = AgentStatus.IDLE

    auctioneer.auction(agents, known_map)

    for agent in agents:
        if agent.status == AgentStatus.REPLANNING:
            if not agent.replan(known_map) and verbose:
                print(f"  [WARN] Agent {agent.id} cannot reach task target yet")


def run_simulation(
    path: str = "generated/darpa1.txt",
    max_steps: int = 200,
    verbose: bool = True,
) -> KnownMap:
    
    ground_truth = load_new_scenario(path)

    rows, cols = ground_truth.rows, ground_truth.cols
    known_map = KnownMap(rows, cols)
    auctioneer = NaiveTaskAuctioneer()
    planner = CBS(rows, cols)

    agents = []
    for i, (sr, sc, atype) in enumerate(ground_truth.agent_starts):
        if atype == AgentType.DRONE.value:
            agents.append(DroneAgent(agent_id=i, start=(sr, sc), planner=planner))
        else:
            agents.append(GroundAgent(agent_id=i, start=(sr, sc), planner=planner))

    print("=" * 52)
    print("  DARPA Exploration Simulation  (task-queue driven)")
    print(f"  {len(agents)} agent(s)   map {rows}×{cols}"
          + (f"   [{path}]" if path else ""))
    print("=" * 52)

    vis = SimulationVisualizer(ground_truth)

    # Bootstrap
    for agent in agents:
        agent.observe(ground_truth, known_map)
    _post_observation_updates(agents, auctioneer, known_map, verbose)

    for step in range(max_steps):
        if verbose:
            statuses = "  ".join(
                f"A{a.id}@{a.pos}[{a.status.name[0]}]" for a in agents
            )
            print(f"\n─── Step {step:3d}  {statuses}  {auctioneer.stats()} ───")

        vis.update(known_map, agents, step, auctioneer.stats())

        # Global microsteps: all agents move once, then drones move a second time.
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
                        print(f"  [BLOCKED] Agent {agent.id} — cell {blocked_at}"
                              f" is obstacle; releasing task back to queue")
                    if agent.current_task:
                        agent.current_task.assigned_to = None
                        agent.current_task = None
                    agent.path = []
                    agent.status = AgentStatus.IDLE

            # No one moved in this microstep, so skip the expensive update.
            if not moved_any:
                continue

            for agent in agents:
                agent.observe(ground_truth, known_map)

            _post_observation_updates(agents, auctioneer, known_map, verbose)

        if auctioneer.all_complete and all(a.status == AgentStatus.IDLE for a in agents):
            vis.update(known_map, agents, step, auctioneer.stats())
            print(f"\n[DONE] All {auctioneer.stats()} — finished in {step + 1} steps.")
            break

    else:
        print(f"\n[TIMEOUT] Simulation ended after {max_steps} steps.")

    vis.finalize(auctioneer.stats())
    return known_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARPA exploration simulation")
    parser.add_argument("path", nargs="?", default="generated/darpa1.txt",
                        help="scenario file to load (default: built-in 10×10 map)")
    parser.add_argument("--steps", type=int, default=200, metavar="N",
                        help="maximum simulation steps (default: 200)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-step output")
    args = parser.parse_args()
    run_simulation(path=args.path, max_steps=args.steps, verbose=not args.quiet)
