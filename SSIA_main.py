"""
SSIA test harness for the DARPA exploration simulation.

Uses the Availability-Constrained Sequential Single-Item Auction with
Global Reauction Trigger from SSIA_task_allocation.py on a scenario file,
defaulting to instances/test_0.txt.
"""

import argparse
from typing import List, Optional

from agents import Agent, AgentStatus, EventType, GroundAgent, DroneAgent
from maps import KnownMap, load_scenario
from planner import CBS
from SSIA_task_allocation import SequentialSingleItemAuctioneer
from visualizer import SimulationVisualizer


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
                print(
                    f"  [TASK ✓] Agent {agent.id} finished task"
                    f" {agent.current_task.task_id}"
                    f"  (observed {agent.current_task.target_loc})"
                )
            agent.current_task = None
            agent.path = []
            agent.status = AgentStatus.IDLE

    auctioneer.update(agents, known_map)

    for agent in agents:
        if agent.status == AgentStatus.REPLANNING:
            if not agent.replan(known_map) and verbose:
                print(f"  [WARN] Agent {agent.id} cannot reach task target yet")


def _do_microstep(agents, ground_truth, known_map, auctioneer, verbose: bool) -> bool:
    """
    Execute one global microstep for all agents.
    Returns True if any agent moved.
    """
    moved_any = False

    for agent in agents:
        ev = agent.step(known_map)
        if ev is None:
            continue

        moved_any = True

        if ev.kind == EventType.PATH_BLOCKED and verbose:
            blocked_at = ev.data["blocked_at"]
            print(f"  [BLOCKED] Agent {agent.id} — cell {blocked_at} is obstacle")

    if not moved_any:
        return False

    for agent in agents:
        agent.observe(ground_truth, known_map)

    _post_observation_updates(agents, auctioneer, known_map, verbose)
    return True


def run_simulation(
    path: str = "instances/test_0.txt",
    max_steps: int = 200,
    verbose: bool = True,
    use_vis: bool = True,
    agent_type: str = "dog",
):
    ground_truth, agent_starts = load_scenario(path)

    rows, cols = ground_truth.rows, ground_truth.cols
    known_map = KnownMap(rows, cols)
    auctioneer = SequentialSingleItemAuctioneer()
    planner = CBS(rows, cols)

    agent_cls = DroneAgent if agent_type == "drone" else GroundAgent
    obs_radius = 2 if agent_type == "drone" else 1
    agents: List[Agent] = [
        agent_cls(agent_id=i, start=start, obs_radius=obs_radius, planner=planner)
        for i, start in enumerate(agent_starts)
    ]

    print("=" * 60)
    print("  SSIA Exploration Simulation")
    print("  allocator=SequentialSingleItemAuctioneer")
    print(f"  {len(agents)} agent(s) type={agent_type}   map {rows}×{cols}   [{path}]")
    print("=" * 60)

    vis = SimulationVisualizer(ground_truth) if use_vis else None

    # Initial observation from start states
    for agent in agents:
        agent.observe(ground_truth, known_map)
    _post_observation_updates(agents, auctioneer, known_map, verbose)

    for step in range(max_steps):
        if verbose:
            statuses = "  ".join(
                f"A{a.id}@{a.pos}[{a.status.name[0]}]" for a in agents
            )
            print(f"\n─── Step {step:3d}  {statuses}  {auctioneer.stats()} ───")

        if vis is not None:
            vis.update(known_map, agents, step, auctioneer.stats())

        # Global microstep 1: everyone moves once
        moved_any = _do_microstep(
            agents, ground_truth, known_map, auctioneer, verbose
        )

        # Global microstep 2: drones only
        if agent_type == "drone":
            drone_agents = [a for a in agents if isinstance(a, DroneAgent)]
            if drone_agents:
                moved_any = (
                    _do_microstep(
                        drone_agents, ground_truth, known_map, auctioneer, verbose
                    )
                    or moved_any
                )

        if auctioneer.all_complete and all(a.status == AgentStatus.IDLE for a in agents):
            if vis is not None:
                vis.update(known_map, agents, step, auctioneer.stats())
            print(f"\n[DONE] All {auctioneer.stats()} — finished in {step + 1} steps.")
            break

        if not moved_any and verbose:
            print("  [STALL] No agents moved this step")

    else:
        print(f"\n[TIMEOUT] Simulation ended after {max_steps} steps.")

    if vis is not None:
        vis.finalize(auctioneer.stats())

    return known_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSIA exploration simulation")
    parser.add_argument(
        "path",
        nargs="?",
        default="instances/test_0.txt",
        help="scenario file to load (default: instances/test_0.txt)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        metavar="N",
        help="maximum simulation steps (default: 200)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-step output",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="disable matplotlib visualization",
    )
    parser.add_argument(
        "--agent-type",
        choices=["dog", "drone"],
        default="dog",
        help="agent class to instantiate (default: dog)",
    )
    args = parser.parse_args()

    run_simulation(
        path=args.path,
        max_steps=args.steps,
        verbose=not args.quiet,
        use_vis=not args.no_vis,
        agent_type=args.agent_type,
    )