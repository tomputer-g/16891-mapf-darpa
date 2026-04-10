"""
SSIA test harness for the DARPA exploration simulation.

Uses the Availability-Constrained Sequential Single-Item Auction with
Global Reauction Trigger from SSIA_task_allocation.py on a scenario file,
defaulting to instances/test_0.txt.
"""

import argparse
from typing import List, Optional

from agents import Agent, AgentStatus, EventType, GroundAgent, DroneAgent
from maps import KnownMap, load_new_scenario
from sim_types import AgentType
from planner import CBS
from SSIA_task_allocation import SequentialSingleItemAuctioneer
from tasks import TriageTask
from visualizer import SimulationVisualizer

# Dwell steps required per agent type for triage tasks
_TRIAGE_DWELL = {AgentType.GROUND: 2, AgentType.DRONE: 4}


def _update_triage_progress(agents, verbose: bool) -> None:
    for agent in agents:
        task = agent.current_task
        if task is None or not isinstance(task, TriageTask):
            continue

        if getattr(task, '_is_investigation', False):
            dwell_needed = 1
        else:
            dwell_needed = _TRIAGE_DWELL.get(agent.agent_type, task.dwell_steps)

        if agent.pos == task.target_loc:
            task.progress += 1
            if verbose:
                if getattr(task, '_is_investigation', False):
                    print(f"  [INVESTIGATE] Agent {agent.id} checking building at {task.target_loc}")
                else:
                    print(f"  [TRIAGE] Agent {agent.id} working on task {task.task_id} "
                          f"progress={task.progress}/{dwell_needed}")
            if task.progress >= dwell_needed:
                task.completed = True
                if verbose and not getattr(task, '_is_investigation', False):
                    print(f"  [TRIAGE DONE] Agent {agent.id} completed triage task {task.task_id}")
        else:
            if task.progress != 0:
                task.progress = 0


def _post_observation_updates(agents, auctioneer, known_map, ground_truth, verbose: bool) -> None:
    """Common bookkeeping after agents have observed the map."""
    # Delegate task creation, sweep, release, reauction, and auction
    # to the auctioneer's update() — single responsibility.
    auctioneer.update(agents, known_map, ground_truth=ground_truth,
                      verbose=verbose)

    # 3. Replan if needed
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
            if agent.current_task:
                agent.current_task.assigned_to = None
                agent.current_task = None
            agent.path = []
            agent.status = AgentStatus.IDLE

    if not moved_any:
        return False

    for agent in agents:
        agent.observe(ground_truth, known_map)

    _post_observation_updates(agents, auctioneer, known_map, ground_truth, verbose)
    return True


def run_simulation(
    path: str = "generated/darpa1.txt",
    max_steps: int = 200,
    verbose: bool = True,
    use_vis: bool = True,
):
    ground_truth = load_new_scenario(path)

    rows, cols = ground_truth.rows, ground_truth.cols
    known_map = KnownMap(rows, cols)
    planner = CBS(rows, cols)
    auctioneer = SequentialSingleItemAuctioneer(cbs=planner)

    agents: List[Agent] = []
    for i, (sr, sc, atype) in enumerate(ground_truth.agent_starts):
        if atype == AgentType.DRONE.value:
            agents.append(DroneAgent(agent_id=i, planner=planner, start=(sr, sc)))
        else:
            agents.append(GroundAgent(agent_id=i, planner=planner, start=(sr, sc)))

    print("=" * 60)
    print("  SSIA Exploration Simulation")
    print("  allocator=SequentialSingleItemAuctioneer")
    print(f"  {len(agents)} agent(s)   map {rows}x{cols}   [{path}]")
    print("=" * 60)

    vis = SimulationVisualizer(ground_truth) if use_vis else None

    # Initial observation from start states
    for agent in agents:
        agent.observe(ground_truth, known_map)
    _post_observation_updates(agents, auctioneer, known_map, ground_truth, verbose)

    for step in range(max_steps):
        if verbose:
            statuses = "  ".join(
                f"A{a.id}@{a.pos}[{a.status.name[0]}]" for a in agents
            )
            print(f"\n--- Step {step:3d}  {statuses}  {auctioneer.stats()} ---")

        if vis is not None:
            vis.update(known_map, agents, step, auctioneer.stats())

        # Global microstep 1: everyone moves once
        moved_any = _do_microstep(
            agents, ground_truth, known_map, auctioneer, verbose
        )

        # Global microstep 2: drones get a second move
        drone_agents = [a for a in agents if isinstance(a, DroneAgent)]
        if drone_agents:
            moved_any = (
                _do_microstep(
                    drone_agents, ground_truth, known_map, auctioneer, verbose
                )
                or moved_any
                )

        # Always tick triage progress (agents dwelling at targets don't move)
        _update_triage_progress(agents, verbose)
        auctioneer.update(agents, known_map, ground_truth=ground_truth,
                          verbose=verbose)

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
        default="generated/darpa1.txt",
        help="scenario file to load (default: generated/darpa1.txt)",
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
    args = parser.parse_args()

    run_simulation(
        path=args.path,
        max_steps=args.steps,
        verbose=not args.quiet,
        use_vis=not args.no_vis,
    )