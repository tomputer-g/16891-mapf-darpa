"""
DARPA Multi-Agent Exploration Simulation — top-level event loop.
"""

import argparse
from typing import Optional

from agents import DroneAgent, GroundAgent
from maps import KnownMap, load_new_scenario
from sim_types import AgentStatus, AgentType, EventType
from tasks import TaskAuctioneer, TriageTask
from visualizer import SimulationVisualizer

# Dwell steps required per agent type for triage tasks
_TRIAGE_DWELL = {AgentType.GROUND: 2, AgentType.DRONE: 4}


# ===========================================================================
# Simulation event loop
# ===========================================================================
def _update_triage_progress(agents, verbose: bool) -> None:
    for agent in agents:
        task = agent.current_task
        if task is None or not isinstance(task, TriageTask):
            continue

        # Investigation tasks always complete in 1 step
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

from tasks import TriageTask, ExplorationTask

def _post_observation_updates(agents, auctioneer, known_map, ground_truth, verbose: bool) -> None:
    """Common bookkeeping after agents have observed the map."""

    # 1. Add exploration frontier tasks
    new_explore = auctioneer.add_frontier_tasks(known_map)
    if new_explore and verbose:
        print(f"  [FRONTIER] +{new_explore} exploration task(s) queued")

    # 2. Add newly revealed triage / investigation tasks
    new_triage = auctioneer.add_revealed_triage_tasks(known_map, ground_truth)
    if new_triage and verbose:
        print(f"  [TASKS]   +{new_triage} task(s) revealed")

    # 3. Update triage progress for agents dwelling at triage targets
    _update_triage_progress(agents, verbose)

    # 4. Promote investigated buildings to triage if ground confirmed occupied
    new_bldg_triage = auctioneer.add_confirmed_building_triage(known_map)
    if new_bldg_triage and verbose:
        print(f"  [TRIAGE]  +{new_bldg_triage} occupied building triage task(s) confirmed")

    # 5. Sweep map-based completions (mainly exploration tasks)
    swept = auctioneer.sweep_completions(known_map)
    if swept and verbose:
        print(f"  [SWEPT]   {swept} task(s) observed as collateral")

    # 5. Release agents whose current task is finished
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

    # 6. Run auction for any idle agents
    auctioneer.auction(agents, known_map)

    # 7. Replan if needed
    for agent in agents:
        if agent.status == AgentStatus.REPLANNING:
            if not agent.replan(known_map) and verbose:
                print(f"  [WARN] Agent {agent.id} cannot reach task target yet")


def run_simulation(
    path: Optional[str] = "generated/darpa1.txt",
    max_steps: int = 200,
    verbose: bool = True,
    use_vis: bool = True,
) -> KnownMap:
    
    ground_truth = load_new_scenario(path or "generated/darpa1.txt")

    rows, cols = ground_truth.rows, ground_truth.cols
    known_map = KnownMap(rows, cols)
    auctioneer = TaskAuctioneer()

    agents = []
    for i, (sr, sc, atype) in enumerate(ground_truth.agent_starts):
        if atype == AgentType.DRONE.value:
            agents.append(DroneAgent(agent_id=i, start=(sr, sc)))
        else:
            agents.append(GroundAgent(agent_id=i, start=(sr, sc)))

    print("=" * 52)
    print("  DARPA Exploration Simulation  (task-queue driven)")
    print(f"  {len(agents)} agent(s)   map {rows}x{cols}"
          + (f"   [{path}]" if path else ""))
    print("=" * 52)

    vis = SimulationVisualizer(ground_truth) if use_vis else None

    # Bootstrap
    for agent in agents:
        agent.observe(ground_truth, known_map)
    _post_observation_updates(agents, auctioneer, known_map, ground_truth, verbose)

    for step in range(max_steps):
        if verbose:
            statuses = "  ".join(
                f"A{a.id}@{a.pos}[{a.status.name[0]}]" for a in agents
            )
            print(f"\n--- Step {step:3d}  {statuses}  {auctioneer.stats()} ---")

        if vis:
            vis.update(known_map, agents, step, auctioneer.stats())

        # Global microsteps: all agents move once, then drones move a second time.
        observed_this_step = False
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

            # No one moved in this microstep, so skip the expensive update.
            if not moved_any:
                continue

            observed_this_step = True
            for agent in agents:
                agent.observe(ground_truth, known_map)

            _post_observation_updates(agents, auctioneer, known_map, ground_truth, verbose)

        # Always tick triage progress even when no agent moved (agents
        # dwelling at their target don't produce a step event).
        if not observed_this_step:
            _update_triage_progress(agents, verbose)
            # Release agents whose triage just finished
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
            # Run auction for newly idle agents
            auctioneer.auction(agents, known_map)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARPA exploration simulation")
    parser.add_argument("path", nargs="?", default=None,
                        help="scenario file to load (default: built-in 10×10 map)")
    parser.add_argument("--steps", type=int, default=200, metavar="N",
                        help="maximum simulation steps (default: 200)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-step output")
    parser.add_argument("--no-vis", action="store_true",
                        help="disable matplotlib visualization")
    args = parser.parse_args()
    run_simulation(path=args.path, max_steps=args.steps, verbose=not args.quiet,
                   use_vis=not args.no_vis)
