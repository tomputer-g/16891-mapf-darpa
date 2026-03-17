"""
DARPA Multi-Agent Exploration Simulation — top-level event loop.

The simulation terminates when the task queue is exhausted — there is no
fixed goal location.  Exploration tasks are generated dynamically as the
agent reveals frontier cells; each task is complete once the auctioneer
confirms an agent has observed its target cell.

Module layout
-------------
  maps.py        – ObservationState, GroundTruthMap, KnownMap, build_default_scenario
  tasks.py       – Task, ExplorationTask, TaskAuctioneer
  agents.py      – AgentStatus, EventType, Event, plan_path, Agent
  visualizer.py  – SimulationVisualizer
  main.py        – run_simulation() (this file)

Extension points are marked with  # EXTEND:  comments throughout.
"""

import argparse
from typing import Optional

from agents import Agent, AgentStatus, EventType, Agent_Dog, Agent_Drone
from maps import KnownMap, build_default_scenario, load_scenario
from tasks import TaskAuctioneer
from visualizer import SimulationVisualizer


# ===========================================================================
# Simulation event loop
# ===========================================================================

def run_simulation(
    path:      Optional[str] = None,
    max_steps: int           = 200,
    verbose:   bool          = True,
) -> KnownMap:
    """
    Main event loop.  Terminates when the task queue is exhausted.

    Parameters
    ----------
    path      : path to a scenario file; uses the built-in default map if None.
    max_steps : hard step limit before timing out.
    verbose   : print per-step diagnostics.

    Per-timestep order:
      1. Each agent observes surroundings            → KnownMap updated
      2. Auctioneer scans for new frontier cells     → ExplorationTasks added
      3. Active tasks checked for passive completion → agents may go IDLE
      4. Auctioneer runs an auction round            → IDLE agents get tasks
      5. Agents that need paths compute them         → status → NAVIGATING
      6. Agents step along their paths               → handle PATH_BLOCKED
      7. Termination check

    Returns the final KnownMap (useful for post-run analysis).
    """
    if path is not None:
        ground_truth, agent_starts = load_scenario(path)
    else:
        ground_truth, agent_starts = build_default_scenario()

    rows, cols = ground_truth.rows, ground_truth.cols
    known_map  = KnownMap(rows, cols)
    auctioneer = TaskAuctioneer()

    agents = [Agent_Drone(agent_id=i, start=start, obs_radius=2)
              for i, start in enumerate(agent_starts)]

    print("=" * 52)
    print("  DARPA Exploration Simulation  (task-queue driven)")
    print(f"  {len(agents)} agent(s)   map {rows}×{cols}"
          + (f"   [{path}]" if path else ""))
    print("=" * 52)

    vis = SimulationVisualizer(ground_truth)

    # ── Bootstrap ────────────────────────────────────────────────────────
    for agent in agents:
        agent.observe(ground_truth, known_map)
    auctioneer.add_frontier_tasks(known_map)
    auctioneer.auction(agents, known_map)
    for agent in agents:
        if agent.status == AgentStatus.REPLANNING:
            agent.replan(known_map)

    # ── Main loop ────────────────────────────────────────────────────────
    for step in range(max_steps):

        # Display ────────────────────────────────────────────────────────
        if verbose:
            statuses = "  ".join(
                f"A{a.id}@{a.pos}[{a.status.name[0]}]" for a in agents
            )
            print(f"\n─── Step {step:3d}  {statuses}  {auctioneer.stats()} ───")

        vis.update(known_map, agents, step, auctioneer.stats())

        # 1. Observe ─────────────────────────────────────────────────────
        for agent in agents:
            agent.observe(ground_truth, known_map)

        # 2. Discover new frontier tasks ─────────────────────────────────
        new_tasks = auctioneer.add_frontier_tasks(known_map)
        if new_tasks and verbose:
            print(f"  [FRONTIER] +{new_tasks} exploration task(s) queued")

        # 2b. Sweep all queued tasks for collateral observations ──────────
        # Cells passed through en-route to another target may already be
        # observed; mark them done now so they are never auctioned.
        swept = auctioneer.sweep_completions(known_map)
        if swept and verbose:
            print(f"  [SWEPT]   {swept} task(s) observed as collateral")

        # 3. Check whether any agent's current task was just completed ────
        # Use .completed directly — sweep_completions already called
        # check_completion (which only fires once) for all tasks.
        for agent in agents:
            if agent.current_task and agent.current_task.completed:
                print(f"  [TASK ✓] Agent {agent.id} finished task"
                      f" {agent.current_task.task_id}"
                      f"  (observed {agent.current_task.target_loc})")
                agent.current_task = None
                agent.path         = []
                agent.status       = AgentStatus.IDLE

        # 4. Auction ─────────────────────────────────────────────────────
        auctioneer.auction(agents, known_map)

        # 5. Replan ──────────────────────────────────────────────────────
        for agent in agents:
            if agent.status == AgentStatus.REPLANNING:
                if not agent.replan(known_map) and verbose:
                    print(f"  [WARN] Agent {agent.id} cannot reach task target yet")

        # 6. Step ────────────────────────────────────────────────────────
        for agent in agents:
            ev = agent.step(known_map)
            if ev is None:
                continue
            if ev.kind == EventType.PATH_BLOCKED:
                blocked_at = ev.data['blocked_at']
                print(f"  [BLOCKED] Agent {agent.id} — cell {blocked_at}"
                      f" is obstacle; releasing task back to queue")
                if agent.current_task:
                    agent.current_task.assigned_to = None
                    agent.current_task = None
                auctioneer.auction(agents, known_map)
                if agent.status == AgentStatus.REPLANNING:
                    agent.replan(known_map)

        # 7. Termination ─────────────────────────────────────────────────
        if auctioneer.all_complete and all(a.status == AgentStatus.IDLE
                                           for a in agents):
            vis.update(known_map, agents, step, auctioneer.stats())
            print(f"\n[DONE] All {auctioneer.stats()} — finished in {step + 1} steps.")
            break

    else:
        print(f"\n[TIMEOUT] Simulation ended after {max_steps} steps.")

    vis.finalize(auctioneer.stats())
    return known_map


# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARPA exploration simulation")
    parser.add_argument("path", nargs="?", default=None,
                        help="scenario file to load (default: built-in 10×10 map)")
    parser.add_argument("--steps", type=int, default=200, metavar="N",
                        help="maximum simulation steps (default: 200)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-step output")
    args = parser.parse_args()
    run_simulation(path=args.path, max_steps=args.steps, verbose=not args.quiet)

