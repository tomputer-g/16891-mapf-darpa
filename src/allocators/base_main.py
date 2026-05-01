"""
Abstract base class for DARPA exploration simulation harnesses.

Provides triage-progress tracking, microstep execution, and the main run
loop shared by the SSIA-family allocators.  Subclasses supply the auctioneer
factory and override the two extension hooks to inject algorithm-specific
behaviour at the replan-failure and post-triage-update points.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.agents import Agent, AgentStatus, EventType, GroundAgent, DroneAgent
from src.maps import KnownMap, load_new_scenario
from src.planner import CBS
from src.sim_types import AgentType
from src.tasks import TriageTask
from src.allocators.base_task_allocation import BaseSSIATaskAuctioneer

_TRIAGE_DWELL = {AgentType.GROUND: 2, AgentType.DRONE: 4}


class SimulationHarness(ABC):
    """
    Template for one DARPA exploration simulation run.

    Subclasses set ``name`` / ``allocator_desc`` and implement
    ``_make_auctioneer``.  Two hooks let subclasses specialise behaviour
    without duplicating the entire run loop:

    ``_handle_replan_failure``  -- what to do when ``agent.replan()`` fails
        inside ``_post_observation_updates`` (default: warn).
        SSICA overrides to call ``handle_invalidated_assignment``.

    ``_post_triage_update``     -- extra logic after the triage-progress tick
        and the end-of-step ``auctioneer.update()`` (default: no-op).
        SSICA overrides to handle agents stuck in REPLANNING.
    """

    name: str = "DARPA Exploration Simulation"
    allocator_desc: str = ""

    # ------------------------------------------------------------------
    # Abstract factory
    # ------------------------------------------------------------------
    @abstractmethod
    def _make_auctioneer(self, planner: CBS) -> BaseSSIATaskAuctioneer:
        """Return a fresh auctioneer instance for this run."""
        ...

    # ------------------------------------------------------------------
    # Triage progress (identical across all allocators)
    # ------------------------------------------------------------------
    def _update_triage_progress(self, agents, verbose: bool) -> None:
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

    # ------------------------------------------------------------------
    # Extension hooks
    # ------------------------------------------------------------------
    def _handle_replan_failure(self, agent, agents, auctioneer,
                                known_map, verbose) -> None:
        """Called when agent.replan() returns False during _post_observation_updates.
        Default: log a warning. SSICA overrides to call handle_invalidated_assignment."""
        if verbose:
            print(f"  [WARN] Agent {agent.id} cannot reach task target yet")

    def _post_triage_update(self, agents, auctioneer, known_map, verbose) -> None:
        """Called after triage progress tick + end-of-step auctioneer.update().
        Default: no-op. SSICA overrides to handle agents stuck in REPLANNING."""
        pass

    # ------------------------------------------------------------------
    # Shared observation update (delegates to auctioneer.update)
    # ------------------------------------------------------------------
    def _post_observation_updates(self, agents, auctioneer, known_map,
                                   ground_truth, verbose: bool) -> None:
        auctioneer.update(agents, known_map, ground_truth=ground_truth, verbose=verbose)
        for agent in agents:
            if agent.status == AgentStatus.REPLANNING:
                if not agent.replan(known_map):
                    self._handle_replan_failure(agent, agents, auctioneer, known_map, verbose)

    # ------------------------------------------------------------------
    # Microstep executor
    # ------------------------------------------------------------------
    def _do_microstep(self, agents, ground_truth, known_map,
                      auctioneer, verbose: bool) -> bool:
        """Execute one movement microstep for the given agents.
        Returns True if any agent moved."""
        moved_any = False
        blocked_agents = []
        for agent in agents:
            ev = agent.step(known_map)
            if ev is None:
                continue
            if ev.kind == EventType.STEP_COMPLETE:
                moved_any = True
                continue
            if ev.kind == EventType.PATH_BLOCKED:
                blocked_agents.append((agent, ev.data["blocked_at"]))
        for agent, blocked_at in blocked_agents:
            if verbose:
                print(f"  [BLOCKED] Agent {agent.id} — cell {blocked_at} is obstacle")
            reauctioned = auctioneer.handle_invalidated_assignment(
                agent, agents, known_map, verbose=verbose, reason="path-blocked"
            )
            if reauctioned:
                break
        if not moved_any:
            return False
        for agent in agents:
            agent.observe(ground_truth, known_map)
        self._post_observation_updates(agents, auctioneer, known_map, ground_truth, verbose)
        return True

    # ------------------------------------------------------------------
    # Main simulation loop
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

        agents: List[Agent] = []
        for i, (sr, sc, atype) in enumerate(ground_truth.agent_starts):
            if atype == AgentType.DRONE.value:
                agents.append(DroneAgent(agent_id=i, planner=planner, start=(sr, sc)))
            else:
                agents.append(GroundAgent(agent_id=i, planner=planner, start=(sr, sc)))

        print("=" * 60)
        print(f"  {self.name}")
        if self.allocator_desc:
            print(f"  {self.allocator_desc}")
        print(f"  {len(agents)} agent(s)   map {rows}x{cols}   [{path}]")
        print("=" * 60)

        if use_vis:
            from src.visualizer import SimulationVisualizer
            vis = SimulationVisualizer(ground_truth)
        else:
            vis = None

        for agent in agents:
            agent.observe(ground_truth, known_map)
        self._post_observation_updates(agents, auctioneer, known_map, ground_truth, verbose)

        for step in range(max_steps):
            if verbose:
                statuses = "  ".join(
                    f"A{a.id}@{a.pos}[{a.status.name[0]}]" for a in agents
                )
                print(f"\n--- Step {step:3d}  {statuses}  {auctioneer.stats()} ---", flush=True)

            if vis is not None:
                try:
                    vis.update(known_map, agents, step, auctioneer.stats(),
                               auctioneer=auctioneer)
                except Exception:
                    import traceback, sys
                    print("[VIS ERROR] Disabling visualizer:", file=sys.stderr, flush=True)
                    traceback.print_exc()
                    vis = None

            moved_any = self._do_microstep(
                agents, ground_truth, known_map, auctioneer, verbose
            )

            drone_agents = [a for a in agents if isinstance(a, DroneAgent)]
            if drone_agents:
                moved_any = (
                    self._do_microstep(
                        drone_agents, ground_truth, known_map, auctioneer, verbose
                    )
                    or moved_any
                )

            self._update_triage_progress(agents, verbose)
            auctioneer.update(agents, known_map, ground_truth=ground_truth, verbose=verbose)
            self._post_triage_update(agents, auctioneer, known_map, verbose)

            if auctioneer.all_complete and all(a.status == AgentStatus.IDLE for a in agents):
                if vis is not None:
                    try:
                        vis.update(known_map, agents, step, auctioneer.stats(),
                                   auctioneer=auctioneer)
                    except Exception:
                        import traceback, sys
                        print("[VIS ERROR] Disabling visualizer:", file=sys.stderr)
                        traceback.print_exc()
                        vis = None
                print(f"\n[DONE] All {auctioneer.stats()} — finished in {step + 1} steps.")
                break

            if not moved_any and verbose:
                print("  [STALL] No agents moved this step")

        else:
            print(f"\n[TIMEOUT] Simulation ended after {max_steps} steps.")

        if vis is not None:
            vis.finalize(auctioneer.stats())
        return known_map
