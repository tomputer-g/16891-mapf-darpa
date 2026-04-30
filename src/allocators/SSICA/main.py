"""
SSICA test harness for the DARPA exploration simulation.

Uses the Sequential Single-Item Concurrent Auction from
``SSICA.task_allocation`` on a scenario file, defaulting to
``generated/darpa1.txt``.
"""

import argparse

from src.agents import AgentStatus
from src.maps import KnownMap
from src.planner import CBS
from .task_allocation import SequentialSingleItemAuctioneer
from src.allocators.base_main import SimulationHarness


class SSICAHarness(SimulationHarness):
    name = "SSICA Exploration Simulation"
    allocator_desc = "allocator=SequentialSingleItemAuctioneer"

    def _make_auctioneer(self, planner: CBS) -> SequentialSingleItemAuctioneer:
        return SequentialSingleItemAuctioneer(cbs=planner)

    def _handle_replan_failure(self, agent, agents, auctioneer,
                                known_map: KnownMap, verbose) -> None:
        """SSICA escalates replan failures to handle_invalidated_assignment."""
        auctioneer.handle_invalidated_assignment(
            agent, agents, known_map, verbose=verbose, reason="replanning"
        )

    def _post_triage_update(self, agents, auctioneer, known_map: KnownMap,
                             verbose) -> None:
        """Handle agents still in REPLANNING after the end-of-step update."""
        for agent in agents:
            if agent.status == AgentStatus.REPLANNING:
                if not agent.replan(known_map):
                    auctioneer.handle_invalidated_assignment(
                        agent, agents, known_map, verbose=verbose, reason="replanning"
                    )


def run_simulation(
    path: str = "generated/darpa1.txt",
    max_steps: int = 200,
    verbose: bool = True,
    use_vis: bool = True,
):
    return SSICAHarness().run_simulation(path, max_steps, verbose, use_vis)


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
