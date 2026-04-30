"""
SSIA-collateral test harness for the DARPA exploration simulation.

Uses the collateral-augmented Sequential Single-Item Auction from
``SSIA_collateral.task_allocation`` on a scenario file, defaulting to
``generated/darpa1.txt``.
"""

import argparse

from src.planner import CBS
from .task_allocation import SequentialSingleItemAuctioneer
from src.allocators.base_main import SimulationHarness


class SSIACollateralHarness(SimulationHarness):
    name = "SSIA-Cost Exploration Simulation"
    allocator_desc = "allocator=SequentialSingleItemAuctioneer (with collateral)"

    def _make_auctioneer(self, planner: CBS) -> SequentialSingleItemAuctioneer:
        return SequentialSingleItemAuctioneer(cbs=planner)


def run_simulation(
    path: str = "generated/darpa1.txt",
    max_steps: int = 200,
    verbose: bool = True,
    use_vis: bool = True,
):
    return SSIACollateralHarness().run_simulation(path, max_steps, verbose, use_vis)


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
