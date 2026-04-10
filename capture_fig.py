"""Capture a mid-simulation screenshot for the report figure."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List

from agents import Agent, AgentStatus, EventType, GroundAgent, DroneAgent
from maps import KnownMap, load_new_scenario
from sim_types import AgentType
from planner import CBS
from SSIA_task_allocation import SequentialSingleItemAuctioneer
from SSIA_main import _update_triage_progress, _post_observation_updates, _do_microstep
from visualizer import SimulationVisualizer

CAPTURE_STEP = 10  # capture at this step (mid-exploration)

path = "generated/darpa1.txt"
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

# Initial observation
for agent in agents:
    agent.observe(ground_truth, known_map)
_post_observation_updates(agents, auctioneer, known_map, ground_truth, verbose=False)

# Run simulation to capture step
for step in range(CAPTURE_STEP):
    moved_any = _do_microstep(agents, ground_truth, known_map, auctioneer, verbose=False)
    drone_agents = [a for a in agents if isinstance(a, DroneAgent)]
    if drone_agents:
        moved_any = _do_microstep(drone_agents, ground_truth, known_map, auctioneer, verbose=False) or moved_any
    _update_triage_progress(agents, verbose=False)
    auctioneer.update(agents, known_map, ground_truth=ground_truth, verbose=False)

# Now create the visualization and save
vis = SimulationVisualizer(ground_truth)
vis.update(known_map, agents, CAPTURE_STEP, auctioneer.stats())

os.makedirs("figures", exist_ok=True)
vis.fig.savefig("figures/visualization.png", dpi=300, bbox_inches="tight",
                facecolor=vis.fig.get_facecolor())
print(f"Saved figures/visualization.png at step {CAPTURE_STEP}")
plt.close(vis.fig)
