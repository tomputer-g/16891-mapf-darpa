"""Capture an animated GIF of the SSIA simulation on a given scenario."""
import matplotlib
matplotlib.use("Agg")

import argparse
import io
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from agents import Agent, AgentStatus, DroneAgent, GroundAgent
from maps import KnownMap, load_new_scenario
from sim_types import AgentType
from planner import CBS
from SSIA_task_allocation import SequentialSingleItemAuctioneer
from SSIA_main import _update_triage_progress, _post_observation_updates, _do_microstep
from visualizer import SimulationVisualizer


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    return img.copy()  # detach from buffer


def capture_gif(
    path: str = "generated/darpa7.txt",
    max_steps: int = 200,
    out: str = "figures/ssia_darpa7.gif",
    frame_duration: int = 200,
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

    # Initial observation
    for agent in agents:
        agent.observe(ground_truth, known_map)
    _post_observation_updates(agents, auctioneer, known_map, ground_truth, verbose=False)

    # Create visualizer (non-interactive)
    plt.ioff()
    vis = SimulationVisualizer.__new__(SimulationVisualizer)
    vis._gt = ground_truth
    vis._rows = rows
    vis._cols = cols
    vis.fig, vis.ax = plt.subplots(figsize=(10, 10))
    vis.fig.patch.set_facecolor("#1a1a2e")
    vis.ax.set_facecolor("#1a1a2e")
    vis.ax.set_xlim(-0.5, cols - 0.5)
    vis.ax.set_ylim(rows - 0.5, -0.5)
    vis.ax.set_aspect("equal")
    vis.ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    vis.ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    vis.ax.grid(which="minor", color="#2a2a4a", linewidth=0.8, zorder=0)
    vis.ax.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)
    ext = [-0.5, cols - 0.5, rows - 0.5, -0.5]
    vis._gt_img = vis._make_gt_image()
    vis.ax.imshow(vis._gt_img, origin="upper", extent=ext,
                  interpolation="nearest", zorder=1)
    vis._know_data = np.zeros((rows, cols, 4))
    vis._know_im = vis.ax.imshow(vis._know_data, origin="upper", extent=ext,
                                  interpolation="nearest", zorder=2)
    vis._fog_data = np.zeros((rows, cols, 4))
    vis._fog_im = vis.ax.imshow(vis._fog_data, origin="upper", extent=ext,
                                 interpolation="nearest", zorder=3)
    vis._agent_artists = {}
    vis._title = vis.ax.set_title("", color="white", fontsize=16,
                                   fontweight="bold", pad=8)
    vis._legend_handle = None
    vis._build_legend([])
    plt.tight_layout()

    frames: List[Image.Image] = []

    def snap(step):
        vis._update_knowledge_layer(known_map)
        vis._update_fog_layer(known_map)
        legend_dirty = False
        for agent in agents:
            if agent.id not in vis._agent_artists:
                vis._create_agent_artists(agent)
                legend_dirty = True
            arts = vis._agent_artists[agent.id]
            if len(agent.path) > 1:
                arts["path"].set_data([p[1] for p in agent.path],
                                      [p[0] for p in agent.path])
            else:
                arts["path"].set_data([], [])
            if agent.current_task and not agent.current_task.completed:
                tr, tc = agent.current_task.target_loc
                arts["target"].set_data([tc], [tr])
            else:
                arts["target"].set_data([], [])
            arts["dot"].set_data([agent.pos[1]], [agent.pos[0]])
            arts["label"].set_visible(False)
        if legend_dirty:
            vis._build_legend(agents)
            # Shrink legend font for GIF
            if vis._legend_handle:
                for t in vis._legend_handle.get_texts():
                    t.set_fontsize(9)
        vis._title.set_text(f"Step {step}   |   {auctioneer.stats()}")
        vis.fig.canvas.draw()
        frames.append(_fig_to_pil(vis.fig))

    # Capture initial state
    snap(0)

    for step in range(1, max_steps + 1):
        moved_any = _do_microstep(agents, ground_truth, known_map, auctioneer, verbose=False)
        drone_agents = [a for a in agents if isinstance(a, DroneAgent)]
        if drone_agents:
            moved_any = _do_microstep(drone_agents, ground_truth, known_map, auctioneer, verbose=False) or moved_any
        _update_triage_progress(agents, verbose=False)
        auctioneer.update(agents, known_map, ground_truth=ground_truth, verbose=False)

        snap(step)
        print(f"  frame {step}", end="\r")

        if auctioneer.all_complete and all(a.status == AgentStatus.IDLE for a in agents):
            print(f"\n  Simulation complete at step {step}")
            break

    # Hold on final frame
    for _ in range(5):
        frames.append(frames[-1].copy())

    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=frame_duration, loop=0)
    print(f"Saved {out}  ({len(frames)} frames)")
    plt.close(vis.fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture SSIA simulation GIF")
    parser.add_argument("path", nargs="?", default="generated/darpa7.txt")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", default="figures/ssia_darpa7.gif")
    parser.add_argument("--duration", type=int, default=200,
                        help="ms per frame (default: 200)")
    args = parser.parse_args()
    capture_gif(path=args.path, max_steps=args.steps, out=args.out,
                frame_duration=args.duration)
