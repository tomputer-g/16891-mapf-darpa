"""
Live matplotlib visualizer for the DARPA exploration simulation.

  SimulationVisualizer – renders ground truth, fog-of-war, agent, and path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from maps import GroundTruthMap, KnownMap, ObservationState

if TYPE_CHECKING:
    from agents import Agent


class SimulationVisualizer:
    """
    Live matplotlib display of the exploration simulation.

    Rendering layers (bottom → top)
    --------------------------------
    1. Ground truth  – always visible; shows real terrain under the fog.
    2. Knowledge     – opaque on observed cells (white=FREE, dark=OBSTACLE);
                       transparent on UNKNOWN so the GT shows through.
    3. Fog            – semi-transparent grey on UNKNOWN cells only.
    4. Path line      – planned route from agent to task target (cyan dashes).
    5. Task marker    – gold star on the current task target cell.
    6. Agent marker   – green circle on the agent's current position.

    The effect: unexplored cells reveal the underlying terrain through a grey
    mist; explored cells appear cleanly mapped.
    """

    # ── Palette ──────────────────────────────────────────────────────────
    _GT_FREE     = (0.85, 0.93, 0.85, 1.0)   # pale green  – GT free cell
    _GT_OBS      = (0.12, 0.12, 0.12, 1.0)   # near-black  – GT obstacle
    _KN_FREE     = (1.00, 1.00, 1.00, 1.0)   # white       – known free
    _KN_OBS      = (0.22, 0.22, 0.22, 1.0)   # charcoal    – known obstacle
    _FOG_COLOUR  = (0.55, 0.55, 0.55, 0.62)  # grey fog    – RGBA

    def __init__(self, ground_truth: GroundTruthMap) -> None:
        self._gt   = ground_truth
        self._rows = ground_truth.rows
        self._cols = ground_truth.cols

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.patch.set_facecolor("#1a1a2e")  # type: ignore[union-attr]
        self.ax.set_facecolor("#1a1a2e")
        self.fig.canvas.manager.set_window_title("DARPA Exploration Sim")  # type: ignore[union-attr]

        # Axes: x = col index, y = row index (row 0 at top)
        self.ax.set_xlim(-0.5, self._cols - 0.5)
        self.ax.set_ylim(self._rows - 0.5, -0.5)   # inverted so row 0 is top
        self.ax.set_aspect("equal")

        # Subtle grid lines at cell boundaries
        self.ax.set_xticks(np.arange(-0.5, self._cols, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self._rows, 1), minor=True)
        self.ax.grid(which="minor", color="#2a2a4a", linewidth=0.6, zorder=0)
        self.ax.tick_params(which="both", bottom=False, left=False,
                            labelbottom=False, labelleft=False)

        # imshow extent maps pixel centres to cell centres
        ext = [-0.5, self._cols - 0.5, self._rows - 0.5, -0.5]

        # Layer 1: ground truth (static — built once)
        self._gt_img = self._make_gt_image()
        self.ax.imshow(self._gt_img, origin="upper", extent=ext,
                       interpolation="nearest", zorder=1)

        # Layer 2: knowledge overlay (dynamic — opaque for known cells)
        self._know_data = np.zeros((self._rows, self._cols, 4))
        self._know_im = self.ax.imshow(self._know_data, origin="upper",
                                       extent=ext, interpolation="nearest",
                                       zorder=2)

        # Layer 3: fog (dynamic — semi-transparent grey on UNKNOWN)
        self._fog_data = np.zeros((self._rows, self._cols, 4))
        self._fog_im = self.ax.imshow(self._fog_data, origin="upper",
                                      extent=ext, interpolation="nearest",
                                      zorder=3)

        # Path line
        self._path_line, = self.ax.plot(
            [], [], "--", color="#00cfff", linewidth=1.8,
            alpha=0.85, zorder=4, label="Planned path")

        # Task target star
        self._task_dot, = self.ax.plot(
            [], [], "*", color="#ffd700", markersize=18,
            markeredgecolor="#7a5200", markeredgewidth=0.8,
            zorder=5, label="Task target")

        # Agent circle
        self._agent_dot, = self.ax.plot(
            [], [], "o", color="#00ff88", markersize=13,
            markeredgecolor="black", markeredgewidth=1.0,
            zorder=6, label="Agent")

        self._title = self.ax.set_title(
            "Initialising…", color="white", fontsize=10, pad=8)

        self._build_legend()
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.01)

    # ── Image builders ───────────────────────────────────────────────────

    def _make_gt_image(self) -> np.ndarray:
        img = np.ones((self._rows, self._cols, 4))
        for r in range(self._rows):
            for c in range(self._cols):
                img[r, c] = (self._GT_OBS if self._gt.is_obstacle((r, c))
                             else self._GT_FREE)
        return img

    def _update_knowledge_layer(self, known_map: KnownMap) -> None:
        for r in range(self._rows):
            for c in range(self._cols):
                s = known_map.state[r][c]
                if s == ObservationState.FREE:
                    self._know_data[r, c] = self._KN_FREE
                elif s == ObservationState.OBSTACLE:
                    self._know_data[r, c] = self._KN_OBS
                else:
                    self._know_data[r, c] = (0, 0, 0, 0)   # transparent → GT visible
        self._know_im.set_data(self._know_data)

    def _update_fog_layer(self, known_map: KnownMap) -> None:
        for r in range(self._rows):
            for c in range(self._cols):
                self._fog_data[r, c] = (
                    self._FOG_COLOUR
                    if known_map.state[r][c] == ObservationState.UNKNOWN
                    else (0, 0, 0, 0)
                )
        self._fog_im.set_data(self._fog_data)

    # ── Legend ───────────────────────────────────────────────────────────

    def _build_legend(self) -> None:
        elements = [
            mpatches.Patch(facecolor=self._KN_FREE[:3],
                           edgecolor="#aaa", label="Free (mapped)"),
            mpatches.Patch(facecolor=self._KN_OBS[:3],
                           edgecolor="#aaa", label="Obstacle (mapped)"),
            mpatches.Patch(facecolor=self._GT_FREE[:3], alpha=0.55,
                           edgecolor="#888", label="Free (unknown)"),
            mpatches.Patch(facecolor=self._GT_OBS[:3],  alpha=0.55,
                           edgecolor="#888", label="Obstacle (unknown)"),
            Line2D([0], [0], color="#00cfff", linestyle="--",
                   linewidth=1.8, label="Planned path"),
            Line2D([0], [0], marker="*", color="#ffd700",
                   markeredgecolor="#7a5200", markersize=13,
                   linestyle="None", label="Task target"),
            Line2D([0], [0], marker="o", color="#00ff88",
                   markeredgecolor="black", markersize=10,
                   linestyle="None", label="Agent"),
        ]
        legend = self.ax.legend(
            handles=elements, loc="upper right", fontsize=7.5,
            facecolor="#222244", edgecolor="#555577", labelcolor="white")
        legend.set_zorder(10)

    # ── Public interface ─────────────────────────────────────────────────

    def update(
        self,
        known_map:  KnownMap,
        agents:     List[Agent],
        step:       int,
        task_stats: str,
    ) -> None:
        """Redraw the display for one simulation step."""
        self._update_knowledge_layer(known_map)
        self._update_fog_layer(known_map)

        # Path (first agent only; extend loop for multi-agent)
        agent = agents[0]
        if len(agent.path) > 1:
            self._path_line.set_data(
                [p[1] for p in agent.path],   # col → x
                [p[0] for p in agent.path])   # row → y
        else:
            self._path_line.set_data([], [])

        # Task target
        if agent.current_task and not agent.current_task.completed:
            tr, tc = agent.current_task.target_loc
            self._task_dot.set_data([tc], [tr])
        else:
            self._task_dot.set_data([], [])

        # Agent position
        self._agent_dot.set_data([agent.pos[1]], [agent.pos[0]])

        # Caption
        self._title.set_text(f"Step {step}   |   {task_stats}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def finalize(self, task_stats: str) -> None:
        """Mark simulation complete and block until the window is closed."""
        self._title.set_text(f"COMPLETE   |   {task_stats}")
        self.fig.canvas.draw()
        plt.ioff()
        plt.show()
