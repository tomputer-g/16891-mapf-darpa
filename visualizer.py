"""
Live matplotlib visualizer for the DARPA exploration simulation.

  SimulationVisualizer – renders ground truth, fog-of-war, agent, and path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from maps import GroundTruthMap, KnownMap
from sim_types import AgentType, ObservationState

if TYPE_CHECKING:
    from agents import Agent

# Cycle through these for agent colours; fall back to random HSV beyond len.
_AGENT_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


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
    _GT_FREE          = (0.85, 0.93, 0.85, 1.0)   # pale green   – GT free cell
    _GT_OBS           = (0.12, 0.12, 0.12, 1.0)   # near-black   – GT obstacle
    _GT_BUILDING      = (0.60, 0.75, 0.90, 1.0)   # steel blue   – unoccupied building (GT)
    _GT_BUILDING_OCC  = (0.95, 0.60, 0.10, 1.0)   # amber        – occupied building (GT)
    _GT_OBJECTIVE     = (0.95, 0.90, 0.10, 1.0)   # gold         – free-standing objective (GT)
    _KN_FREE          = (1.00, 1.00, 1.00, 1.0)   # white        – known free
    _KN_OBS           = (0.22, 0.22, 0.22, 1.0)   # charcoal     – known obstacle
    _KN_BUILDING      = (0.75, 0.88, 1.00, 1.0)   # light blue   – unoccupied building (known)
    _KN_BUILDING_OCC  = (1.00, 0.78, 0.35, 1.0)   # light amber  – occupied building (known)
    _KN_OBJECTIVE     = (1.00, 0.97, 0.40, 1.0)   # light gold   – free-standing objective (known)
    _FOG_COLOUR       = (0.55, 0.55, 0.55, 0.62)  # grey fog     – RGBA

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

        # Per-agent artists created lazily in _create_agent_artists()
        self._agent_artists: Dict[int, dict] = {}

        self._title = self.ax.set_title(
            "Initialising…", color="white", fontsize=10, pad=8)

        self._legend_handle = self._build_legend([])
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.01)

    # ── Image builders ───────────────────────────────────────────────────

    def _gt_colour(self, r: int, c: int) -> tuple:
        """Return the ground-truth RGBA colour for cell (r, c)."""
        if self._gt.is_obstacle((r, c)):
            return self._GT_OBS
        loc = (r, c)
        if loc in self._gt.buildings:
            return self._GT_BUILDING_OCC if self._gt.buildings[loc] else self._GT_BUILDING
        if loc in self._gt.objectives:
            return self._GT_OBJECTIVE
        return self._GT_FREE

    def _make_gt_image(self) -> np.ndarray:
        img = np.ones((self._rows, self._cols, 4))
        for r in range(self._rows):
            for c in range(self._cols):
                img[r, c] = self._gt_colour(r, c)
        return img

    def _known_colour(self, r: int, c: int, state: ObservationState) -> tuple:
        """Return the knowledge-layer RGBA colour for an observed cell."""
        if state == ObservationState.OBSTACLE:
            return self._KN_OBS
        if state == ObservationState.FREE:
            loc = (r, c)
            if loc in self._gt.buildings:
                return self._KN_BUILDING_OCC if self._gt.buildings[loc] else self._KN_BUILDING
            if loc in self._gt.objectives:
                return self._KN_OBJECTIVE
            return self._KN_FREE
        return (0, 0, 0, 0)   # UNKNOWN → transparent so GT layer shows through

    def _update_knowledge_layer(self, known_map: KnownMap) -> None:
        for r in range(self._rows):
            for c in range(self._cols):
                self._know_data[r, c] = self._known_colour(r, c, known_map.state[r][c])
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

    def _build_legend(self, agents: List[Agent]) -> None:
        elements: list = [
            mpatches.Patch(facecolor=self._KN_FREE[:3],
                           edgecolor="#aaa", label="Free (mapped)"),
            mpatches.Patch(facecolor=self._KN_OBS[:3],
                           edgecolor="#aaa", label="Obstacle (mapped)"),
            mpatches.Patch(facecolor=self._GT_FREE[:3], alpha=0.55,
                           edgecolor="#888", label="Free (unknown)"),
            mpatches.Patch(facecolor=self._GT_OBS[:3],  alpha=0.55,
                           edgecolor="#888", label="Obstacle (unknown)"),
        ]
        if self._gt.buildings:
            elements += [
                mpatches.Patch(facecolor=self._KN_BUILDING[:3],
                               edgecolor="#aaa", label="Building (empty)"),
                mpatches.Patch(facecolor=self._KN_BUILDING_OCC[:3],
                               edgecolor="#aaa", label="Building (occupied)"),
            ]
        if self._gt.objectives:
            elements.append(mpatches.Patch(facecolor=self._KN_OBJECTIVE[:3],
                                           edgecolor="#aaa", label="Objective"))
        if agents:
            n_drones  = sum(1 for a in agents if a.agent_type == AgentType.DRONE)
            n_ground  = sum(1 for a in agents if a.agent_type == AgentType.GROUND)
            if n_drones:
                elements.append(Line2D(
                    [0], [0], marker="^", color=_AGENT_PALETTE[0],
                    markeredgecolor="black", markersize=9,
                    linestyle="--", linewidth=1.5,
                    label=f"Drone ({n_drones})"))
            if n_ground:
                elements.append(Line2D(
                    [0], [0], marker="s", color=_AGENT_PALETTE[2 % len(_AGENT_PALETTE)],
                    markeredgecolor="black", markersize=9,
                    linestyle="--", linewidth=1.5,
                    label=f"Ground ({n_ground})"))
        if hasattr(self, "_legend_handle") and self._legend_handle:
            self._legend_handle.remove()
        legend = self.ax.legend(
            handles=elements, loc="upper right", fontsize=7.5,
            facecolor="#222244", edgecolor="#555577", labelcolor="white")
        legend.set_zorder(10)
        self._legend_handle = legend

    def _create_agent_artists(self, agent: Agent) -> None:
        """Lazily create path/target/dot/label artists for a new agent."""
        idx   = len(self._agent_artists)
        color = _AGENT_PALETTE[idx % len(_AGENT_PALETTE)]

        # Drones: triangle-up, Ground: square
        is_drone = agent.agent_type == AgentType.DRONE
        marker = "^" if is_drone else "s"
        msize  = 13  if is_drone else 11

        path_line, = self.ax.plot(
            [], [], "--", color=color, linewidth=1.6, alpha=0.85, zorder=4)
        task_dot, = self.ax.plot(
            [], [], "*", color=color, markersize=16,
            markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        agent_dot, = self.ax.plot(
            [], [], marker, color=color, markersize=msize,
            markeredgecolor="black", markeredgewidth=1.0, zorder=6)
        label = self.ax.text(
            0, 0, str(agent.id),
            color="white", fontsize=6, fontweight="bold",
            ha="center", va="center", zorder=7)

        self._agent_artists[agent.id] = {
            "color":     color,
            "path":      path_line,
            "target":    task_dot,
            "dot":       agent_dot,
            "label":     label,
        }

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

        legend_dirty = False
        for agent in agents:
            if agent.id not in self._agent_artists:
                self._create_agent_artists(agent)
                legend_dirty = True

            arts = self._agent_artists[agent.id]

            # Path
            if len(agent.path) > 1:
                arts["path"].set_data(
                    [p[1] for p in agent.path],   # col → x
                    [p[0] for p in agent.path])   # row → y
            else:
                arts["path"].set_data([], [])

            # Task target
            if agent.current_task and not agent.current_task.completed:
                tr, tc = agent.current_task.target_loc
                arts["target"].set_data([tc], [tr])
            else:
                arts["target"].set_data([], [])

            # Agent dot + ID label
            arts["dot"].set_data([agent.pos[1]], [agent.pos[0]])
            arts["label"].set_position((agent.pos[1], agent.pos[0]))

        if legend_dirty:
            self._build_legend(agents)

        self._title.set_text(f"Step {step}   |   {task_stats}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.3)

    def finalize(self, task_stats: str) -> None:
        """Mark simulation complete and block until the window is closed."""
        self._title.set_text(f"COMPLETE   |   {task_stats}")
        self.fig.canvas.draw()
        plt.ioff()
        plt.show()
