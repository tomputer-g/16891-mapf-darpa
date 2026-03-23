"""
show_map.py — Static visualizer for a scenario file.

Renders the ground-truth map (obstacles, buildings, objectives, agent starts)
without running a simulation.

Usage
-----
    python3 show_map.py <scenario_file>
"""

import argparse
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from maps import load_new_scenario


# ── Palette (matches SimulationVisualizer) ───────────────────────────────────
_GT_FREE         = (0.85, 0.93, 0.85)
_GT_OBS          = (0.12, 0.12, 0.12)
_GT_BUILDING     = (0.60, 0.75, 0.90)
_GT_BUILDING_OCC = (0.95, 0.60, 0.10)
_GT_OBJECTIVE    = (0.95, 0.90, 0.10)
_DRONE_COLOR     = "#4363d8"
_GROUND_COLOR    = "#e6194b"


def _cell_color(r: int, c: int, gt) -> tuple:
    if gt.is_obstacle((r, c)):
        return _GT_OBS
    loc = (r, c)
    if loc in gt.buildings:
        return _GT_BUILDING_OCC if gt.buildings[loc] else _GT_BUILDING
    if loc in gt.objectives:
        return _GT_OBJECTIVE
    return _GT_FREE


def show_map(path: str) -> None:
    gt = load_new_scenario(path)

    img = np.array([
        [_cell_color(r, c, gt) for c in range(gt.cols)]
        for r in range(gt.rows)
    ])

    fig, ax = plt.subplots(figsize=(max(6, gt.cols * 0.55), max(6, gt.rows * 0.55)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    fig.canvas.manager.set_window_title(f"Map: {path}")

    ext = [-0.5, gt.cols - 0.5, gt.rows - 0.5, -0.5]
    ax.imshow(img, origin="upper", extent=ext, interpolation="nearest", zorder=1)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, gt.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, gt.rows, 1), minor=True)
    ax.grid(which="minor", color="#2a2a4a", linewidth=0.6, zorder=2)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.set_xlim(-0.5, gt.cols - 0.5)
    ax.set_ylim(gt.rows - 0.5, -0.5)

    # Agent start markers
    for r, c, atype in gt.agent_starts:
        color  = _DRONE_COLOR if atype == "D" else _GROUND_COLOR
        marker = "^" if atype == "D" else "s"
        ax.plot(c, r, marker, color=color, markersize=11,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.text(c, r, atype, color="white", fontsize=6,
                fontweight="bold", ha="center", va="center", zorder=6)

    # Legend
    legend_entries = [
        mpatches.Patch(facecolor=_GT_FREE,         edgecolor="#888", label="Free"),
        mpatches.Patch(facecolor=_GT_OBS,          edgecolor="#888", label="Obstacle"),
    ]
    if gt.buildings:
        legend_entries += [
            mpatches.Patch(facecolor=_GT_BUILDING,     edgecolor="#888", label="Building (empty)"),
            mpatches.Patch(facecolor=_GT_BUILDING_OCC, edgecolor="#888", label="Building (occupied)"),
        ]
    if gt.objectives:
        legend_entries.append(
            mpatches.Patch(facecolor=_GT_OBJECTIVE, edgecolor="#888", label="Objective"))
    if any(t == "D" for _, _, t in gt.agent_starts):
        legend_entries.append(mpatches.Patch(color=_DRONE_COLOR, label="Drone start (▲)"))
    if any(t == "G" for _, _, t in gt.agent_starts):
        legend_entries.append(mpatches.Patch(color=_GROUND_COLOR, label="Ground start (■)"))

    legend = ax.legend(handles=legend_entries, loc="upper right", fontsize=8,
                       facecolor="#222244", edgecolor="#555577", labelcolor="white")
    legend.set_zorder(10)

    ax.set_title(path, color="white", fontsize=9, pad=6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a scenario file (static).")
    parser.add_argument("path", help="path to scenario file (new format)")
    args = parser.parse_args()
    show_map(args.path)
