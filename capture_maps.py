"""Generate a composite figure showing all 7 DARPA scenario maps."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

from maps import load_new_scenario

# Palette (matches show_map.py / visualizer.py)
_GT_FREE         = (0.85, 0.93, 0.85)
_GT_OBS          = (0.12, 0.12, 0.12)
_GT_BUILDING     = (0.60, 0.75, 0.90)
_GT_BUILDING_OCC = (0.95, 0.60, 0.10)
_GT_OBJECTIVE    = (0.95, 0.90, 0.10)
_DRONE_COLOR     = "#4363d8"
_GROUND_COLOR    = "#e6194b"

MAPS = [
    ("generated/darpa1.txt", "(a) darpa1"),
    ("generated/darpa2.txt", "(b) darpa2"),
    ("generated/darpa3.txt", "(c) darpa3"),
    ("generated/darpa4.txt", "(d) darpa4"),
    ("generated/darpa5.txt", "(e) darpa5"),
    ("generated/darpa6.txt", "(f) darpa6"),
    ("generated/darpa7.txt", "(g) darpa7"),
]


def _cell_color(r, c, gt):
    if gt.is_obstacle((r, c)):
        return _GT_OBS
    loc = (r, c)
    if loc in gt.buildings:
        return _GT_BUILDING_OCC if gt.buildings[loc] else _GT_BUILDING
    if loc in gt.objectives:
        return _GT_OBJECTIVE
    return _GT_FREE


def render_map(ax, path, title):
    gt = load_new_scenario(path)
    img = np.array([
        [_cell_color(r, c, gt) for c in range(gt.cols)]
        for r in range(gt.rows)
    ])

    ext = [-0.5, gt.cols - 0.5, gt.rows - 0.5, -0.5]
    ax.imshow(img, origin="upper", extent=ext, interpolation="nearest", zorder=1)

    # Grid
    ax.set_xticks(np.arange(-0.5, gt.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, gt.rows, 1), minor=True)
    ax.grid(which="minor", color="#2a2a4a", linewidth=0.5, zorder=2)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.set_xlim(-0.5, gt.cols - 0.5)
    ax.set_ylim(gt.rows - 0.5, -0.5)
    ax.set_aspect("equal")

    # Agent starts
    for r, c, atype in gt.agent_starts:
        color = _DRONE_COLOR if atype == "D" else _GROUND_COLOR
        marker = "^" if atype == "D" else "s"
        ax.plot(c, r, marker, color=color, markersize=14,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)

    ax.set_title(title, color="white", fontsize=32, fontweight="bold", pad=8)


# Layout: top row = 4 maps (darpa1-4, smaller), bottom row = 3 maps (darpa5-7)
# Use gridspec with height_ratios for tight packing
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(36, 24))
fig.patch.set_facecolor("#1a1a2e")

gs = GridSpec(2, 12, figure=fig, height_ratios=[1, 1.1],
              hspace=0.08, wspace=0.3,
              left=0.02, right=0.98, top=0.95, bottom=0.06)

# Top row: 4 maps spanning 3 columns each
for i in range(4):
    ax = fig.add_subplot(gs[0, i*3:(i+1)*3])
    ax.set_facecolor("#1a1a2e")
    render_map(ax, MAPS[i][0], MAPS[i][1])

# Bottom row: 3 maps spanning 4 columns each
for i in range(3):
    ax = fig.add_subplot(gs[1, i*4:(i+1)*4])
    ax.set_facecolor("#1a1a2e")
    render_map(ax, MAPS[4 + i][0], MAPS[4 + i][1])

# Shared legend at bottom
legend_entries = [
    mpatches.Patch(facecolor=_GT_FREE, edgecolor="#888", label="Free"),
    mpatches.Patch(facecolor=_GT_OBS, edgecolor="#888", label="Obstacle"),
    mpatches.Patch(facecolor=_GT_BUILDING, edgecolor="#888", label="Building (empty)"),
    mpatches.Patch(facecolor=_GT_BUILDING_OCC, edgecolor="#888", label="Building (occupied)"),
    mpatches.Patch(facecolor=_GT_OBJECTIVE, edgecolor="#888", label="Objective"),
    mpatches.Patch(color=_DRONE_COLOR, label="Drone start"),
    mpatches.Patch(color=_GROUND_COLOR, label="Ground start"),
]
fig.legend(handles=legend_entries, loc="lower center", ncol=7, fontsize=32,
           facecolor="#222244", edgecolor="#555577", labelcolor="white",
           bbox_to_anchor=(0.5, -0.01))

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/all_maps.png", dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved figures/all_maps.png")
plt.close(fig)
