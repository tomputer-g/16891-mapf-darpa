import os
import re
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
APPROACHES = [
    {"file": "results/naive_results.txt",             "name": "Greedy",            "color": "#888888"},
    {"file": "results/ssia_results.txt",              "name": "SSIA",             "color": "#DD8452"},
    {"file": "results/ssica_results.txt",             "name": "SSICA",            "color": "#55A868"},
    {"file": "results/ssia_collateral_results.txt",   "name": "SSIA+Collateral",  "color": "#C44E52"},
]

MAP_LABELS = [f"DARPA {i}" for i in range(1, 8)]   # darpa1 … darpa7
# ──────────────────────────────────────────────────────────────────────────────


def parse_results(filepath: str) -> dict[str, int]:
    """Return {map_name: makespan} from a results file."""
    data = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r".+/(darpa\d+)\.txt:\s*(\d+)", line)
            if m:
                data[m.group(1)] = int(m.group(2))
    return data


def main():
    map_keys = [f"darpa{i}" for i in range(1, 8)]
    n_maps = len(map_keys)
    n_approaches = len(APPROACHES)

    # Load all results
    results = []
    for approach in APPROACHES:
        parsed = parse_results(approach["file"])
        results.append([parsed.get(k, 0) for k in map_keys])

    # Bar layout
    x = np.arange(n_maps)
    total_width = 0.75
    bar_width = total_width / n_approaches
    offsets = np.linspace(-(total_width - bar_width) / 2,
                          (total_width - bar_width) / 2,
                          n_approaches)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (approach, values) in enumerate(zip(APPROACHES, results)):
        bars = ax.bar(
            x + offsets[i],
            values,
            width=bar_width,
            label=approach["name"],
            color=approach["color"],
            edgecolor="white",
            linewidth=0.6,
        )
        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                str(val),
                ha="center", va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(MAP_LABELS)
    ax.set_ylabel("Makespan (timesteps)")
    ax.set_title("Makespan Comparison Across Baselines")
    ax.legend(title="Approach")
    ax.set_ylim(0, max(v for row in results for v in row) * 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = "figures/makespan_chart.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
