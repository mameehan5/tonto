"""Overlay win rate vs greedy for several training runs. Usage: python scripts/plot_runs.py LABEL=models/run ..."""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.2))
for arg in sys.argv[1:-1]:
    label, run = arg.split("=")
    rows = list(csv.DictReader(open(Path(run) / "eval.csv")))
    rows = [r for i, r in enumerate(rows) if i == 0 or int(r["steps"]) > int(rows[i - 1]["steps"]) + 1000]
    ax.plot([int(r["steps"]) / 1e6 for r in rows], [float(r["win_greedy"]) for r in rows], "o-", ms=4, label=label)
ax.axhline(0.5, ls="--", color="#c0392b", lw=1, label="even with greedy")
ax.set(xlabel="training steps (millions)", ylabel="win rate vs greedy (2 players)", ylim=(0, 0.65),
       title="PPO training runs vs the greedy bot")
ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
fig.savefig(sys.argv[-1], dpi=120); print("saved", sys.argv[-1])
