"""Round-robin head-to-head win rates between bots (2 players), as a table and a heatmap.

Usage: python scripts/compare.py random greedy "PPO=ppo:models/sp_2p4c/best.zip" --cards 4 --games 1000
(optional LABEL=spec names a bot in the output)
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings; warnings.simplefilter("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from eval_bots import evaluate
from game.bots import make_bot

ap = argparse.ArgumentParser()
ap.add_argument("bots", nargs="+")
ap.add_argument("--cards", type=int, default=4)
ap.add_argument("--games", type=int, default=1000)
ap.add_argument("--out", default="plots/compare.png")
a = ap.parse_args()

names = [b.split("=")[0] if "=" in b else b for b in a.bots]
bots = [make_bot(b.split("=")[-1]) for b in a.bots]
n = len(bots)
M = np.full((n, n), np.nan)
for i in range(n):
    for j in range(i + 1, n):
        wins, _ = evaluate([bots[i], bots[j]], a.games, cards=a.cards, seed0=20_000)
        M[i, j], M[j, i] = wins[0] / a.games, wins[1] / a.games

print(f"Row's win rate vs column (2 players, {a.cards} cards, {a.games} games each, ±{1.96 * 0.5 / np.sqrt(a.games):.1%})")
w = max(map(len, names)) + 2
print(" " * w + "".join(f"{s:>{w}}" for s in names))
for i in range(n):
    print(f"{names[i]:<{w}}" + "".join(f"{'—' if np.isnan(v) else f'{v:.1%}':>{w}}" for v in M[i]))

fig, ax = plt.subplots(figsize=(1.6 * n + 2, 1.3 * n + 1.5))
ax.imshow(np.nan_to_num(M, nan=0.5), cmap="RdYlGn", vmin=0, vmax=1)
for i in range(n):
    for j in range(n):
        ax.text(j, i, "—" if i == j else f"{M[i, j]:.0%}", ha="center", va="center", fontsize=12)
ax.set_xticks(range(n), names, rotation=20, ha="right")
ax.set_yticks(range(n), names)
ax.set_title(f"Head-to-head win rate (row vs column)\n2 players, {a.cards} cards, {a.games} games each")
fig.tight_layout()
Path(a.out).parent.mkdir(exist_ok=True)
fig.savefig(a.out, dpi=120)
print("saved", a.out)
