"""Pit bots against each other. Usage: python scripts/eval_bots.py greedy random --games 500 --cards 4"""
import sys, argparse
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.simplefilter("ignore")
import numpy as np
from game.engine import TontoGame
from game.bots import BOTS

def play(bots, decks, rounds, cards, seed):
    g = TontoGame(len(bots), decks, rounds, cards, seed=seed)
    while not g.game_state.complete:
        while not g.round_state.complete:
            pid = g.game_state.current_player
            g.step(bots[pid].act(g, pid))
        g.next_round()
    return g

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bots", nargs="+")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--decks", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--cards", type=int, default=4)
    a = ap.parse_args()
    wins, totals = Counter(), {i: [] for i in range(len(a.bots))}
    for s in range(a.games):
        # rotate seats so no bot always goes first
        order = [(s + i) % len(a.bots) for i in range(len(a.bots))]
        g = play([BOTS[a.bots[k]](**({"seed": s} if a.bots[k] == "random" else {})) for k in order],
                 a.decks, a.rounds, a.cards, s)
        wins[order[g.winner]] += 1
        for seat, k in enumerate(order):
            totals[k].append(g.player_state[seat].total_score)
    for k, name in enumerate(a.bots):
        print(f"seat {k} {name:7s} win rate {wins[k]/a.games:6.1%}  mean score {np.mean(totals[k]):6.2f}")
