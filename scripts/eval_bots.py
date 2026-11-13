"""Pit bots against each other; seats are shuffled every game.

Usage: python scripts/eval_bots.py greedy random --games 500 --cards 4
       python scripts/eval_bots.py ppo:models/best_model.zip greedy greedy greedy
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.simplefilter("ignore")
import numpy as np
from game.engine import TontoGame
from game.bots import make_bot


def play(bots, decks, rounds, cards, seed):
    g = TontoGame(len(bots), decks, rounds, cards, seed=seed)
    while not g.game_state.complete:
        while not g.round_state.complete:
            pid = g.game_state.current_player
            g.step(bots[pid].act(g, pid))
        g.next_round()
    return g


def evaluate(specs, games=500, decks=1, rounds=1, cards=4, seed0=0):
    """Returns per-spec-index win counts and score lists."""
    P = len(specs)
    bots = [make_bot(s, seed=seed0 + k) if isinstance(s, str) else s for k, s in enumerate(specs)]
    wins, scores = np.zeros(P), [[] for _ in range(P)]
    rng = np.random.default_rng(seed0)
    for s in range(games):
        order = rng.permutation(P)  # seat -> spec index; shuffling avoids turn-order adjacency bias
        g = play([bots[k] for k in order], decks, rounds, cards, seed0 + s)
        wins[order[g.winner]] += 1
        for seat, k in enumerate(order):
            scores[k].append(g.player_state[seat].total_score)
    return wins, scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bots", nargs="+")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--decks", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--cards", type=int, default=4)
    a = ap.parse_args()
    wins, scores = evaluate(a.bots, a.games, a.decks, a.rounds, a.cards)
    print(f"{len(a.bots)} players, {a.cards} cards, {a.decks} deck(s), {a.rounds} round(s), {a.games} games "
          f"(even skill = {1 / len(a.bots):.0%} each)")
    for k, name in enumerate(a.bots):
        p = wins[k] / a.games
        ci = 1.96 * np.sqrt(p * (1 - p) / a.games)
        print(f"  {k}: {name:28s} win {p:6.1%} ± {ci:4.1%}   mean score {np.mean(scores[k]):6.2f}")
