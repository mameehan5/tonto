"""Generate expert-iteration training data: SearchBot plays every seat; record each searched decision.

Usage: python training/expert_data.py --players 2 --cards 8 --games 2000 --samples 16 --out data/exit1
Writes shards data/exit1/shard_<seed>.npz with obs, masks, actions, search values, and final outcomes.
Restartable: existing shards are skipped.
"""
import argparse
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np

from env.selfplay import encode_obs
from game.actions import Action
from game.engine import TontoGame
from game.search import SearchBot, outcome

NA = int(Action.NUM_POSSIBLE_ACTIONS)


def make_shard(args):
    seed0, games, P, N, samples, out, rollout = args
    path = Path(out) / f"shard_{seed0}.npz"
    if path.exists():
        return 0
    roll = None
    if rollout:
        from training.policy_bot import PolicyBot
        roll = PolicyBot(rollout)
    bot = SearchBot(samples, rollout=roll, seed=seed0)
    X, M, A, Q, V = [], [], [], [], []
    for s in range(seed0, seed0 + games):
        g = TontoGame(P, 1, 1, N, seed=s)
        recs = []
        while not g.round_state.complete:
            pid = g.game_state.current_player
            a = bot.act(g, pid)
            if bot.last_values is not None:
                q = np.full(NA, np.nan, np.float32)
                for k, v in bot.last_values.items():
                    q[k] = v
                recs.append((pid, len(X)))
                X.append(encode_obs(g, pid))
                M.append(g.legal_mask().astype(bool))
                A.append(a)
                Q.append(q)
                V.append(0.0)
            g.step(a)
        for pid, i in recs:
            V[i] = outcome(g, pid)
    np.savez_compressed(path, obs=np.array(X), masks=np.array(M), actions=np.array(A),
                        q=np.array(Q), value=np.array(V, np.float32))
    return len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--cards", type=int, default=8)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--rollout", help="model .zip for rollouts instead of GreedyBot (later iterations)")
    ap.add_argument("--shard", type=int, default=50, help="games per shard")
    ap.add_argument("--seed", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="data/exit1")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(a.seed + i, a.shard, a.players, a.cards, a.samples, a.out, a.rollout)
            for i in range(0, a.games, a.shard)]
    total = 0
    with Pool(a.workers) as pool:
        for k, n in enumerate(pool.imap_unordered(make_shard, jobs)):
            total += n
            print(f"shard {k + 1}/{len(jobs)} done ({total:,} new decisions)", flush=True)


if __name__ == "__main__":
    main()
