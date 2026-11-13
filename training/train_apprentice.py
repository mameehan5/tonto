"""Expert iteration, apprentice step: train the network to copy SearchBot's moves.

Usage: python training/train_apprentice.py data/exit1 --init models/bc_2p8c/bc.zip --out models/exit1/apprentice.zip
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO

from env.selfplay import SelfPlayEnv, save_config, load_config
from training.imitate import fit, collect
from training.obs_greedy import greedy_from_obs


def improve_labels(X, M, A, Q, N, delta):
    """Greedy's move unless the search rates its own choice better by more than `delta`."""
    labels = A.copy()
    for i in range(len(X)):
        g = greedy_from_obs(X[i], M[i], N)
        if Q[i, A[i]] - Q[i, g] <= delta:
            labels[i] = g
    return labels


def load(dirs):
    shards = [np.load(f) for d in dirs for f in sorted(Path(d).glob("shard_*.npz"))]
    cat = lambda k: np.concatenate([s[k] for s in shards])
    return cat("obs"), cat("masks"), cat("actions"), cat("value"), cat("q"), len(shards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+", help="data directories (all shards are used)")
    ap.add_argument("--init", help="model to start from (else a fresh network)")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--cards", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--delta", type=float, default=None,
                    help="label = greedy's move unless search's move is better by > DELTA (denoises targets)")
    ap.add_argument("--greedy-games", type=int, default=0, help="mix in greedy self-play decisions from this many games")
    ap.add_argument("--soft", type=float, default=0, help="soft targets softmax(q/SOFT) instead of argmax")
    ap.add_argument("--out", default="models/exit1/apprentice.zip")
    a = ap.parse_args()

    X, M, A, V, Q, n = load(a.data)
    print(f"{len(X):,} decisions from {n} shards", flush=True)
    if a.delta is not None:
        A2 = improve_labels(X, M, A, Q, a.cards, a.delta)
        print(f"delta={a.delta}: search move kept for {np.mean(A2 == A):.1%} of decisions "
              f"(greedy agrees with search on {np.mean([greedy_from_obs(X[i], M[i], a.cards) == A[i] for i in range(0, len(X), 10)]):.1%})", flush=True)
        A = A2
    if a.greedy_games:
        from multiprocessing import Pool
        with Pool() as pool:
            parts = pool.map(collect, [(500_000 + i * 1000, 1000, a.players, a.cards, 0.0, 1.0, 0.05)
                                       for i in range(a.greedy_games // 1000)])
        Xg, Mg, Ag, Vg = (np.concatenate(z) for z in zip(*parts))
        print(f"+ {len(Xg):,} greedy decisions", flush=True)
        X, M, A, V = np.concatenate([X, Xg]), np.concatenate([M, Mg]), np.concatenate([A, Ag]), np.concatenate([V, Vg])
    env = SelfPlayEnv(a.players, a.cards)
    if a.init:
        model = MaskablePPO.load(a.init, env=env, device="cpu")
    else:
        model = MaskablePPO("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]}, device="cpu", verbose=0)
    fit(model, X, M, A, V, a.epochs, lr=a.lr, Q=Q if a.soft else None, tau=a.soft or 0.1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out)
    save_config(a.out, {**env.config(), "greedy_jokers": True})  # joker calls aren't in search data
    print("saved", a.out)


if __name__ == "__main__":
    main()
