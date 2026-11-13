"""Behavior cloning: train the PPO network to copy GreedyBot, as a warm start for PPO fine-tuning.

Usage: python training/imitate.py --players 2 --cards 4 --games 20000 --out models/bc_2p4c/bc.zip
The value head is also fit to the self-play reward (same shaping/win/margin as train_selfplay).
"""
import argparse
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np
import torch
from sb3_contrib.ppo_mask import MaskablePPO

from env.selfplay import SelfPlayEnv, encode_obs, estimated_score, save_config
from game.bots import GreedyBot
from game.engine import TontoGame


def collect(args):
    """Play greedy-vs-greedy games; return (obs, masks, actions, value targets) for every real decision."""
    seed0, games, P, N, shaping, win, margin = args
    bot = GreedyBot()
    X, M, A, V = [], [], [], []
    for s in range(seed0, seed0 + games):
        g = TontoGame(P, 1, 1, N, seed=s)
        recs = []  # (pid, index into X, potential at decision)
        while not g.round_state.complete:
            pid = g.game_state.current_player
            legal = g.legal_actions()
            a = bot.act(g, pid)
            if len(legal) > 1:
                recs.append((pid, len(X), -estimated_score(g.player_state[pid])))
                X.append(encode_obs(g, pid))
                M.append(g.legal_mask().astype(bool))
                A.append(a)
                V.append(0.0)
            g.step(a)
        scores = [p.final_flipped_score for p in g.player_state]
        for pid, i, phi in recs:
            best_opp = min(sc for j, sc in enumerate(scores) if j != pid)
            V[i] = shaping * (-scores[pid] - phi) + win * np.sign(best_opp - scores[pid]) + margin * (best_opp - scores[pid])
    return np.array(X), np.array(M), np.array(A), np.array(V, dtype=np.float32)


def fit(model, X, M, A, V, epochs, lr=1e-3, value_coef=0.5, Q=None, tau=0.1):
    """Supervised training of the policy and value head (MSE on V).
    Policy target: the chosen action A, or if Q (per-action search values, NaN = not a candidate) is
    given, the soft target softmax(Q / tau), which is less noisy than the argmax."""
    if Q is not None:
        Qt = np.where(np.isnan(Q), -np.inf, Q / tau)
        P = np.exp(Qt - Qt.max(1, keepdims=True))
        P = (P / P.sum(1, keepdims=True)).astype(np.float32)
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    n_val = len(X) // 10
    perm = np.random.default_rng(0).permutation(len(X))
    val, tr = perm[:n_val], perm[n_val:]
    T = lambda z, idx: torch.as_tensor(z[idx])
    for ep in range(epochs):
        pol.train()
        np.random.shuffle(tr)
        for b in range(0, len(tr), 1024):
            idx = tr[b:b + 1024]
            values, logp, _ = pol.evaluate_actions(T(X, idx), T(A, idx), action_masks=T(M, idx))
            if Q is not None:
                logits = pol.get_distribution(T(X, idx), action_masks=M[idx]).distribution.logits
                logp = (T(P, idx) * logits).sum(1)  # logits are log-probs (normalized); masked ones ~ -1e8 x 0
            loss = -logp.mean() + value_coef * torch.mean((values.flatten() - T(V, idx)) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        pol.eval()
        with torch.no_grad():
            dist = pol.get_distribution(T(X, val), action_masks=M[val])
            acc = (dist.distribution.probs.argmax(1).numpy() == A[val]).mean()
            vloss = torch.mean((pol.predict_values(T(X, val)).flatten() - T(V, val)) ** 2).item()
        print(f"epoch {ep + 1:2d}: held-out accuracy {acc:.1%}  value MSE {vloss:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--cards", type=int, default=4)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--shaping", type=float, default=0.1)
    ap.add_argument("--win", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--out", default="models/bc/bc.zip")
    a = ap.parse_args()

    jobs = [(i * 1000, 1000, a.players, a.cards, a.shaping, a.win, a.margin) for i in range(a.games // 1000)]
    with Pool() as pool:
        parts = pool.map(collect, jobs)
    X, M, A, V = (np.concatenate(z) for z in zip(*parts))
    print(f"{len(X):,} decisions from {a.games:,} games", flush=True)

    env = SelfPlayEnv(a.players, a.cards)
    model = MaskablePPO("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]}, device="cpu", verbose=0)
    fit(model, X, M, A, V, a.epochs)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out)
    save_config(a.out, env.config())
    print("saved", a.out)


if __name__ == "__main__":
    main()
