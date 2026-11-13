"""Train a MaskablePPO agent against bots and frozen copies of itself.

Usage: python train_selfplay.py --players 2 --cards 4 --steps 2000000 --out models/sp_2p4c

Writes to --out: latest.zip, best.zip (best win rate vs greedy), snapshots/, eval.csv, eval.png
"""
import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from env.selfplay import SelfPlayEnv, save_config
from eval_bots import evaluate
from training.policy_bot import PolicyBot


class EvalCallback(BaseCallback):
    """Every `every` steps: win rates vs greedy and random, plot, best model, and a self-play snapshot."""

    def __init__(self, a, config):
        super().__init__()
        self.a, self.config, self.out = a, config, Path(a.out)
        self.rows, self.next_eval, self.next_snap, self.best = [], 0, a.snapshot_every, -1
        self.t0 = time.time()

    def _on_training_start(self):
        # Schedule relative to the starting step (matters when resuming)
        self.next_eval = self.num_timesteps
        self.next_snap = self.num_timesteps + self.a.snapshot_every

    def _on_step(self):
        n = self.num_timesteps
        if n >= self.next_eval:
            self.next_eval += self.a.eval_every
            self._eval(n)
        if self.a.snapshot_every and n >= self.next_snap:
            self.next_snap += self.a.snapshot_every
            path = self.out / "snapshots" / f"snap_{n}.zip"
            self.model.save(path)
            save_config(path, self.config)
            self.training_env.env_method("add_opponent", f"ppo:{path}")
        return True

    def _eval(self, n):
        a = self.a
        me = PolicyBot(model=self.model, config=self.config)
        row = {"steps": n, "minutes": round((time.time() - self.t0) / 60, 1)}
        for opp in ["greedy", "random"]:
            wins, scores = evaluate([me] + [opp] * (a.players - 1), a.eval_games, cards=a.cards, seed0=10_000)
            row[f"win_{opp}"] = wins[0] / a.eval_games
            row[f"score_{opp}"] = float(np.mean(scores[0]))
        self.rows.append(row)
        self.model.save(self.out / "latest.zip")
        save_config(self.out / "latest.zip", self.config)
        if row["win_greedy"] > self.best:
            self.best = row["win_greedy"]
            self.model.save(self.out / "best.zip")
            save_config(self.out / "best.zip", self.config)
        with open(self.out / "eval.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(self.rows)
        self._plot()
        print(f"[{row['minutes']:5.1f} min] {n:>9,} steps | win vs greedy {row['win_greedy']:.1%} "
              f"(score {row['score_greedy']:.2f}) | vs random {row['win_random']:.1%} | best {self.best:.1%}", flush=True)

    def _plot(self):
        a = self.a
        steps = [r["steps"] / 1e6 for r in self.rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, [r["win_random"] for r in self.rows], "o-", color="#9aa5b1", label="vs random")
        ax.plot(steps, [r["win_greedy"] for r in self.rows], "o-", color="#2a7556", label="vs greedy")
        even = 1 / a.players
        ax.axhline(even, ls="--", color="#c0392b", lw=1, label=f"even skill ({even:.0%})")
        ax.set_xlabel("training steps (millions)")
        ax.set_ylabel("PPO win rate")
        ax.set_ylim(0, 1)
        ax.set_title(f"Self-play PPO: {a.players} players, {a.cards} cards")
        ax.grid(alpha=.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(self.out / "eval.png", dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--cards", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--opponents", default="greedy,greedy,random", help="initial pool; repeats weight sampling")
    ap.add_argument("--snapshot-every", type=int, default=250_000, help="0 disables self-play snapshots")
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--eval-games", type=int, default=400)
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--shaping", type=float, default=0.05, help="reward per point of estimated hand score")
    ap.add_argument("--win", type=float, default=1.0, help="terminal reward for beating the best opponent")
    ap.add_argument("--margin", type=float, default=0.02, help="terminal reward per point of score margin")
    ap.add_argument("--clip", type=float, default=0.2, help="PPO clip range (used with --resume)")
    ap.add_argument("--resume", help="path to a model .zip to continue training (e.g. from training/imitate.py)")
    ap.add_argument("--out", default="models/selfplay")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    opps = tuple(a.opponents.split(","))
    make = lambda i: lambda: SelfPlayEnv(a.players, a.cards, opponents=opps, shaping=a.shaping, margin=a.margin, win=a.win, seed=a.seed + i)
    venv = (SubprocVecEnv if a.n_envs > 1 else DummyVecEnv)([make(i) for i in range(a.n_envs)])
    config = SelfPlayEnv(a.players, a.cards).config()

    if a.resume:
        # Override hyperparameters saved with the model (e.g. a behavior-cloned warm start)
        model = MaskablePPO.load(a.resume, env=venv, device="cpu", custom_objects={
            "learning_rate": a.lr, "lr_schedule": lambda _: a.lr, "ent_coef": a.ent, "clip_range": lambda _: a.clip,
            "n_steps": 2048, "batch_size": 512, "n_epochs": 6, "gamma": 0.995, "gae_lambda": 0.95})
    else:
        model = MaskablePPO(
            "MlpPolicy", venv, learning_rate=a.lr, n_steps=2048, batch_size=512, n_epochs=6,
            gamma=0.995, gae_lambda=0.95, ent_coef=a.ent, policy_kwargs={"net_arch": [256, 256]},
            device="cpu", seed=a.seed, verbose=0,
        )
    print(f"Training {a.players}p/{a.cards} cards for {a.steps:,} steps -> {out}", flush=True)
    model.learn(a.steps, callback=EvalCallback(a, config), reset_num_timesteps=not a.resume)


if __name__ == "__main__":
    main()
