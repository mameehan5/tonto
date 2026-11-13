"""Self-play training env: the learner controls one random seat, bots play the others.

Observation (flat, from the learner's own view):
  own slots      [face_up, card one-hot(15)] x obs_cards   (peeked face-down cards are visible)
  holding        card one-hot(15) x obs_cards
  discard top    card one-hot(15) x obs_cards              (top card first)
  opponents      [exists, face_up, card one-hot(15)] x obs_cards x (obs_players - 1), next to act first
  scalars        phase one-hot(5), num_to_flip, deck size, discard size, my estimated score
Card one-hot: ranks 0-12, 13 = undeclared joker, 14 = unknown.

Reward: terminal +1 win / -1 loss / 0 tie against the best opponent, plus a small score-margin term,
plus potential-based shaping on the learner's estimated hand score (telescopes, so it doesn't change
which hands are best).
"""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game.actions import Action
from game.bots import estimate, make_bot
from game.engine import TontoGame
from game.phases import Phase

NC = 15  # card one-hot size
UNKNOWN = 14
TRUE_UNKNOWN_VALUE = 4.5  # mean value of an unseen card, used for shaping


def obs_size(obs_players, obs_cards):
    return obs_cards * (16 + NC + NC) + (obs_players - 1) * obs_cards * 17 + 9


def encode_obs(game, pid, obs_players=None, obs_cards=None):
    P = obs_players or game.num_players
    N = obs_cards or game.num_cards
    x = np.zeros(obs_size(P, N), dtype=np.float32)
    me = game.player_state[pid]
    o = 0

    for i, c in enumerate(me.cards):
        known = me.face_up[i] or me.private_seen[i]
        x[o + 16 * i] = me.face_up[i]
        x[o + 16 * i + 1 + (c.rank if known else UNKNOWN)] = 1
    o += 16 * N
    for i, c in enumerate(me.holding_cards):
        x[o + NC * i + c.rank] = 1
    o += NC * N
    for i, c in enumerate(reversed(game.round_state.discard_pile[-N:])):
        x[o + NC * i + c.rank] = 1
    o += NC * N
    for k in range(1, game.num_players):
        opp = game.player_state[(pid + k) % game.num_players]
        base = o + (k - 1) * 17 * N
        for i, c in enumerate(opp.cards):
            x[base + 17 * i] = 1
            x[base + 17 * i + 1] = opp.face_up[i]
            x[base + 17 * i + 2 + (c.rank if opp.face_up[i] else UNKNOWN)] = 1
    o += (P - 1) * 17 * N

    x[o + int(game.phase)] = 1
    rs = game.round_state
    x[o + 5] = me.num_to_flip / N
    x[o + 6] = len(rs.draw_deck) / (54 * game.num_decks)
    x[o + 7] = len(rs.discard_pile) / (54 * game.num_decks)
    x[o + 8] = estimated_score(me) / 20
    return x


def estimated_score(p):
    known = [c.rank for c, up, seen in zip(p.cards, p.face_up, p.private_seen) if up or seen]
    return estimate(known, len(p.cards) - len(known), TRUE_UNKNOWN_VALUE)


class SelfPlayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, num_players=2, num_cards=4, num_decks=1, opponents=("greedy",),
                 shaping=0.05, margin=0.02, win=1.0, obs_players=None, obs_cards=None, seed=None):
        self.P, self.N, self.D = num_players, num_cards, num_decks
        self.obs_players = obs_players or num_players
        self.obs_cards = obs_cards or num_cards
        self.shaping, self.margin, self.win = shaping, margin, win
        self.pool = [make_bot(s) for s in opponents]
        self.seed = seed
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size(self.obs_players, self.obs_cards),), np.float32)
        self.action_space = spaces.Discrete(int(Action.NUM_POSSIBLE_ACTIONS))

    def add_opponent(self, spec):
        """Add a bot spec (e.g. 'ppo:models/snap_100000.zip') to the opponent pool."""
        self.pool.append(make_bot(spec))

    def config(self):
        return {"obs": "selfplay_v1", "obs_players": self.obs_players, "obs_cards": self.obs_cards,
                "num_players": self.P, "num_cards": self.N, "num_decks": self.D}

    def reset(self, seed=None, options=None):
        super().reset(seed=self.seed if seed is None and not hasattr(self, "game") else seed)
        self.rng = self.np_random
        self.game = TontoGame(self.P, self.D, 1, self.N, seed=self.rng)
        self.me = int(self.rng.integers(self.P))
        self.opps = {i: self.pool[self.rng.integers(len(self.pool))] for i in range(self.P) if i != self.me}
        self._advance()
        self.phi = self._potential()
        return self._obs(), {}

    def step(self, action):
        if not self.game.is_legal(int(action)):
            return self._obs(), -1.0, False, True, {}
        self.game.step(int(action))
        self._advance()
        done = self.game.round_state.complete
        phi = self._potential()
        reward = self.shaping * (phi - self.phi)
        self.phi = phi
        if done:
            scores = [p.final_flipped_score for p in self.game.player_state]
            mine = scores[self.me]
            best_opp = min(s for i, s in enumerate(scores) if i != self.me)
            reward += self.win * np.sign(best_opp - mine) + self.margin * (best_opp - mine)
        return self._obs(), float(reward), done, False, {}

    def action_masks(self):
        return self.game.legal_mask().astype(bool)

    def _potential(self):
        return -estimated_score(self.game.player_state[self.me])

    def _obs(self):
        return encode_obs(self.game, self.me, self.obs_players, self.obs_cards)

    def _advance(self):
        """Play opponents and auto-apply forced moves until the learner has a real choice or the round ends."""
        g = self.game
        while not g.round_state.complete:
            pid = g.game_state.current_player
            legal = g.legal_actions()
            if pid != self.me:
                g.step(self.opps[pid].act(g, pid))
            elif len(legal) == 1:
                g.step(legal[0])
            else:
                return


def save_config(model_path, config):
    Path(model_path).with_suffix(".json").write_text(json.dumps(config, indent=1))


def load_config(model_path):
    p = Path(model_path).with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else None
