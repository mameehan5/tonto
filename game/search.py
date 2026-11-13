"""Determinized Monte Carlo search bot ("expert" for expert iteration).

For each decision: sample the cards this player can't see (deck, own unpeeked cards, opponents'
face-down cards) many times; for each sample, try every candidate move and play the round out with a
fast rollout policy (GreedyBot by default). Pick the move with the best average result.
"""
import numpy as np

from game.bots import GreedyBot
from game.phases import Phase


def determinize(g, pid, rng):
    """Shuffle every card `pid` can't see among the positions it can't see (in place)."""
    rs = g.round_state
    slots = [(rs.draw_deck, i) for i in range(len(rs.draw_deck))]
    for p in g.player_state:
        for i in range(len(p.cards)):
            hidden = not p.face_up[i] and (p.player_number != pid or not p.private_seen[i])
            if hidden:
                slots.append((p._cards, i))
    cards = [lst[i] for lst, i in slots]
    rng.shuffle(cards)
    for (lst, i), c in zip(slots, cards):
        lst[i] = c


def outcome(g, pid, margin=0.05):
    """+1 win / -1 loss / 0 tie vs the best opponent, plus a small score-margin term."""
    scores = [p.final_flipped_score for p in g.player_state]
    best_opp = min(s for i, s in enumerate(scores) if i != pid)
    return np.sign(best_opp - scores[pid]) + margin * (best_opp - scores[pid])


class SearchBot:
    def __init__(self, samples=16, rollout=None, seed=None):
        self.samples = samples
        self.rollout = rollout or GreedyBot()
        self.greedy = GreedyBot()
        self.rng = np.random.default_rng(seed)
        self.last_values = None  # per-candidate mean values from the last search (for training targets)

    def act(self, game, pid):
        legal = game.legal_actions()
        if len(legal) == 1 or game.phase == Phase.DECLARE_JOKER:
            self.last_values = None
            return legal[0] if len(legal) == 1 else self.greedy.act(game, pid)
        totals = np.zeros(len(legal))
        for _ in range(self.samples):
            base = game.clone()
            determinize(base, pid, self.rng)
            for j, a in enumerate(legal):  # same sampled deal for every move (common random numbers)
                g = base.clone()
                g.step(a)
                self._play_out(g)
                totals[j] += outcome(g, pid)
        self.last_values = dict(zip(legal, totals / self.samples))
        return legal[int(np.argmax(totals))]

    def _play_out(self, g):
        while not g.round_state.complete:
            pid = g.game_state.current_player
            g.step(self.rollout.act(g, pid))
