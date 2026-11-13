"""Rule-based bots. Each bot decides using only its own player view."""
from collections import Counter
from functools import lru_cache

import numpy as np

from game.actions import Action
from game.state import rank_score

UNKNOWN_VALUE = 2.5  # value assumed for an unseen card (tuned; below its true mean ~4.5, which favors keeping unknowns)
JOKER = 13


def card_value(rank):
    return rank_score(rank, 1) if rank != JOKER else -5


def estimate(ranks, n_unknown, unknown_value=UNKNOWN_VALUE):
    """Estimated hand score: known ranks grouped by the scoring rules, unknowns at their expected value.
    Jokers are assigned whichever rank gives the lowest score."""
    fixed = [r for r in ranks if r != JOKER]
    for _ in range(len(ranks) - len(fixed)):
        fixed.append(min(range(13), key=lambda r: _score(fixed + [r])))
    return _score(fixed) + unknown_value * n_unknown


def _score(ranks):
    return _score_sorted(tuple(sorted(ranks)))


@lru_cache(maxsize=200_000)
def _score_sorted(ranks):
    return sum(rank_score(r, n) for r, n in Counter(ranks).items())


class RandomBot:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, game, pid):
        return int(self.rng.choice(game.legal_actions()))


class GreedyBot:
    """One-step greedy on the estimated hand score."""

    def __init__(self, pickup_threshold=5.0, unknown_value=UNKNOWN_VALUE):
        self.pickup_threshold = pickup_threshold
        self.unknown_value = unknown_value

    def act(self, game, pid):
        # Read only what this player can see (equivalent to game.view(pid), but much faster)
        p = game.player_state[pid]
        slots = [c.rank if (up or seen) else None for c, up, seen in zip(p.cards, p.face_up, p.private_seen)]
        face_down = [i for i, up in enumerate(p.face_up) if not up]
        legal = set(game.legal_actions())
        phase = game.phase.name
        discard_top = [c.rank for c in game.round_state.discard_pile[-game.num_cards:]]

        def est_with(replacements):
            s = list(slots)
            for i, r in replacements.items():
                s[i] = r
            return estimate([r for r in s if r is not None], s.count(None), self.unknown_value)

        def best_slot(rank, used=()):
            options = [i for i in face_down if i not in used]
            return min(options, key=lambda i: est_with({i: rank}))

        now = est_with({})

        if phase == "DRAW":
            # Consider picking up the top n discards, placed greedily into face-down slots
            top = discard_top[::-1]  # top card first
            best_n, best_gain = 0, self.pickup_threshold
            for n in range(1, game.max_draw_from_discard + 1):
                repl = {}
                for rank in top[:n][::-1]:  # placed in the order they're popped
                    repl[best_slot(rank, repl)] = rank
                gain = now - est_with(repl)
                if gain > best_gain:
                    best_n, best_gain = n, gain
            return Action.PICKUP_DISCARD_0 + best_n - 1 if best_n else Action.DRAW_DECK

        if phase == "DISCARD":
            rank = p.holding_cards[-1].rank
            i = best_slot(rank)
            if Action.DISCARD_DRAWN in legal and est_with({i: rank}) >= now:
                return Action.DISCARD_DRAWN
            return Action.DISCARD_OWN_0 + i

        if phase == "FLIP":
            # Commit the best known card; otherwise flip an unknown one
            seen = [i for i in face_down if slots[i] is not None]
            if seen:
                return Action.FLIP_0 + min(seen, key=lambda i: card_value(slots[i]))
            return Action.FLIP_0 + face_down[0]

        if phase == "DECLARE_JOKER":
            j = next(i for i, c in enumerate(p.cards) if p.face_up[i] and c.rank == JOKER)
            return Action.DECLARE_JOKER_0 + min(range(13), key=lambda r: est_with({j: r}))

        return Action.END_TURN


BOTS = {
    "greedy": GreedyBot,
    "greedy_v0": lambda: GreedyBot(pickup_threshold=2.5, unknown_value=4.5),  # untuned original
    "random": RandomBot,
}


def make_bot(spec, seed=None):
    """'greedy', 'greedy_v0', 'random', 'search[:samples]', or 'ppo:<path to model.zip>'."""
    if spec.startswith("search"):
        from game.search import SearchBot
        return SearchBot(int(spec.split(":")[1]) if ":" in spec else 16, seed=seed)
    if spec.startswith("ppo:"):
        from training.policy_bot import PolicyBot  # needs torch/sb3; import only when used
        return PolicyBot(spec[4:])
    return RandomBot(seed) if spec == "random" else BOTS[spec]()
