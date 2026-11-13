"""Reconstruct GreedyBot's move from a self-play observation (greedy only uses information in the obs)."""
from types import SimpleNamespace

import numpy as np

from env.selfplay import NC, UNKNOWN
from game.bots import GreedyBot
from game.phases import Phase

_greedy = GreedyBot()


def greedy_from_obs(x, mask, N):
    o = 0
    cards, face_up, seen = [], [], []
    for i in range(N):
        up = bool(x[o + 16 * i])
        r = int(np.argmax(x[o + 16 * i + 1: o + 16 * i + 16]))
        cards.append(SimpleNamespace(rank=r if r != UNKNOWN else 0))
        face_up.append(up)
        seen.append(r != UNKNOWN)
    o += 16 * N
    holding = [SimpleNamespace(rank=int(np.argmax(x[o + NC * i: o + NC * (i + 1)])))
               for i in range(N) if x[o + NC * i: o + NC * (i + 1)].any()]
    o += NC * N
    discard = [SimpleNamespace(rank=int(np.argmax(x[o + NC * i: o + NC * (i + 1)])))
               for i in range(N) if x[o + NC * i: o + NC * (i + 1)].any()][::-1]  # obs is top-first
    phase = Phase(int(np.argmax(x[len(x) - 9: len(x) - 4])))
    player = SimpleNamespace(cards=cards, face_up=np.array(face_up), private_seen=np.array(seen),
                             holding_cards=holding)
    legal = [int(a) for a in np.flatnonzero(mask)]
    game = SimpleNamespace(
        player_state={0: player}, phase=phase, num_cards=N, legal_actions=lambda: legal,
        round_state=SimpleNamespace(discard_pile=discard),
        max_draw_from_discard=sum(1 for a in legal if 1 <= a <= 20),  # PICKUP_DISCARD actions
    )
    return _greedy.act(game, 0)
