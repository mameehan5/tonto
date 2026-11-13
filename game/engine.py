"""Pure Tonto game engine: rules, turn flow, scoring, per-player views, serialization.

The Gym env, GUI, and (later) game server all wrap this class.
"""
import copy

import numpy as np

from game.cards import Card, create_standard_deck
from game.state import GameState, RoundState, PlayerState
from game.actions import compute_action_mask, decode_action, RANK_STRS
from game.phases import Phase


def card_dict(c: Card):
    return {"id": c.id, "rank": c.rank, "suit": c.suit, "str": c.card_str, "joker": c.suit == 4}


class TontoGame:
    def __init__(self, num_players, num_decks, num_rounds, num_cards, seed=None):
        self.num_players = num_players
        self.num_decks = num_decks
        self.num_rounds = num_rounds
        self.num_cards = num_cards
        self.rng = np.random.default_rng(seed)  # accepts an int, None, or an existing Generator

        self.game_state = GameState(
            num_players=num_players, num_decks=num_decks,
            num_rounds=num_rounds, num_cards=num_cards,
        )
        self.player_state = [PlayerState(i, num_cards) for i in range(num_players)]
        self.active_players = list(range(num_players))
        self.winner = None
        self.log = []
        self.start_round()

    # Properties --------------------------------------------------------------

    @property
    def current(self) -> PlayerState:
        return self.player_state[self.game_state.current_player]

    @property
    def phase(self) -> Phase:
        return self.round_state.phase

    @property
    def max_draw_from_discard(self):
        return min(self.round_state.num_discard, self.current.num_face_down)

    def legal_mask(self):
        return compute_action_mask(self)

    def legal_actions(self):
        return [int(a) for a in np.flatnonzero(self.legal_mask())]

    def is_legal(self, action):
        return bool(self.legal_mask()[action])

    # Flow --------------------------------------------------------------------

    def start_round(self):
        deck = create_standard_deck(self.num_decks)
        self.rng.shuffle(deck)
        self.round_state = RoundState(draw_deck=deck, discard_pile=[deck.pop()])
        for i in self.active_players:
            self.player_state[i].reset([deck.pop() for _ in range(self.num_cards)])
        if self.game_state.current_player not in self.active_players:
            self.advance_player()
        self._log(None, f"--- {self.round_name} ---")

    @property
    def round_name(self):
        r = self.game_state.current_round
        if self.game_state.complete:
            return "Final"
        return f"Round {r + 1}" if r < self.num_rounds else f"Tiebreak (players {self.active_players})"

    def next_round(self):
        """Score the finished round, then deal the next round or end the game."""
        assert self.round_state.complete
        gs = self.game_state
        tiebreak = gs.current_round >= self.num_rounds
        for i in self.active_players:
            p = self.player_state[i]
            (p.tiebreak_scores if tiebreak else p.individual_scores).append(p.final_flipped_score)
        gs.current_round += 1
        if gs.current_round >= self.num_rounds:
            # Players tied for the lowest score play another round until the tie is broken
            score = lambda i: self.player_state[i].tiebreak_scores[-1] if tiebreak else self.player_state[i].total_score
            low = min(score(i) for i in self.active_players)
            self.active_players = [i for i in self.active_players if score(i) == low]
            if len(self.active_players) == 1:
                gs.complete = True
                self.winner = self.active_players[0]
                self._log(None, f"Game over: player {self.winner} wins")
                return
        self.start_round()

    def advance_player(self):
        gs = self.game_state
        gs.current_player = (gs.current_player + 1) % self.num_players
        while gs.current_player not in self.active_players:
            gs.current_player = (gs.current_player + 1) % self.num_players

    def end_turn(self):
        self.advance_player()
        rs = self.round_state
        rs.complete = all(self.player_state[i].is_face_up_all() for i in self.active_players)
        rs.phase = Phase.END if self.current.is_face_up_all() else Phase.DRAW
        if rs.complete:
            self._log(None, "Round complete: " + ", ".join(
                f"P{i}={self.player_state[i].flipped_score}" for i in self.active_players))

    def reshuffle(self):
        """Shuffle the discard pile (except the top card) into a new draw deck."""
        rs = self.round_state
        rs.draw_deck = rs.discard_pile[:-1]
        rs.discard_pile = rs.discard_pile[-1:]
        self.rng.shuffle(rs.draw_deck)
        self._log(None, f"Deck empty: reshuffled {len(rs.draw_deck)} discards")

    # Actions -----------------------------------------------------------------

    def step(self, action: int):
        """Apply one legal action for the current player."""
        if not self.is_legal(action):
            raise ValueError(f"Illegal action {action} ({decode_action(action)[2]}) in phase {self.phase.name}")
        act, val, _ = decode_action(action)
        p = self.current
        rs = self.round_state
        pid = p.player_number

        if rs.phase == Phase.DRAW:
            if act == "draw_deck":
                if len(rs.draw_deck) == 0:
                    self.reshuffle()
                cards = [rs.draw_deck.pop()]
                self._log(pid, "draws from deck", cards[0].card_str)
            else:
                cards = [rs.discard_pile.pop() for _ in range(val)]
                self._log(pid, "picks up " + " ".join(c.card_str for c in cards) + " from discard")
            p.holding_cards = cards
            p.num_to_flip = len(cards)
            rs.phase = Phase.DISCARD

        elif rs.phase == Phase.DISCARD:
            if act == "discard_drawn":
                c = p.holding_cards.pop()
                rs.discard_pile.append(c)
                rs.phase = Phase.FLIP
                self._log(pid, f"discards drawn {c.card_str}")
            else:
                old = p.cards[val]
                rs.discard_pile.append(old)
                p.cards[val] = p.holding_cards.pop()
                p.flip(val)
                p.num_to_flip -= 1
                self._log(pid, f"replaces slot {val} ({old.card_str}) with {p.cards[val].card_str}")
                if p.num_to_flip == 0:
                    rs.phase = Phase.DECLARE_JOKER if np.any(p.visible_undeclared_jokers) else Phase.END

        elif rs.phase == Phase.FLIP:
            p.flip(val)
            p.num_to_flip -= 1
            self._log(pid, f"flips slot {val}: {p.cards[val].card_str}")
            rs.phase = Phase.DECLARE_JOKER if np.any(p.visible_undeclared_jokers) else Phase.END

        elif rs.phase == Phase.DECLARE_JOKER:
            p.declare_joker(val)
            self._log(pid, f"calls joker as {RANK_STRS[val]}")
            rs.phase = Phase.DECLARE_JOKER if np.any(p.visible_undeclared_jokers) else Phase.END

        elif rs.phase == Phase.END:
            p.action = (act, val)
            self.end_turn()
            return

        p.action = (act, val)

    # Views -------------------------------------------------------------------

    def view(self, viewer=None):
        """What `viewer` can see. viewer=None reveals everything (for debugging/evaluation)."""
        gs, rs = self.game_state, self.round_state
        cur = gs.current_player

        def player_view(p):
            me = viewer is None or viewer == p.player_number
            cards = []
            for i, c in enumerate(p.cards):
                visible = viewer is None or p.face_up[i] or (me and p.private_seen[i])
                cards.append({
                    "card": card_dict(c) if visible else None,
                    "face_up": bool(p.face_up[i]),
                    "seen": bool(p.private_seen[i]),
                })
            # Cards picked up from the discard pile are public; a deck draw is private
            public_hold = p.action[0] == "pickup_discard"
            return {
                "id": p.player_number,
                "active": p.player_number in self.active_players,
                "cards": cards,
                "holding": [card_dict(c) if (me or public_hold) else None for c in p.holding_cards],
                "flipped_score": int(p.flipped_score),
                "known_score": int(p.known_score) if me else None,
                "scores": list(p.individual_scores),
                "tiebreak_scores": list(p.tiebreak_scores),
                "total": int(p.total_score),
            }

        my_turn = viewer is None or viewer == cur
        return {
            "round": gs.current_round,
            "round_name": self.round_name,
            "num_rounds": self.num_rounds,
            "phase": rs.phase.name,
            "current_player": cur,
            "round_complete": rs.complete,
            "game_complete": gs.complete,
            "winner": self.winner,
            "deck_count": len(rs.draw_deck),
            "discard_top": [card_dict(c) for c in rs.discard_pile[-self.num_cards:]],
            "discard_count": len(rs.discard_pile),
            "players": [player_view(p) for p in self.player_state],
            "legal_actions": [
                {"action": a, "label": self.action_label(a)} for a in self.legal_actions()
            ] if my_turn and not rs.complete else [],
            # `detail` holds private info (e.g. the card drawn from the deck)
            "log": [e if viewer is None or e["player"] == viewer else {**e, "detail": None} for e in self.log],
        }

    def action_label(self, action):
        act, val, text = decode_action(action)
        p = self.current
        if act == "pickup_discard":
            cards = self.round_state.discard_pile[-val:][::-1]
            return f"Pick up {' '.join(c.card_str for c in cards)}"
        if act == "discard_own":
            return f"Replace slot {val}"
        if act == "flip":
            return f"Flip slot {val}"
        if act == "discard_drawn":
            return f"Discard drawn {p.holding_cards[-1].card_str}"
        return text

    def _log(self, player, text, detail=None):
        if self.log is None:  # logging disabled (search rollouts)
            return
        self.log.append({"player": player, "text": text, "detail": detail})

    # Fast copy for simulations ------------------------------------------------

    def clone(self):
        """Independent copy for search rollouts (log not copied; rng is fresh)."""
        cc = lambda cs: [copy.copy(c) if c.suit == 4 else c for c in cs]  # only jokers are mutated
        g = copy.copy(self)
        g.rng = np.random.default_rng(self.rng.integers(1 << 62))
        g.log = None
        g.active_players = list(self.active_players)
        g.game_state = copy.copy(self.game_state)
        rs = copy.copy(self.round_state)
        rs.draw_deck, rs.discard_pile = cc(rs.draw_deck), cc(rs.discard_pile)
        g.round_state = rs
        g.player_state = []
        for p in self.player_state:
            q = copy.copy(p)
            q._cards, q.holding_cards = cc(p._cards), cc(p.holding_cards)
            q.face_up, q.private_seen = p.face_up.copy(), p.private_seen.copy()
            q.individual_scores, q.tiebreak_scores = list(p.individual_scores), list(p.tiebreak_scores)
            g.player_state.append(q)
        return g

    # Serialization -----------------------------------------------------------

    def to_dict(self):
        cards = lambda cs: [[c.id, c.rank, c.suit] for c in cs]
        gs, rs = self.game_state, self.round_state
        return {
            "config": [self.num_players, self.num_decks, self.num_rounds, self.num_cards],
            "rng": self.rng.bit_generator.state,
            "game": {"current_player": gs.current_player, "current_round": gs.current_round,
                     "complete": gs.complete, "active_players": self.active_players, "winner": self.winner},
            "round": {"draw_deck": cards(rs.draw_deck), "discard_pile": cards(rs.discard_pile),
                      "phase": int(rs.phase), "complete": rs.complete},
            "players": [{
                "cards": cards(p.cards), "face_up": p.face_up.tolist(), "private_seen": p.private_seen.tolist(),
                "holding_cards": cards(p.holding_cards), "num_to_flip": p.num_to_flip, "action": list(p.action),
                "individual_scores": p.individual_scores, "tiebreak_scores": p.tiebreak_scores,
            } for p in self.player_state],
            "log": self.log,
        }

    @classmethod
    def from_dict(cls, d):
        cards = lambda cs: [Card(id=i, rank=r, suit=s) for i, r, s in cs]
        g = cls.__new__(cls)
        g.num_players, g.num_decks, g.num_rounds, g.num_cards = d["config"]
        g.rng = np.random.default_rng()
        g.rng.bit_generator.state = d["rng"]
        g.game_state = GameState(num_players=g.num_players, num_decks=g.num_decks,
                                 num_rounds=g.num_rounds, num_cards=g.num_cards)
        gd = d["game"]
        g.game_state.current_player = gd["current_player"]
        g.game_state.current_round = gd["current_round"]
        g.game_state.complete = gd["complete"]
        g.active_players = list(gd["active_players"])
        g.winner = gd["winner"]
        rd = d["round"]
        g.round_state = RoundState(draw_deck=cards(rd["draw_deck"]), discard_pile=cards(rd["discard_pile"]))
        g.round_state.phase = Phase(rd["phase"])
        g.round_state.complete = rd["complete"]
        g.player_state = []
        for i, pd in enumerate(d["players"]):
            p = PlayerState(i, g.num_cards)
            p._cards = cards(pd["cards"])
            p.face_up = np.array(pd["face_up"], dtype=bool)
            p.private_seen = np.array(pd["private_seen"], dtype=bool)
            p.holding_cards = cards(pd["holding_cards"])
            p.num_to_flip = pd["num_to_flip"]
            p.action = tuple(pd["action"])
            p.individual_scores = list(pd["individual_scores"])
            p.tiebreak_scores = list(pd["tiebreak_scores"])
            g.player_state.append(p)
        g.log = list(d["log"])
        return g
