from dataclasses import dataclass, field
from typing import Optional
from game.cards import Card
from game.phases import Phase

import numpy as np
import copy
import warnings

from collections import Counter

@dataclass
class GameState:
    num_players: int
    num_decks: int
    num_rounds: int
    num_cards: int = 4

    def __post_init__(self):
        assert type(self.num_players) is int
        assert type(self.num_decks) is int
        assert type(self.num_rounds) is int
        assert type(self.num_cards) is int
        assert self.num_cards % 2 == 0
        assert 2 <= self.num_cards <= 20
        assert 2 <= self.num_players <= 12
        assert 1 <= self.num_decks <= 4
        assert 1 <= self.num_rounds <= 10

        # Need at least 2 cards left over (1 draw pile + 1 discard) to deal and reshuffle
        dealt = self.num_players * self.num_cards
        assert dealt <= self.num_decks * 54 - 2, f"Too many cards dealt ({dealt}) for {self.num_decks} deck(s)"
        # Each turn draws at most one card from the deck, and a round lasts at most ~dealt turns
        if 2 * dealt + 1 > self.num_decks * 54:
            warnings.warn(
                f"{self.num_players} players x {self.num_cards} cards with {self.num_decks} deck(s): "
                "the draw deck may run out, and reshuffling the discard pile may be required."
            )

        self.current_player = 0
        self.current_round = 0
        self.complete = False

        self.num_total_cards = self.num_decks * 54
        self.num_ranks = 13

        self.min_possible_score = -5 * self.num_cards
        self.max_possible_score = 10 * self.num_cards

@dataclass
class RoundState:
    draw_deck: list[Card] = field(default_factory=list)
    discard_pile: list[Card] = field(default_factory=list)

    def __post_init__(self):
        self.complete = False
        self.phase = Phase.DRAW

    @property
    def num_discard(self) -> int:
        return len(self.discard_pile)

@dataclass
class PlayerState:
    player_number: int
    num_cards: int

    def __post_init__(self):
        self.individual_scores: list[int] = []
        self.tiebreak_scores: list[int] = []
        self.reset()

    def reset(self, cards: Optional[list[Optional[Card]]] = None):
        if cards is None:
            self._cards = [None] * self.num_cards
        else:
            self._cards = cards
        self.face_up = np.zeros((self.num_cards), dtype=bool)
        self.private_seen = np.zeros((self.num_cards), dtype=bool)
        self.private_seen[:self.num_cards//2] = True
        self.holding_cards = []
        self.num_to_flip = 0
        self.action = (None, None)
        self.previous_known_score = self.known_score if cards is not None else 0

    @property
    def cards(self):
        return self._cards

    @cards.setter
    def cards(self, cards: list[Optional[Card]]):
        if len(cards) != self.num_cards:
            raise ValueError(f"Expected {self.num_cards} cards, got {len(cards)}")
        self._cards = cards

    @property
    def card_suits(self):
        return np.array([c.suit for c in self.cards])
    
    @property
    def card_ranks(self):
        return np.array([c.rank for c in self.cards])

    def is_face_up(self, idx: int) -> bool:
        return self.face_up[idx]
    
    def is_face_down(self, idx: int) -> bool:
        return not self.face_up[idx]

    def is_face_up_all(self) -> bool:
        return all(self.face_up)
    
    @property
    def face_down(self):
        return ~self.face_up

    def is_privately_seen(self, idx: int) -> bool:
        return self.private_seen[idx]

    @property
    def num_face_down(self) -> int:
        return sum(1 for up in self.face_up if not up)

    def flip(self, idx: int):
        if self.face_up[idx]:
            raise ValueError(f'Card index {idx} is already flipped up')
        self.face_up[idx] = True

    def declare_joker(self, rank):
        assert np.any(self.visible_undeclared_jokers)
        idx = np.argmax(self.visible_undeclared_jokers)
        assert self.is_face_up(idx)
        assert self.cards[idx].rank == 13 and self.cards[idx].suit == 4
        self.cards[idx].rank = rank

    @property
    def known_score(self) -> int:
        cards = copy.deepcopy(self.cards + self.holding_cards)
        known = self.face_up | self.private_seen
        known = np.concatenate([known, np.ones_like(self.holding_cards, dtype=bool)])
        # Assign each known, undeclared joker its best possible rank
        for card, k in zip(cards, known):
            if k and card.rank == 13:
                best, best_rank = None, 13
                for r in range(13):
                    card.rank = r
                    counts = Counter(c.rank for c, kk in zip(cards, known) if kk)
                    score = sum(rank_score(rank, n) for rank, n in counts.items())
                    if best is None or score < best:
                        best, best_rank = score, r
                card.rank = best_rank

        card_counts = Counter(c.rank for c, k in zip(cards, known) if k)
        return sum(rank_score(rank, count) for rank, count in card_counts.items())

    @property
    def flipped_score(self) -> int:
        card_counts = Counter(c.rank for c, up in zip(self.cards, self.face_up) if up)
        return sum(rank_score(rank, count) for rank, count in card_counts.items())
    
    @property
    def final_flipped_score(self) -> int:
        assert self.is_face_up_all()
        return self.flipped_score

    @property
    def total_score(self) -> int:
        return sum(self.individual_scores)
    
    @property
    def visible_undeclared_jokers(self):
        return self.face_up & (self.card_suits == 4) & (self.card_ranks == 13)

def rank_score(rank, count):
    assert 0 <= rank <= 13
    assert count >= 1

    # Jokers
    if rank == 13:
        return 0

    # Kings
    if rank == 12:
        if count == 1 or count == 2:
            return 0
        if count == 3:
            return -10
        elif count >= 4:
            return -5 * count

    # Fives
    if rank == 4:
        return -5 * count

    # Face value
    if rank in [10,11]:
        face_value = 10
    elif rank == 0:
        face_value = 1
    else:
        face_value = int(rank + 1)

    # General grouping rules
    if count == 1:
        return face_value
    elif count == 2:
        return 0
    elif count == 3:
        return -face_value
    elif count >= 4:
        return -5 * count

    raise ValueError(f'No score determined. Rank: {rank}. Count: {count}')

