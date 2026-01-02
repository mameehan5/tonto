from dataclasses import dataclass, field
from typing import Optional
from game.cards import Card

import numpy as np

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

        self.current_player = 0
        self.current_round = 0
        self.complete = False

        self.max_num_cards = 20

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
        self.phase = 'draw'

    @property
    def num_discard(self) -> int:
        return len(self.discard_pile)

@dataclass
class PlayerState:
    player_number: int
    num_cards: int
    # cards: list[Optional[Card]] = field(init=False, repr=False)

    def __post_init__(self):
        self.individual_scores: list[int] = []
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

    @property
    def cards(self) -> list[Optional[Card]]:
        return self._cards
    
    @property
    def face_down(self):
        return ~self.face_up

    @cards.setter
    def cards(self, cards: list[Optional[Card]]):
        if len(cards) != self.num_cards:
            raise ValueError(f"Expected {self.num_cards} cards, got {len(cards)}")
        self._cards = cards

    def is_face_up(self, idx: int) -> bool:
        return self.face_up[idx]
    
    def is_face_down(self, idx: int) -> bool:
        return not self.face_up[idx]

    def is_face_up_all(self) -> bool:
        return all(self.face_up)

    def is_privately_seen(self, idx: int) -> bool:
        return self.private_seen[idx]

    @property
    def num_face_down(self) -> int:
        return sum(1 for up in self.face_up if not up)

    def flip(self, idx: int):
        if self.face_up[idx]:
            raise ValueError(f'Card index {idx} is already flipped up')
        self.face_up[idx] = True

    @property
    def score(self) -> int:
        return sum(self.individual_scores)