from typing import List, Optional
import numpy as np
from core.cards import Card

class GameState:
    def __init__(self, current_player: int, round_number: int):
        self.current_player = current_player
        self.round_number = round_number

class RoundState:
    def __init__(self, draw_deck: List[Card], discard_pile: List[Card]):
        self.draw_deck = draw_deck
        self.discard_pile = discard_pile
    
    @property
    def num_discard_pile(self) -> int:
        return len(self.discard_pile)

class PlayerState:
    def __init__(self, player_number: int, num_cards: int):
        self.player_number = player_number
        self.num_cards = num_cards
        self.individual_scores: List[int] = []
        self.reset()

    @property
    def cards(self):
        return self._cards
    
    @cards.setter
    def cards(self, cards: List[Card]):
        assert len(cards) == self.num_cards
        self._cards = cards

    def is_face_up(self, idx: int) -> bool:
        return self.face_up[idx]
    
    def is_face_up_all(self):
        return all(self.face_up)

    def is_privately_seen(self, idx: int) -> bool:
        return self.private_seen[idx]

    @property
    def num_face_down(self) -> int:
        return sum(1 for up in self.face_up if not up)

    def flip(self, idx: int):
        if self.face_up[idx] is True:
            raise ValueError(f'Card index {idx} is already flipped up')
        self.face_up[idx] = True

    def reset(self, cards: Optional[List[Card]] = None):
        if cards is None:
            self.cards = [None] * self.num_cards
        else:
            self.cards = cards
        self.face_up = np.array([False] * self.num_cards)
        self.private_seen = np.array([True] * (self.num_cards // 2) + [False] * (self.num_cards // 2))
    
    @property
    def score(self) -> int:
        return sum(self.individual_scores)
