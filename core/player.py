from typing import List, Set, Optional
from core.cards import Card

class PlayerState:
    def __init__(self, cards: List[Card]):
        self.cards: List[Card] = cards
        self.num_cards = len(cards)
        self.face_up = [False] * self.num_cards
        self.private_seen = [True] * (self.num_cards // 2) + [False] * (self.num_cards // 2)

    def is_face_up(self, idx: int) -> bool:
        return self.face_up[idx]
    
    def all_face_up(self) -> bool:
        return all(self.face_up)

    def is_privately_seen(self, idx: int) -> bool:
        return self.private_seen[idx]

    def flip(self, idx: int):
        if self.face_up[idx] is True:
            raise ValueError(f'Card index {idx} is already flipped up')
        self.face_up[idx] = True
