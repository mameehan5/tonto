from typing import List, Set, Optional
from core.cards import Card

class PlayerState:
    def __init__(self, cards: List[Card]):
        self.cards: List[Card] = cards
        self.face_up: Set[int] = set()
        self.private_seen: Set[int] = set()

    def is_face_up(self, idx: int) -> bool:
        return idx in self.face_up

    def is_privately_seen(self, idx: int) -> bool:
        return idx in self.private_seen

    def flip(self, idx: int):
        if idx not in self.face_up:
            self.face_up.add(idx)
