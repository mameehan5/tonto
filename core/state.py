from dataclasses import dataclass, field
from typing import List, Optional
from core.cards import Card

@dataclass
class GameState:
    current_player: int
    round_number: int
    complete: bool = False

@dataclass
class RoundState:
    draw_deck: List[Card] = field(default_factory=list)
    discard_pile: List[Card] = field(default_factory=list)
    complete: bool = False

    @property
    def discard_count(self) -> int:
        return len(self.discard_pile)

@dataclass
class PlayerState:
    player_number: int
    num_cards: int
    individual_scores: List[int] = field(default_factory=list)
    _cards: List[Optional[Card]] = field(init=False, repr=False)
    face_up: List[bool] = field(init=False, repr=False)
    private_seen: List[bool] = field(init=False, repr=False)

    # TODO: rename face up to flipped

    def __post_init__(self):
        self.reset()

    @property
    def cards(self) -> List[Optional[Card]]:
        return self._cards

    @cards.setter
    def cards(self, cards: List[Optional[Card]]):
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

    def reset(self, cards: Optional[List[Optional[Card]]] = None):
        if cards is None:
            self.cards = [None] * self.num_cards
        else:
            self.cards = cards
        self.face_up = [False] * self.num_cards
        half = self.num_cards // 2
        self.private_seen = [True] * half + [False] * half

    @property
    def score(self) -> int:
        return sum(self.individual_scores)