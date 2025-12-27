from typing import List, Set
from core.cards import Card
from core.phases import TurnPhase

class PlayerState:
    def __init__(self, cards: List[Card]):
        self.cards: List[Card] = cards
        self.num_cards = len(self.cards)
        self.face_up = [False] * self.num_cards
        self.private_seen = [True] * (self.num_cards // 2) + [False] * (self.num_cards // 2)

    def is_face_up(self, idx: int) -> bool:
        return self.face_up[idx]
    
    def all_face_up(self) -> bool:
        return all(self.face_up)

    def is_privately_seen(self, idx: int) -> bool:
        return self.private_seen[idx]

    def count_not_face_up(self) -> int:
        return sum(1 for up in self.face_up if not up)

    def flip(self, idx: int):
        if self.face_up[idx] is True:
            raise ValueError(f'Card index {idx} is already flipped up')
        self.face_up[idx] = True

class GameState:
    def __init__(self, players: List[PlayerState], draw_deck: List[Card], discard_pile: List[Card],
                 current_player: int, round_number: int, rng):
        self.players = players
        self.draw_deck = draw_deck
        self.discard_pile = discard_pile
        self.current_player = current_player
        # self.phase = phase
        self.round_number = round_number
        self.rng = rng
