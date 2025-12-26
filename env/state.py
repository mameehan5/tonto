from typing import List, Set
from core.cards import Card
from core.phases import TurnPhase

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

class GameState:
    def __init__(self, players: List[PlayerState], draw_deck: List[Card], discard_pile: List[Card],
                 current_player: int, phase: TurnPhase, round_number: int, rng):
        self.players = players
        self.draw_deck = draw_deck
        self.discard_pile = discard_pile
        self.current_player = current_player
        self.phase = phase
        self.round_number = round_number
        self.rng = rng
