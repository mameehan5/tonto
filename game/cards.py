from dataclasses import dataclass
from typing import List

RANK_STRS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUIT_STRS = ["♠","♥","♦","♣",'J'] # 5th is joker

RANKS = list(range(len(RANK_STRS)))
SUITS = list(range(len(SUIT_STRS)))  

@dataclass
class Card:
    """Immutable card representation used throughout the codebase."""
    id: int           # unique card id
    rank: int         # 0..12
    suit: int         # 0..3
    face_up: bool = False

    def __repr__(self) -> str:
        return f"Card(id={self.id}, rank={self.rank}, suit={SUITS[self.suit]})"

    @property
    def rank_str(self):
        if self.rank == 13:
            return '?'
        return RANK_STRS[self.rank]
    
    @property
    def suit_str(self):
        return SUIT_STRS[self.suit]
    
    @property
    def card_str(self):
        return f"{self.rank_str}{self.suit_str}"

    @property
    def card_value(self):
        if self.rank == 0: # Ace
            return 1
        elif self.rank == 4:  # 5
            return -5
        elif self.rank in [10,11]:  # J or Q
            return 10
        elif self.rank == 12: # King
            return 0
        elif 0 < self.rank < 3 or 4 < self.rank < 10:
            return int(self.rank + 1)
        assert self.suit == 4 and self.face_up is False
        return None

def create_standard_deck(num_decks: int = 1) -> List[Card]:
    deck: List[Card] = []
    id = 0
    for _ in range(num_decks):
        for s in range(4):
            for r in RANKS:
                deck.append(Card(id=id, rank=r, suit=s))
                id += 1
        for _ in range(2):
            deck.append(Card(id=id, rank=13, suit=4))
            id += 1
    return deck
