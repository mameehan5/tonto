from dataclasses import dataclass
from typing import List

SUITS = list(range(4))  
RANKS = list(range(13))

RANK_STRS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUIT_STRS = ["♠","♥","♦","♣"]

@dataclass(frozen=True)
class Card:
    """Immutable card representation used throughout the codebase."""
    id: int           # unique card id
    rank: int         # 0..12
    suit: int         # 0..3
    deck: int         # deck index for multi-deck games

    def __repr__(self) -> str:
        return f"Card(id={self.id}, rank={self.rank}, suit={SUITS[self.suit]}, deck={self.deck})"

    def __post_init__(self):
        object.__setattr__(self, 'rank_str', RANK_STRS[self.rank])
        object.__setattr__(self, 'suit_str', SUIT_STRS[self.suit])
        object.__setattr__(self, 'card_str', f"{RANK_STRS[self.rank]}{SUIT_STRS[self.suit]}")
        object.__setattr__(self, 'card_value', card_value(self))

# TODO: add Joker handling

def create_standard_deck(num_decks: int = 1) -> List[Card]:
    deck: List[Card] = []
    id = 0
    for d in range(num_decks):
        for s in range(4):
            for r in RANKS:
                deck.append(Card(id=id, rank=r, suit=s, deck=d))
                id += 1
    return deck

def card_value(card: Card) -> int:
    if card.rank == 0: # Ace
        return 1
    elif card.rank == 4:  # 5
        return -5
    elif card.rank in [10,11]:  # J or Q
        return 10
    elif card.rank == 12: # King
        return 0
    return int(card.rank + 1)
