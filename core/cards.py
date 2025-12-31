from typing import List, Tuple

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]

Card = Tuple[str, str]  # (rank, suit)
#I need to draw a card based on this "card" data structure

def create_standard_deck(num_decks: int = 1) -> List[Card]:
    deck: List[Card] = []
    for _ in range(num_decks):
        for suit in SUITS:
            for rank in RANKS:
                deck.append((rank, suit))
        # deck.append(("JOKER",""))
        # deck.append(("JOKER",""))
    #I want to use create 52 instances of card_display here as well. Or maybe I can just draw it based on the tuple data
    return deck

def card_str(card: Card) -> str:
    rank, suit = card
    if rank == "JOKER":
        return "🃏"
    return f"{rank}{suit}"

def card_value(card: Card) -> int:
    rank, _ = card
    if rank == "A":
        return 1
    if rank in ["J","Q"]:
        return 10
    if rank == "K":
        return 0
    if rank == "JOKER":
        return 0
    return int(rank)
