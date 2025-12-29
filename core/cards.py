from typing import List, Tuple

RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["S","H","D","C"]
# SUITS_DISPLAY = ["♠", "♥", "♦", "♣"]

Card = Tuple[str, str, int]  # (rank, suit, deck)

# TODO: make Card a dataclass?
# TODO: add Joker handling?

def create_standard_deck(num_decks: int = 1) -> List[Card]:
    deck: List[Card] = []
    for deck_ind in range(num_decks):
        for suit in SUITS:
            for rank in RANKS:
                deck.append((rank, suit, deck_ind))
        # deck.append(("JOKER",""))
        # deck.append(("JOKER",""))
    return deck

def card_str(card: Card) -> str:
    rank, suit, _ = card
    if rank == "JOKER":
        return "🃏"
    return f"{rank}{suit}"

def card_value(card: Card) -> int:
    rank, _, _ = card
    if rank == "A":
        return 1
    elif rank in ["J","Q"]:
        return 10
    elif rank == "K":
        return 0
    elif rank == "JOKER":
        return 0
    return int(rank)

def card_rank(card: Card) -> str:
    rank, _, _ = card
    return rank

def card_rank_idx(card: Card) -> int:
    rank, _, _ = card
    return RANKS.index(rank)

def encode_card(card) -> int:
    rank, suit, deck = card
    irank = RANKS.index(rank)
    isuit = SUITS.index(suit)
    return deck * 52 + isuit * 13 + irank

def decode_card(card_int: int) -> Card:
    deck = card_int // 52
    rem = card_int % 52
    isuit = rem // 13
    irank = rem % 13
    rank = RANKS[irank]
    suit = SUITS[isuit]
    return (rank, suit, deck)
