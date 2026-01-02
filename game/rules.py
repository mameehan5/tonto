from typing import List
from game.cards import Card
from game.state import GameState
from game.phases import GamePhase

# =============================================================================
# Core Actions 
# =============================================================================

def draw_from_deck(state: GameState):
    assert state.phase == GamePhase.DRAW
    card = state.deck.pop()
    state.current_hand.append(card)
    state.phase = GamePhase.DISCARD


def pickup_discard(state: GameState, index: int):
    assert state.phase == GamePhase.DRAW
    card = state.discard_pile.pop()
    state.current_hand.append(card)
    state.phase = GamePhase.DISCARD


def discard_card(state: GameState, index: int):
    assert state.phase == GamePhase.DISCARD
    card = state.current_hand.pop(index)
    state.discard_pile.append(card)


def end_discard_phase(state: GameState):
    assert state.phase == GamePhase.DISCARD
    state.phase = GamePhase.FLIP


def flip_card(state: GameState, index: int):
    assert state.phase == GamePhase.FLIP
    card = state.current_hand[index]
    card.face_up = True

    if card.is_joker and card.declared_rank is None:
        state.pending_joker_index = index
        state.phase = GamePhase.DECLARE_JOKER


def declare_joker(state: GameState, rank: int):
    assert state.phase == GamePhase.DECLARE_JOKER
    idx = state.pending_joker_index
    card = state.current_hand[idx]

    card.declared_rank = rank
    state.pending_joker_index = None
    state.phase = GamePhase.FLIP


def end_turn(state: GameState):
    state.advance_player()


# ---------- Scoring ----------

def hand_score(hand: List[Card]) -> int:
    """
    Computes score for a hand.
    Face-down or undeclared joker cards should not occur at scoring time.
    """
    score = 0
    for card in hand:
        r = card.effective_rank()
        if r is None:
            raise ValueError("Unresolved card at scoring time")
        score += r
    return score