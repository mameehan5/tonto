from enum import Enum

class GamePhase(Enum):
    DRAW = "draw"
    DISCARD = "discard"
    FLIP = "flip"
    DECLARE_JOKER = "declare_joker"
    END_TURN = "end_turn"