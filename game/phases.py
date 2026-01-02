# from enum import Enum

# class GamePhase(Enum):
#     DRAW = "draw"
#     DISCARD = "discard"
#     FLIP = "flip"
#     DECLARE_JOKER = "declare_joker"
#     END_TURN = "end_turn"

from enum import IntEnum

class Phase(IntEnum):
    DRAW = 0
    DISCARD = 1
    FLIP = 2
    DECLARE_JOKER = 3
    END = 3