from enum import IntEnum

class Phase(IntEnum):
    DRAW = 0
    DISCARD = 1
    FLIP = 2
    DECLARE_JOKER = 3
    END = 4