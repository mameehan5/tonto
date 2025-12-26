from enum import Enum, auto

class TurnPhase(Enum):
    DRAW = auto()
    FLIP = auto()
    DISCARD = auto()
