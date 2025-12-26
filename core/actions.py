from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class ActionType(Enum):
    DRAW_DECK = auto()
    PICKUP_DISCARD = auto()
    FLIP_DRAWN = auto()
    FLIP_OWN = auto()
    DISCARD = auto()
    CALL_JOKER = auto()

@dataclass
class Action:
    type: ActionType
    data: Optional[int] = None  # index or count
