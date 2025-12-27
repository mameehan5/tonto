from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class ActionType(Enum):
    DRAW_DECK = 0
    PICKUP_DISCARD = 1
    FLIP_DRAWN = 2
    FLIP_OWN = 3
    DISCARD = 4
    CALL_JOKER = 5

@dataclass
class Action:
    type: ActionType
    data: Optional[int] = None  # index or count
