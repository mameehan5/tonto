from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class ActionType(Enum):
    DRAW_DECK = 0
    PICKUP_DISCARD = 1
    DISCARD_OWN = 2
    DISCARD_DRAWN = 3
    FLIP = 4
    # CALL_JOKER = 6

@dataclass
class Action:
    type: ActionType
    data: Optional[int] = None  # index or count
