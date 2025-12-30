from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from itertools import combinations

class ActionType(Enum):
    PASS = -1
    DRAW_DECK = 0
    PICKUP_DISCARD = 1
    DISCARD_OWN = 2
    DISCARD_DRAWN = 3
    FLIP = 4

@dataclass
class Action:
    type: ActionType
    data: Optional[int] = None

def legal_actions_draw(env) -> List[Action]:
    player = env.player_state[env.game_state.current_player]
    if player.is_face_up_all():
        return [Action(ActionType.PASS)]
    else:
        return [Action(ActionType.DRAW_DECK)] + [
            Action(ActionType.PICKUP_DISCARD, i) for i in range(1, env.max_discard() + 1)
        ]

def legal_actions_discard(env, draw_action) -> List[Action]:
    player = env.player_state[env.game_state.current_player]

    # No available actions allowed
    if draw_action.type == ActionType.PASS:
        return []

    # Get number of cards to discard
    num_discard = 1
    if draw_action.type == ActionType.PICKUP_DISCARD:
        num_discard = draw_action.data

    # Gather possible discard actions from own hand
    actions = []
    for combo in combinations(np.arange(player.num_cards), num_discard):
        if all(player.is_face_down(i) for i in combo):
            actions.append(Action(ActionType.DISCARD_OWN, list(combo)))

    # If drawn from deck, can discard drawn card directly
    if draw_action.type == ActionType.DRAW_DECK:
        actions.append(Action(ActionType.DISCARD_DRAWN, [0]))

    return actions

def legal_actions_flip(env, draw_action, discard_action) -> List[Action]:
    player = env.player_state[env.game_state.current_player]

    if draw_action.type == ActionType.PASS:
        return []

    elif draw_action.type == ActionType.PICKUP_DISCARD:
        return [Action(ActionType.FLIP, discard_action.data)]

    elif discard_action.type == ActionType.DISCARD_OWN:
        return [Action(ActionType.FLIP, discard_action.data)]

    elif discard_action.type == ActionType.DISCARD_DRAWN:
        return [
            Action(ActionType.FLIP, [ic]) for ic,up in enumerate(player.face_up) if not up
        ]
    
    raise ValueError("Invalid combination of draw and discard actions")

def view_drawn_cards(env, action: Action):
    round_state = env.round_state

    if action.type == ActionType.DRAW_DECK:
        return [round_state.draw_deck[-1]]
    elif action.type == ActionType.PICKUP_DISCARD:
        return [round_state.discard_pile[-i] for i in range(action.data, 0, -1)]
    else:
        raise ValueError("Action is not a draw action")

def apply_draw_action(env, action: Action) -> List:
    round_state = env.round_state

    if action.type == ActionType.PASS:
        return []
    if action.type == ActionType.DRAW_DECK:
        return [round_state.draw_deck.pop()]
    elif action.type == ActionType.PICKUP_DISCARD:
        return [round_state.discard_pile.pop() for _ in range(action.data)]
    else:
        raise ValueError("Action is not a valid draw action")

def apply_discard_action(env, drawn_cards: List, action: Action):
    round_state = env.round_state
    player = env.player_state[env.game_state.current_player]

    if action.type == ActionType.DISCARD_DRAWN:
        discarded_cards = drawn_cards
    elif action.type == ActionType.DISCARD_OWN:
        # If we are discarding own cards, swap in drawn cards and only flip
        # action allowed is to flip those indices
        discarded_cards = [player.cards[index] for index in action.data]
        for i, index in enumerate(action.data):
            player.cards[index] = drawn_cards[i]
    else:
        raise ValueError("Action is not a valid discard action")
    round_state.discard_pile.extend(discarded_cards)


def apply_flip_action(env, action: Action):
    player = env.player_state[env.game_state.current_player]

    assert action.type == ActionType.FLIP

    if type(action.data) is int:
        player.flip(action.data)
    elif type(action.data) is list:
        for idx in action.data:
            player.flip(idx)
    else:
        raise ValueError("Invalid flip action data")
