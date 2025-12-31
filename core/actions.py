from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List

from itertools import combinations

#This is the index of each action in the terminal
class ActionType(Enum):
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
    game_state = env.game_state
    round_state = env.round_state
    player = env.player_state[game_state.current_player]

    actions = [Action(ActionType.DRAW_DECK)]
    max_discard = min(round_state.num_discard_pile, player.num_face_down)
    actions.extend([Action(ActionType.PICKUP_DISCARD, i) for i in range(1, max_discard + 1)])
    return actions

def legal_actions_discard(env, draw_action) -> List[Action]:
    player = env.player_state[env.game_state.current_player]

    # Get number of cards to discard
    num_discard = 1
    if draw_action.type == ActionType.PICKUP_DISCARD:
        num_discard = draw_action.data

    # Gather possible discard actions from own hand
    indices = [i for i, ic in enumerate(player.face_up) if not ic]
    assert len(indices) > 0
    actions = [
        Action(ActionType.DISCARD_OWN, list(combo)) for combo in combinations(indices, num_discard)
    ]

    # If drawn from deck, can discard drawn card directly
    if draw_action.type == ActionType.DRAW_DECK:
        actions.append(Action(ActionType.DISCARD_DRAWN, [0]))

    return actions

def legal_actions_flip(env, draw_action, discard_action) -> List[Action]:
    player = env.player_state[env.game_state.current_player]

    if draw_action.type == ActionType.PICKUP_DISCARD:
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


def apply_draw_action(env, action):
    round_state = env.round_state

    if action.type == ActionType.DRAW_DECK:
        return [round_state.draw_deck.pop()]
    elif action.type == ActionType.PICKUP_DISCARD:
        return [round_state.discard_pile.pop() for _ in range(action.data)]
    else:
        raise ValueError("Action is not a valid draw action")
    
def apply_discard_action(env, drawn_cards, action):
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
