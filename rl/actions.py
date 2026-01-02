from enum import IntEnum
import numpy as np

MAX_CARDS = 20
NUM_RANKS = 13

class Action(IntEnum):
    DRAW_DECK = 0
    PICKUP_DISCARD_0 = 1
    # PICKUP_DISCARD_i = PICKUP_DISCARD_0 + i

    DISCARD_DRAWN = PICKUP_DISCARD_0 + MAX_CARDS
    DISCARD_OWN_0 = DISCARD_DRAWN + 1
    # DISCARD
    # DISCARD_i = DISCARD_0 + i

    FLIP_0 = DISCARD_OWN_0 + MAX_CARDS
    # FLIP_i = FLIP_0 + i

    DECLARE_JOKER_0 = FLIP_0 + MAX_CARDS

    END_TURN = DECLARE_JOKER_0 + 1

    NUM_POSSIBLE_ACTIONS = END_TURN + 1

def decode_action(action: int):
    if action == Action.DRAW_DECK:
        return "draw_deck", None, "Draw from deck"

    elif Action.PICKUP_DISCARD_0 <= action < Action.PICKUP_DISCARD_0 + MAX_CARDS:
        n = action - Action.PICKUP_DISCARD_0 + 1
        plural = 's' if n > 1 else ''
        return "pickup_discard", n, f"Pick up {n} card{plural} from the discard pile"

    elif action == Action.DISCARD_DRAWN:
        return "discard_drawn", None, "Discard card drawn from deck"

    elif Action.DISCARD_OWN_0 <= action < Action.DISCARD_OWN_0 + MAX_CARDS:
        c = action - Action.DISCARD_OWN_0
        return "discard_own", c, f"Discard card at index {c}"

    elif Action.FLIP_0 <= action < Action.FLIP_0 + MAX_CARDS:
        c = action - Action.FLIP_0
        return "flip", c, f"Flip card index {c}"
    
    elif Action.DECLARE_JOKER_0 <= action < Action.DECLARE_JOKER_0 + MAX_CARDS:
        r = action - Action.DECLARE_JOKER_0
        return "declare_joker", r, "Declare joker as rank {r}"

    elif action == Action.END_TURN:
        return "end_turn", None, "Turn has ended"

    raise ValueError(f"Invalid action {action}")

def compute_action_mask(env):
    mask = np.zeros(Action.NUM_POSSIBLE_ACTIONS, dtype=bool)

    player = env.player_state[env.game_state.current_player]
    phase = env.round_state.phase
    num_cards = env.game_state.num_cards

    if phase == "draw":
        mask[0] = True
        mask[Action.PICKUP_DISCARD_0 : (Action.PICKUP_DISCARD_0 + env.max_draw_from_discard)] = True

    elif phase == "discard":
        if player.action[0] == "draw_deck":
            mask[Action.DISCARD_DRAWN] = True
        mask[Action.DISCARD_OWN_0 : Action.DISCARD_OWN_0 + num_cards] = player.face_down

    elif phase == "flip":
        mask[Action.FLIP_0 : Action.FLIP_0 + num_cards] = player.face_down

    return mask