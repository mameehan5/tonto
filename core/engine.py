from typing import List
from core.actions import Action, ActionType
from env.state import GameState, PlayerState
from core.phases import TurnPhase

def legal_actions(env, phase) -> List[Action]:
    state = env.state
    player = state.players[state.current_player]
    actions: List[Action] = []

    # DRAW
    if phase == TurnPhase.DRAW:
        actions.append(Action(ActionType.DRAW_DECK))
        max_discard = min(len(state.discard_pile), player.count_not_face_up())
        actions.append(Action(ActionType.PICKUP_DISCARD, len(state.discard_pile)))
        return actions

    # FLIP
    if state.phase == TurnPhase.FLIP:
        if env.pending_draw:
            for i in range(len(env.pending_draw)):
                actions.append(Action(ActionType.FLIP_DRAWN, i))
        else:
            for i in range(len(player.cards)):
                if i not in player.face_up:
                    actions.append(Action(ActionType.FLIP_OWN, i))
        return actions

    # DISCARD
    if state.phase == TurnPhase.DISCARD:
        pending = len(env.pending_draw) if env.pending_draw else 1
        hand_indices = [i for i in range(len(player.cards)) if i not in player.face_up]
        if len(hand_indices) >= pending:
            from itertools import combinations
            for combo in combinations(hand_indices, pending):
                actions.append(Action(ActionType.DISCARD, list(combo)))
        return actions

    return []

def print_legal_actions(actions: List[Action]):
    for idx, act in enumerate(actions):
        desc = act.type.name
        if act.data is not None:
            desc += f" ({act.data})"
        print(f"{idx}: {desc}")

def choose_action(actions: List[Action]) -> Action:
    choice = None
    while choice is None:
        try:
            idx = int(input("Choose action index: "))
            if 0 <= idx < len(actions):
                choice = actions[idx]
            else:
                print("Invalid index")
        except ValueError:
            print("Enter a number")
    return choice

def apply_action(env, action: Action):
    state = env.state
    player = state.players[state.current_player]

    # DRAW
    if action.type == ActionType.DRAW_DECK:
        card = state.draw_deck.pop()
        env.pending_draw = [card]
        env.pending_source = "deck"
        env.pending_count = 1
        state.phase = TurnPhase.FLIP
        return

    if action.type == ActionType.PICKUP_DISCARD:
        n = action.data
        env.pending_draw = state.discard_pile[-n:]
        state.discard_pile = state.discard_pile[:-n]
        env.pending_source = "discard"
        env.pending_count = n
        state.phase = TurnPhase.FLIP
        return

    # FLIP
    if action.type == ActionType.FLIP_DRAWN:
        env.just_flipped = list(range(len(env.pending_draw)))
        state.phase = TurnPhase.DISCARD
        return

    if action.type == ActionType.FLIP_OWN:
        idx = action.data
        player.flip(idx)
        env.just_flipped = []
        state.phase = TurnPhase.DISCARD
        return

    # DISCARD
    if action.type == ActionType.DISCARD:
        indices = sorted(action.data, reverse=True)
        discarded = [player.cards.pop(idx) for idx in indices]
        state.discard_pile.extend(discarded)
        if env.pending_source == "deck" and env.pending_draw:
            player.cards.append(env.pending_draw[0])
        env.pending_draw = None
        env.pending_source = None
        env.pending_count = 0
        env.just_flipped = []
        env.advance_turn()
        return
