from env.environment import CardGameEnv
from core.cards import card_str
from core.engine import legal_actions, apply_action

def run_terminal_game(env: CardGameEnv):
    state = env.state
    player = state.players[state.current_player]

    print(f"\n=== Round {state.round_number + 1} begins ===")

    while True:
        state = env.state
        player = state.players[state.current_player]

        print(f"\nPlayer {state.current_player} | Phase: {state.phase.name}")

        hand_size = len(player.cards)
        top_discard = state.discard_pile[-hand_size:] if state.discard_pile else []
        print("Top discard pile:", " ".join(card_str(c) for c in top_discard))

        if env.pending_draw:
            print("Drawn card(s):", " ".join(card_str(c) for c in env.pending_draw))

        print("Your cards:")
        for i, c in enumerate(player.cards):
            visible = player.is_face_up(i) or player.is_privately_seen(i)
            display = card_str(c) if visible else "🂠"
            flipped_mark = " (flipped)" if i in player.face_up else ""
            print(f"{i}: {display}{flipped_mark}")

        actions = legal_actions(env)
        if not actions:
            print("No legal actions. Round might be over.")
            break

        print("\nLegal actions:")
        for idx, act in enumerate(actions):
            desc = act.type.name
            if act.data is not None:
                desc += f" ({act.data})"
            print(f"{idx}: {desc}")

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
        apply_action(env, choice)
