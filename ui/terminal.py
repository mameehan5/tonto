from env.environment import CardGameEnv
from core.cards import card_str
from core.engine import legal_actions, print_legal_actions, choose_action, apply_action
from core.phases import TurnPhase

def run_terminal_game(env: CardGameEnv):
    print(80*"=")
    print("Game begins!")
    print(80*"=")

    for _ in range(env.config.num_rounds):
        print(f'\nRound {env.state.round_number+1} ----------------')

        for _ in range(env.config.num_players):

            state = env.state
            player = state.players[state.current_player]

            # Skip player turn if all cards are flipped
            if player.all_face_up():
                print(f"\nPlayer {state.current_player} | Phase: all cards flipped")
                continue

            # Play hand -------------------------------------------------------
            print(f"\nPlayer {state.current_player} | Phase: DRAW")

            num_discard = min(len(state.discard_pile), player.num_cards)
            assert num_discard >= 1
            top_discards = state.discard_pile[-num_discard:]
            print("Top discard pile:", " ".join(card_str(c) for c in top_discards))

            if env.pending_draw:
                print("Drawn card(s):", " ".join(card_str(c) for c in env.pending_draw))

            print("Your cards:")
            for i, c in enumerate(player.cards):
                visible = player.is_face_up(i) or player.is_privately_seen(i)
                display = card_str(c) if visible else "🂠"
                flipped_mark = " (flipped)" if player.is_face_up(i) else ""
                print(f"{i}: {display}{flipped_mark}")

            # actions = legal_actions(env)
            # if not actions:
            #     print("No legal actions. Round might be over.")
            #     break

            print("\nActions:")
            actions = legal_actions(env, TurnPhase.DRAW)
            print_legal_actions(actions)
            choice = choose_action(actions)
            apply_action(env, choice)
