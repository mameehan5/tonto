from core.environment import CardGameEnv
from core.actions import legal_actions_draw, legal_actions_discard, legal_actions_flip
from core.cards import card_str
from core.actions import get_drawn_cards, choose_action
from core.actions import apply_discard_action, apply_flip_action

def print_scores(env: CardGameEnv):
    print("\nScores after round", env.state.round_number)
    for i, player in enumerate(env.state.players):
        print(f"Player {i}: Individual scores: {player.individual_scores} | Total score: {player.score}")

def print_cards(player):
    for i, c in enumerate(player.cards):
        visible = player.is_face_up(i) or player.is_privately_seen(i)
        display = card_str(c) if visible else "🂠"
        flipped_mark = " (flipped)" if player.is_face_up(i) else ""
        print(f"{i}: {display}{flipped_mark}")

def print_legal_actions(actions):
    for idx, act in enumerate(actions):
        desc = act.type.name
        if act.data is not None:
            desc += f" ({act.data})"
        print(f"{idx}: {desc}")

def run_terminal_game(env: CardGameEnv):
    print(80*"=")
    print("Game begins!")
    print(80*"=")

    for _ in range(env.config.num_rounds):
        print(f'\nRound {env.state.round_number+1} ===============================')

        for _ in range(env.config.cards_per_player):
            for _ in range(env.config.num_players):
                state = env.state
                player = state.players[state.current_player]

                # Skip player turn if all cards are flipped
                if player.all_face_up():
                    print(f"\nPlayer {state.current_player} | Phase: all cards flipped")
                    env.advance_turn()
                    continue

                # Play hand -------------------------------------------------------
                print(f"\nPlayer {state.current_player} | Phase: DRAW")

                # Display discard pile
                num_discard = min(len(state.discard_pile), player.num_cards)
                assert num_discard >= 1
                top_discards = state.discard_pile[-num_discard:]
                print("Top discard pile:", " ".join(card_str(c) for c in top_discards))

                # Display player's cards
                print("Your cards:")
                print_cards(player)

                # Draw
                print("\nDraw options:")
                actions = legal_actions_draw(env)
                print_legal_actions(actions)
                draw_action = choose_action(actions)
                drawn_cards = get_drawn_cards(env, draw_action)
                print("\nDrawn card(s):", " ".join(card_str(c) for c in drawn_cards))

                # Discard
                print("\nDiscard options:")
                actions = legal_actions_discard(env, draw_action)
                print_legal_actions(actions)
                discard_action = choose_action(actions)
                discarded_cards = apply_discard_action(env, drawn_cards, discard_action)

                # Flip
                print("\nFlip options:")
                actions = legal_actions_flip(env, draw_action, discard_action)
                print_legal_actions(actions)
                choice = choose_action(actions)
                apply_flip_action(env, choice)

                # UPDATE 
                if len(discarded_cards) > 1:
                    # In the future, allow next player to order cards
                    pass

                # Display hand state
                print("Hand at end of turn:")
                print_cards(player)
                    
                # Advance turn
                env.advance_turn(discarded_cards)

        # Advance round
        env.advance_round()
        print_scores(env)

