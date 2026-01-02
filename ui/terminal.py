import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.environment import CardGameEnv
from core.config import GameConfig
from core.actions import legal_actions_draw, legal_actions_discard, legal_actions_flip
from core.actions import view_drawn_cards




def print_scores(env: CardGameEnv):
    print("\nScores after round", env.game_state.round_number)
    for i, player in enumerate(env.player_state):
        print(f"Player {i}: Individual scores: {player.individual_scores} | Total score: {player.score}")

def print_cards(player):
    for i, c in enumerate(player.cards):
        visible = player.is_face_up(i) or player.is_privately_seen(i)
        display = c.card_str if visible else "🂠"
        flipped_mark = " (flipped)" if player.is_face_up(i) else ""
        print(f"{i}: {display}{flipped_mark}")

def print_legal_actions(actions):
    for idx, act in enumerate(actions):
        desc = act.type.name
        if act.data is not None:
            desc += f" ({act.data})"
        print(f"{idx}: {desc}")

def choose_action(actions):
    # Potential quick returns with user input
    if len(actions) == 0:
        return None
    elif len(actions) == 1:
        return actions[0]
    
    # Get user input
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

def play_turn(env: CardGameEnv):
    game_state = env.game_state
    round_state = env.round_state
    player = env.player_state[game_state.current_player]

    # Play hand -------------------------------------------------------
    print(f"\nPlayer {player.player_number}")

    # Display discard pile
    num_discard = min(len(round_state.discard_pile), player.num_cards)
    assert num_discard >= 1
    top_discards = round_state.discard_pile[-num_discard:]
    print("\nTop discard pile:", " ".join(c.card_str for c in top_discards))

    # Display player's cards
    print("Your cards:")
    print_cards(player)

    # Draw
    print("\nDraw options:")
    actions = legal_actions_draw(env)
    print_legal_actions(actions)
    draw_action = choose_action(actions)
    drawn_cards = view_drawn_cards(env, draw_action)
    print("\nDrawn card(s):", " ".join(c.card_str for c in drawn_cards))

    # Discard
    actions = legal_actions_discard(env, draw_action)
    print("\nDiscard options:")
    print_legal_actions(actions)
    discard_action = choose_action(actions)

    # Flip
    actions = legal_actions_flip(env, draw_action, discard_action)
    print("\nFlip options:")
    print_legal_actions(actions)
    flip_action = choose_action(actions)

    # Map actions to int for environment step
    action_int = env.encode_actions((draw_action, discard_action, flip_action))

    # Execute the composite action via the environment step (pass previewed drawn_cards)
    # obs, reward, done, err, info = env.step(action_int)
    env.apply_action(action_int)
    env.advance_turn()

    # Display hand state from env for the acting player
    print("\nHand at end of turn:")
    print_cards(player)

def run_terminal_game(env: CardGameEnv):
    print(80*"=")
    print("Game begins!")
    print(80*"=")

    while not env.game_state.complete:
        print(f'\nRound {env.game_state.round_number+1} ===============================')

        env.initialize_round()

        while not env.round_state.complete:
            play_turn(env)

        # Advance round
        env.advance_round()
        print_scores(env)

    print("\n" + 80*"=")
    print("Game over! All rounds completed.")
    print(80*"=")

if __name__ == '__main__':
    config = GameConfig(num_players=2, num_decks=1, num_rounds=3, num_cards=2)
    env = CardGameEnv(config, seed=42)
    run_terminal_game(env)
