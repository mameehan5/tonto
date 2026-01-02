import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np

import argparse

from env.environment import TontoEnv

from game.actions import compute_action_mask, decode_action

def print_scores(env: TontoEnv):
    print("\nScores after round", env.game_state.current_round)
    for i, player in enumerate(env.player_state):
        print(f"Player {i}: Individual scores: {player.individual_scores} | Total score: {player.total_score}")

def print_cards(player):
    for i, c in enumerate(player.cards):
        visible = player.is_face_up(i) or player.is_privately_seen(i)
        display = c.card_str if visible else "🂠"
        flipped_mark = " (flipped)" if player.is_face_up(i) else ""
        print(f"{i}: {display}{flipped_mark}")

def print_legal_actions(env):
    action_mask = compute_action_mask(env)
    legal_actions = []
    idx = 0
    for action, legal in enumerate(action_mask):
        if legal:
            legal_actions.append(action)
            _, _, text = decode_action(action)
            print(f"{idx}: {text}")
            idx += 1
    return legal_actions

def view_drawn_cards(env, action):
    action_str, n, _ = decode_action(action)
    round_state = env.round_state

    if action_str == "end_turn":
        return []
    if action_str == "draw_deck":
        return [round_state.draw_deck[-1]]
    elif action_str == "pickup_discard":
        return [round_state.discard_pile[-i] for i in range(n, 0, -1)]
    else:
        raise ValueError("Action is not a draw action")

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
            idx = int(input("\nChoose action index: "))
            if 0 <= idx < len(actions):
                choice = actions[idx]
            else:
                print("Invalid index")
        except ValueError:
            print("Enter a number")
    return choice

def play_turn(env: TontoEnv):
    game_state = env.game_state
    round_state = env.round_state
    player = env.player_state[game_state.current_player]

    # Play hand -------------------------------------------------------
    print(f"\nPlayer {player.player_number}")

    # Display discard pile
    top_discards = round_state.discard_pile[-env.num_cards:]
    print("\nTop discard pile:", " ".join(c.card_str for c in top_discards))

    # Display player's cards
    print("\nYour cards:")
    print_cards(player)

    # Draw
    print("\nDraw options:")
    actions = print_legal_actions(env)
    draw_action = choose_action(actions)
    drawn_cards = view_drawn_cards(env, draw_action)
    print("\nDrawn card(s):", " ".join(c.card_str for c in drawn_cards))
    env.step(draw_action)

    # Discard
    print("\nDiscard options:")
    while len(player.holding_cards) > 0:
        actions = print_legal_actions(env)
        discard_action = choose_action(actions)
        env.step(discard_action)

    # Flip
    if player.num_to_flip > 0:
        print("\nFlip options:")
        actions = print_legal_actions(env)
        flip_action = choose_action(actions)
        env.step(flip_action)
    
    # Declare joker(s)
    while np.any(player.visible_undeclared_jokers):
        print("\nDeclare joker:")
        actions = print_legal_actions(env)
        call_action = choose_action(actions)
        env.step(call_action)
    
    # Advance turn
    env.advance_turn()

    # Display hand state from env for the acting player
    print("\nHand at end of turn:")
    print_cards(player)

# def run_terminal_game(env, controllers, ppo_model=None, log_level="basic"):

    # obs, _ = env.reset()
    # done = False

    # while not done:
    #     p = env.game_state.current_player

    #     if log_level != "silent":
    #         print("\n" + "=" * 60)
    #         print(f"Player {p}'s turn")

    #     if controllers[p] == "human":
    #         print_player_view(env, p)
    #         action = get_human_action(env)
    #     else:
    #         action = ppo_choose_action(ppo_model, env, obs)
    #         if log_level == "debug":
    #             print(f"🤖 PPO chose action {action}")

    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    # print_scores(env)

def run_terminal_game(env: TontoEnv):
    print(80*"=")
    print("Game begins!")
    print(80*"=")

    while not env.game_state.complete:
        print(f'\nRound {env.game_state.current_round+1} ===============================')

        # env.initialize_round()

        while not env.round_state.complete:
            play_turn(env)

        # Advance round
        env.advance_round()
        print_scores(env)

    print("\n" + 80*"=")
    print("Game over! All rounds completed.")
    print(80*"=")

if __name__ == '__main__':
    # config = GameConfig(num_players=2, num_decks=1, num_rounds=3, num_cards=2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--num_cards", type=int, default=4)
    parser.add_argument("--num_decks", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)

    # parser.add_argument(
    #     "--human_players",
    #     type=int,
    #     nargs="*",
    #     default=[0],
    #     help="Indices of human players (e.g. 0 1)"
    # )

    # parser.add_argument(
    #     "--ppo_model",
    #     type=str,
    #     default=None,
    #     help="Path to PPO model (.zip)"
    # )

    args = parser.parse_args()

    env = TontoEnv(
        num_players=args.num_players,
        num_decks=args.num_decks, 
        num_rounds=args.num_rounds, 
        num_cards=args.num_cards,
        seed=args.seed
    )

    # ppo_model = None
    # if args.ppo_model is not None:
    #     ppo_model = load_ppo_agent(args.ppo_model, env)

    run_terminal_game(env)

    # run_terminal_game(
    #     env,
    #     human_players=set(args.human_players),
    #     ppo_model=ppo_model
    # )
