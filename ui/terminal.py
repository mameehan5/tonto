import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

import numpy as np
import time

import argparse

from env.environment import TontoEnv

from game.actions import compute_action_mask, decode_action

from sb3_contrib.ppo_mask import MaskablePPO

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

def play_turn(env):
    tonto_env = env.unwrapped
    game_state = tonto_env.game_state
    round_state = tonto_env.round_state
    player = tonto_env.player_state[game_state.current_player]

    # Play hand -------------------------------------------------------
    print(f"\nPlayer {player.player_number}")
    
    # Draw
    print("\nDraw options:")
    actions = print_legal_actions(tonto_env)
    draw_action = choose_action(actions)
    draw_action_str, _, _ = decode_action(draw_action)
    if draw_action_str == 'end_turn':
        obs, _, _, _, _ = env.step(draw_action)
        return obs
    drawn_cards = view_drawn_cards(tonto_env, draw_action)
    print("\nDrawn card(s):", " ".join(c.card_str for c in drawn_cards))
    env.step(draw_action)

    # Discard
    print("\nDiscard options:")
    while len(player.holding_cards) > 0:
        actions = print_legal_actions(tonto_env)
        discard_action = choose_action(actions)
        env.step(discard_action)

    # Flip
    if player.num_to_flip > 0:
        print("\nFlip options:")
        actions = print_legal_actions(tonto_env)
        flip_action = choose_action(actions)
        env.step(flip_action)
    
    # Declare joker(s)
    while np.any(player.visible_undeclared_jokers):
        print("\nDeclare joker:")
        actions = print_legal_actions(tonto_env)
        call_action = choose_action(actions)
        env.step(call_action)
    
    # End turn
    actions = print_legal_actions(tonto_env)
    end_action = choose_action(actions)
    obs, _, _, _, _ = env.step(end_action)
    return obs

def load_ppo_agent(model_path, env):
    # vec_env = DummyVecEnv([lambda: env])
    # model = MaskablePPO.load(model_path, env=vec_env)
    # vec_env = DummyVecEnv([lambda: env])
    model = MaskablePPO.load(model_path)
    return model

def ppo_choose_action(model, env, obs):
    masks = get_action_masks(env)
    assert masks is not None, "Masking NOT active in terminal env"
    action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    return int(action)

def run_terminal_game(env, human_players, ppo_model=None):
    print(80*"=")
    print("Game begins!")
    print(80*"=")

    obs, _ = env.reset()

    tonto_env = env.unwrapped

    while not tonto_env.game_state.complete:
        print(80*"-")
        print(f'\nRound {tonto_env.game_state.current_round+1}')
        print(80*"-")

        while not tonto_env.round_state.complete:
            p = tonto_env.game_state.current_player
            player = tonto_env.player_state[p]

            # print("\n" + "=" * 60)
            print(f"Player {p}'s turn ---------------------------")

            top_discards = tonto_env.round_state.discard_pile[-tonto_env.num_cards:]
            print("\nTop discard pile:", " ".join(c.card_str for c in top_discards))

            print("\nCurrent hand:")
            print_cards(player)

            if p in human_players:
                obs = play_turn(env)
            else:
                act = ppo_choose_action(ppo_model, tonto_env, obs)
                action, _, action_str = decode_action(act)
                print(f"\n🤖 PPO chose action {action_str}")
                drawn_cards = view_drawn_cards(tonto_env, act)
                print("\nDrawn card(s):", " ".join(c.card_str for c in drawn_cards))
                obs, _, _, _, _ = env.step(act)
                while action != 'end_turn':
                    time.sleep(1.0)
                    act = ppo_choose_action(ppo_model, env, obs)
                    action, _, action_str = decode_action(act)
                    print(f"\n🤖 PPO chose action {action_str}")
                    obs, _, _, _, _ = env.step(act)
                time.sleep(1.0)

            print("\nHand at end of turn:")
            print_cards(player)
            
        # Advance round
        tonto_env.advance_round()
        print_scores(tonto_env)

    print("\n" + 80*"=")
    print("Game over! All rounds completed.")
    print(80*"=")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--num_cards", type=int, default=4)
    parser.add_argument("--num_decks", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--human_players",
        type=int,
        nargs="*",
        default=[0],
        required=True,
        help="Indices of human players (e.g. 0 1)"
    )

    parser.add_argument(
        "--ppo_model",
        type=str,
        default=None,
        required=True,
        help="Path to PPO model (.zip)"
    )

    args = parser.parse_args()

    env = TontoEnv(
        num_players=args.num_players,
        num_decks=args.num_decks, 
        num_rounds=args.num_rounds, 
        num_cards=args.num_cards,
        seed=args.seed
    )
    env = ActionMasker(env, lambda e: e.action_masks())

    ppo_model = None
    if args.ppo_model is not None:
        ppo_model = load_ppo_agent(args.ppo_model, env)

    run_terminal_game(
        env,
        human_players=set(args.human_players),
        ppo_model=ppo_model
    )
