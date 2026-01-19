import gymnasium as gym
from gymnasium.envs.registration import register

import numpy as np
from game.actions import decode_action

# Create our training environment
register(id="TontoTestEnv", entry_point="env.environment:TontoEnv")
env = gym.make("TontoTestEnv", num_players=4, num_decks=4, num_rounds=1, num_cards=12)

# Play over several games for testing
for i in range(100):
    print(f'Game {i+1}')

    # Reset environment to start a new episode
    observation, info = env.reset()

    # Iterate until round is over
    episode_over = False
    total_reward = 0
    idx = 1
    while not episode_over:
        # print(f"Play {idx}")

        # Identify legal action
        mask = np.array(observation['actions_mask'], dtype=np.int8)
        action = env.action_space.sample(mask=mask)

        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)

        # Collect information
        total_reward += reward
        episode_over = terminated

        if truncated:
            raise RuntimeError('Game terminated early')

        idx += 1

print(f"Finished all games!")
env.close()