import gymnasium as gym
from gymnasium.envs.registration import register

from core.config import GameConfig

# Create our training environment
register(id="TontoTestEnv", entry_point="core.environment:CardGameEnv")
cfg = GameConfig(num_players=4, num_decks=3, num_rounds=1, num_cards=4)
env = gym.make("TontoTestEnv", config=cfg)

# Reset environment to start a new episode
observation, info = env.reset(seed=42)

# Iterate until round is over
episode_over = False
total_reward = 0
idx = 1
while not episode_over:
    print(f"Hand {idx}")

    # Identify legal action
    mask = observation['legal_actions_mask']
    action = env.action_space.sample(mask=mask)

    # Take the action
    observation, reward, terminated, truncated, info = env.step(action)

    # Collect information
    total_reward += reward
    episode_over = terminated

    if truncated:
        raise RuntimeError('Game terminated early')

    idx += 1

print(f"Round finished!")
env.close()