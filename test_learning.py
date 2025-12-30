import gymnasium as gym
from gymnasium.envs.registration import register

from core.config import GameConfig

# Create our training environment
register(id="TontoTestEnv", entry_point="core.environment:CardGameEnv")
cfg = GameConfig(num_players=8, num_decks=3, cards_per_player=8, num_rounds=1)
env = gym.make("TontoTestEnv", config=cfg)

# Reset environment to start a new episode
observation, info = env.reset(seed=42)

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0
idx = 1
while not episode_over:
    # Identify legal action
    mask = observation['legal_actions_mask']
    action = env.action_space.sample(mask=mask)

    # Take the action
    observation, reward, terminated, truncated, info = env.step(action)

    # Collect information
    total_reward += reward
    episode_over = terminated

    print(f"\nObservation {idx}: {observation}")

    idx += 1


print(f"Episode finished! Total reward: {total_reward}")
env.close()