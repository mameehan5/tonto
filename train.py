from tqdm import tqdm 

import gymnasium as gym

import numpy as np

from core.wrappers import obs_to_key

from core.config import GameConfig
from core.environment import CardGameEnv
from training.agent import TontoAgent

env_name = "TontoEnv"
num_eval_episodes = 1

# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
num_episodes = 10000       # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
final_epsilon = 0.1         # Always keep some exploration

# TODO: make cards change per game
# TODO: rename cardgame to tonto

# Create environment and agent
cfg = GameConfig(num_players=2, num_decks=1, num_rounds=1, num_cards=2)
env = CardGameEnv(config=cfg)
agent = TontoAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=(start_epsilon - final_epsilon) / num_episodes,
    final_epsilon=final_epsilon,
)

print(f"Starting training for {num_episodes} episodes...")

for episode in tqdm(range(num_episodes)):
    # Start a new hand
    obs, info = env.reset()
    obs = obs_to_key(obs, env.observation_space)

    # Play one complete round
    while not env.round_state.complete:

        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(obs, info)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = obs_to_key(next_obs, env.observation_space)

        # Learn from this experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Move to next state
        obs = next_obs
    
    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()

assert len(agent.total_scores) == num_episodes
print(agent.total_scores)

from matplotlib import pyplot as plt

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 100
# fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
fig, (ax1,ax2) = plt.subplots(1, 2)

# Episode performance
ax2.set_title("Machine Performance")
total_scores_average = get_moving_avgs(
    agent.total_scores,
    rolling_length,
    "same"
)
ax1.plot(np.arange(num_episodes)+1, total_scores_average)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Player Score")


# Training error (how much we're still learning)
ax2.set_title("Training Error")
training_error = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
ax2.plot(range(len(training_error)), training_error)
ax2.set_xlabel("Step")
ax2.set_ylabel("Temporal Difference Error")

plt.tight_layout()

fig.set_size_inches(5.25,2.0,forward=True)
plt.savefig('training.png', dpi=300)
