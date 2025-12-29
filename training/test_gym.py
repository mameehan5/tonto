from core.config import GameConfig
from core.environment import CardGameEnv
import numpy as np

cfg = GameConfig(num_players=2, num_decks=1, cards_per_player=4, num_rounds=1)
env = CardGameEnv(cfg)

obs = env.reset()
assert isinstance(obs, dict), "reset() must return a dict-like observation"

# basic observation_space containment check
if not env.observation_space.contains(obs):
    raise AssertionError("Observation does not match observation_space")

# try full gym/gymnasium checker if available and action_space is defined
try:
    from gymnasium.utils.env_checker import check_env as _check_env  # gymnasium
except Exception:
    try:
        from gym.utils.env_checker import check_env as _check_env  # gym
    except Exception:
        _check_env = None

if hasattr(env, "action_space") and _check_env is not None:
    _check_env(env)  # may raise on incompatibilities

print("Environment passed basic observation checks.")