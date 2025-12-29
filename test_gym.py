from core.config import GameConfig

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env as _check_env

register(id="TontoCardGameTest", entry_point="core.environment:CardGameEnv")
cfg = GameConfig(num_players=4, num_decks=2, cards_per_player=8, num_rounds=4)
env = gym.make("TontoCardGameTest", config=cfg)

if hasattr(env, "action_space") and _check_env is not None:
    # run checks against the raw environment to avoid wrapper interference
    try:
        raw = env.unwrapped
    except Exception:
        raw = env
    # ensure a minimal spec so the checker can test render modes
    if getattr(raw, "spec", None) is None:
        from types import SimpleNamespace
        raw.spec = SimpleNamespace(id="TontoCardGame-v0")
    _check_env(raw)
