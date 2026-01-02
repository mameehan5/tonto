import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env as _check_env

register(id="TontoTestEnv", entry_point="env.environment:TontoEnv")
env = gym.make("TontoTestEnv", num_players=4, num_decks=2, num_rounds=4, num_cards=8)

if hasattr(env, "action_space") and _check_env is not None:
    # run checks against the raw environment to avoid wrapper interference
    try:
        raw = env.unwrapped
    except Exception:
        raw = env
    # ensure a minimal spec so the checker can test render modes
    if getattr(raw, "spec", None) is None:
        from types import SimpleNamespace
        raw.spec = SimpleNamespace(id="Tonto-v0")
    _check_env(raw)
