"""Use a trained MaskablePPO model as a bot (same interface as game/bots.py)."""
from sb3_contrib.ppo_mask import MaskablePPO

from env.environment import make_obs
from env.selfplay import encode_obs, load_config
from game.bots import GreedyBot
from game.phases import Phase


class PolicyBot:
    def __init__(self, path=None, model=None, config=None, deterministic=True):
        self.model = model or MaskablePPO.load(path, device="cpu")
        self.config = config or (load_config(path) if path else None)  # None -> legacy TontoEnv obs
        self.deterministic = deterministic
        self.greedy = GreedyBot() if self.config and self.config.get("greedy_jokers") else None

    def act(self, game, pid):
        if self.greedy and game.phase == Phase.DECLARE_JOKER:
            return self.greedy.act(game, pid)
        if self.config:
            obs = encode_obs(game, pid, self.config["obs_players"], self.config["obs_cards"])
        else:
            obs = make_obs(game)
        action, _ = self.model.predict(obs, action_masks=game.legal_mask(), deterministic=self.deterministic)
        return int(action)
