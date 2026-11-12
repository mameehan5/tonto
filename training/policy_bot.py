"""Use a trained MaskablePPO model as a bot (same interface as game/bots.py)."""
from sb3_contrib.ppo_mask import MaskablePPO

from env.environment import make_obs


class PolicyBot:
    def __init__(self, path, deterministic=True):
        self.model = MaskablePPO.load(path, device="cpu")
        self.deterministic = deterministic

    def act(self, game, pid):
        action, _ = self.model.predict(make_obs(game), action_masks=game.legal_mask(),
                                       deterministic=self.deterministic)
        return int(action)
