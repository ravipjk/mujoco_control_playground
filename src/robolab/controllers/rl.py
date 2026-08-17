import numpy as np

from robolab.core.controller import Controller


class PolicyController(Controller):
    """Adapts a trained Stable-Baselines3 policy to the Controller interface,
    so a learned policy plugs into rollout exactly like PD/LQR/CPG."""

    def __init__(self, model, deterministic=True):
        self.model = model
        self.deterministic = deterministic

    def act(self, observation):
        # SB3 policies expose .predict(obs, deterministic=...) -> (action, state).
        # Return just the action.
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return action