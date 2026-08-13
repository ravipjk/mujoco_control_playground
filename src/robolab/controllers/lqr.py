import numpy as np

from robolab.core.controller import Controller


class LQRController(Controller):
    """Full-state feedback: u = -K (obs - x_ref)."""

    def __init__(self, K, x_ref):
        self.K = np.asarray(K)
        self.x_ref = np.asarray(x_ref)

    def act(self, observation):
        # obs IS the state x = [cart_x, pole_angle, cart_vel, pole_rate].
        # Return -K @ (x - x_ref); the result is already shape (nu,) = (1,).
        return -self.K @ (np.asarray(observation) - self.x_ref)