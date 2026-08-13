import numpy as np

from robolab.core.env import MujocoRobotEnv


class CartpoleEnv(MujocoRobotEnv):
    """Cartpole with a balancing task: start slightly off-upright, fail if it tips.

    obs = [cart_x, pole_angle, cart_vel, pole_rate]; action = 1-dim cart force.
    """

    def __init__(self, xml_path, init_angle=0.1, angle_limit=0.8, **kwargs):
        # init_angle: size of the random initial pole tilt (radians)
        # angle_limit: |pole_angle| beyond which we declare the pole "fallen"
        self.init_angle = init_angle
        self.angle_limit = angle_limit
        super().__init__(xml_path, **kwargs)

    def _reset_model(self):
        # Perturb the pole off upright so balancing is a real task.
        # Set self.data.qpos[1] to a small random angle in [-init_angle, +init_angle].
        # (self.np_random exists thanks to gym.Env.reset — use it for reproducibility.)
        self.data.qpos[1] = self.np_random.uniform(-self.init_angle, self.init_angle)

    def _get_reward(self, action):
        # +1 per step the pole is still up (a natural "survival" reward for later RL).
        if abs(self.data.qpos[1]) < self.angle_limit:
            reward = 1
        else:
            reward = 0
        return reward


    def _get_terminated(self):
        # True once |pole_angle| exceeds self.angle_limit.
        return abs(self.data.qpos[1]) > self.angle_limit