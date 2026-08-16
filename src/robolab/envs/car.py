import numpy as np

from robolab.core.env import MujocoRobotEnv


class CarEnv(MujocoRobotEnv):
    """Differential-drive car. Observation is planar pose [x, y, yaw] so a
    path-follower gets a clean view instead of the raw 17-vector.

    action = [forward, turn], both in [-1, 1].
    """

    def _get_obs(self):
        # Position: the "car" body's world xpos (we only need x, y).
        # Heading: yaw from the body rotation matrix xmat (row-major 3x3):
        #   world-forward = (xmat[0], xmat[3]), so yaw = atan2(xmat[3], xmat[0]).
        # Return np.array([x, y, yaw], dtype=np.float32).
        pos = self.data.body("car").xpos
        xmat = self.data.body("car").xmat
        yaw = np.arctan2(xmat[3], xmat[0])
        return np.array([pos[0], pos[1], yaw], dtype=np.float32)