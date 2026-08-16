import numpy as np

from robolab.core.env import MujocoRobotEnv


class WalkerEnv(MujocoRobotEnv):
    """Planar biped. Obs = 6 joint angles + 6 joint velocities (root DOFs
    excluded; the gait controller only needs joint feedback).

    qpos = [rootz, rootx, rooty, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle]
    action = 6 joint torques in [-1, 1].
    """

    def _get_obs(self):
        # joint angles  = qpos[3:]   (6 values)
        # joint velocities = qvel[3:] (6 values)
        # return them concatenated as float32, shape (12,)
        joint_angles  = self.data.qpos[3:]
        joint_velocities  = self.data.qvel[3:]
        return np.concatenate([joint_angles, joint_velocities]).astype(np.float32)

    def _get_terminated(self):
        # Fell over: terminate when the torso's world height drops below 0.7 m
        # (it starts at ~1.3 m). Use self.data.body("torso").xpos[2].
        torso_height = self.data.body("torso").xpos[2]
        return torso_height < 0.7