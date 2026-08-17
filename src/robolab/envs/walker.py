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
        return bool(torso_height < 0.7)

class WalkerRunEnv(WalkerEnv):
    """Walker set up for RL locomotion: richer observation + a forward-progress
    reward. Inherits the fall-termination from WalkerEnv.

    obs = [rootz, rooty, 6 joint angles, 9 velocities]  (absolute x excluded)
    """

    def _get_obs(self):
        # qpos = [rootz, rootx, rooty, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle]
        # exclude rootx (qpos[1]); keep rootz, rooty, the 6 joints, and all qvel.
        return np.concatenate([
            self.data.qpos[0:1],   # torso height
            self.data.qpos[2:],    # torso pitch + 6 joint angles
            self.data.qvel,        # all 9 velocities (incl. forward vel qvel[1])
        ]).astype(np.float32)

    def _get_terminated(self):
        torso_height = self.data.body("torso").xpos[2]
        torso_pitch = self.data.qpos[2]          # rooty hinge = torso tilt (0 = upright)
        fallen = torso_height < 0.8              # raised from 0.7
        tipped = abs(torso_pitch) > 1.0          # ~57 deg from upright
        return bool(fallen or tipped)

    def _get_reward(self, action):
        forward = self.data.qvel[1]
        alive = 1.0
        control_cost = 0.001 * np.sum(np.square(action))
        return float(forward + alive - control_cost)   # vertical-KE term removed