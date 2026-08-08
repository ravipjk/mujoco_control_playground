import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MujocoRobotEnv(gym.Env):
    """A thin, Gymnasium-compatible MuJoCo plant.

    A controlled discrete-time dynamical system:
      reset()        -> observation
      step(action)   -> (observation, reward, terminated, truncated, info)

    Sensor/actuator seam: every observation is produced by _get_obs() (the output
    map h) and every actuation goes through _apply_action() (the input map).
    Subclasses customize behavior by overriding those two plus the task hooks.
    """

    def __init__(self, xml_path, ctrl_dt=0.02, episode_length=1000):
        """Build the plant from an MJCF file.
        
        Postconditions: self.model, self.data exist; self.n_substeps encodes the
        zero-order-hold ratio (ctrl_dt / sim_dt); self.action_space and
        self.observation_space are defined.
        """
        model = mujoco.MjModel.from_xml_path(xml_path)
        data  = mujoco.MjData(model)
        self.model = model
        self.data = data
        self.ctrl_dt = ctrl_dt
        self.episode_length = episode_length
        sim_dt = model.opt.timestep
        self.n_substeps = int(ctrl_dt / sim_dt)
        self.action_space = spaces.Box(low=model.actuator_ctrlrange[:,0].astype(np.float32), high=model.actuator_ctrlrange[:,1].astype(np.float32), shape=(model.nu,))
        sample_obs = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32)
        super().__init__()
        

    def reset(self, seed=None, options=None):
        """Return the plant to an initial condition. Returns (observation, info)."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._reset_model()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        """Apply one action under zero-order hold for n_substeps physics steps.

        Returns (observation, reward, terminated, truncated, info).
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_action(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._get_reward(action)
        terminated = self._get_terminated()
        truncated = self.data.time >= self.episode_length * self.ctrl_dt 
        return obs, reward, terminated, truncated, {}

    # ---- sensor / actuator SEAM (subclasses override) ----
    def _get_obs(self):
        """Output map h(state) -> observation. Default: [qpos, qvel] as float32."""
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _apply_action(self, action):
        """Input map: write the controller's action into the actuators."""
        self.data.ctrl[:] = action

    # ---- task hooks (trivial defaults; subclasses override) ----
    def _reset_model(self):
        """Optional initial-state randomization. Default: no-op."""
        return 

    def _get_reward(self, action):
        """Scalar reward for the last step. Default: 0.0 (rewards come in Phase 3)."""
        return 0.0

    def _get_terminated(self):
        """Whether the task ended intrinsically this step. Default: False."""
        return False
