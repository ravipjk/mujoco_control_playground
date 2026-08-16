import numpy as np

from robolab.core.controller import Controller


class CPGController(Controller):
    """Feedforward central pattern generator: one shared phase clock drives all
    joints with sine waves. Phase offsets between joints create the gait.

    action_j = amplitude_j * sin(phase + phase_offset_j)

    Joint order (6): [r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle].
    """

    def __init__(self, freq, amplitude, phase_offset, dt):
        self.freq = float(freq)                                    # cycles/sec
        self.amplitude = np.asarray(amplitude, dtype=float)       # (6,)
        self.phase_offset = np.asarray(phase_offset, dtype=float) # (6,)
        self.dt = dt
        self.reset()

    def reset(self):
        # the internal clock
        self.phase = 0.0

    def act(self, observation):
        # 1. action = amplitude * sin(phase + phase_offset)   (elementwise, ignores obs)
        # 2. advance the clock: phase += 2*pi * freq * dt
        # 3. return the (6,) action
        action = self.amplitude * np.sin(self.phase + self.phase_offset)
        self.phase += 2*np.pi * self.freq * self.dt
        return action

class CPGTrackingController(Controller):
    """CPG sets rhythmic TARGET joint angles; a per-joint PD loop tracks them.

    target_j = offset_j + amplitude_j * sin(phase + phase_offset_j)
    torque_j = kp * (target_j - angle_j) - kd * velocity_j

    obs = [6 joint angles, 6 joint velocities]  (from WalkerEnv).
    """

    def __init__(self, freq, amplitude, phase_offset, offset, kp, kd, dt):
        self.freq = float(freq)
        self.amplitude = np.asarray(amplitude, dtype=float)
        self.phase_offset = np.asarray(phase_offset, dtype=float)
        self.offset = np.asarray(offset, dtype=float)
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self.reset()

    def reset(self):
        self.phase = 0.0

    def act(self, observation):
        # 1. split obs -> angles = observation[:6], vels = observation[6:]
        # 2. target = offset + amplitude * sin(phase + phase_offset)
        # 3. torque = kp*(target - angles) - kd*vels
        # 4. advance phase; return torque (6,)
        angles = observation[:6]
        vels = observation[6:]
        target = self.offset + self.amplitude * np.sin(self.phase + self.phase_offset)
        torque = self.kp*(target - angles) - self.kd*vels
        self.phase += 2*np.pi * self.freq * self.dt
        return torque