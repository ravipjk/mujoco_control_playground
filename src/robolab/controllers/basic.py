import numpy as np
from robolab.core.controller import Controller


class RandomController(Controller):
    """Stateless baseline: sample a uniform-random action every step."""

    def __init__(self, action_space):
        """Store whatever you need to produce valid actions."""
        self.action_space = action_space

    def act(self, observation):
        """Return a random action within bounds (ignores the observation)."""
        action = self.action_space.sample()
        return action


class SinusoidController(Controller):
    """Open-loop, stateful: action_i(t) = amplitude_i * sin(2*pi * freq_i * t).

    Ignores the observation. Advances its own clock by `dt` each act().
    """

    def __init__(self, dt, amplitude, freq):
        """dt: control timestep; amplitude, freq: per-action-dim (scalars ok too)."""
        self.dt = dt
        self.amplitude = np.asarray(amplitude, dtype=float)
        self.freq = np.asarray(freq, dtype=float)
        self.t = 0.0

    def reset(self):
        """Stateful controller -> must zero its internal clock here."""
        self.t = 0.0
        

    def act(self, observation):
        """Compute the sinusoid at the current time, then advance the clock."""
        action = self.amplitude * np.sin(2.0*np.pi * self.freq * self.t)
        self.t = self.t + self.dt
        return action