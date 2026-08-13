import numpy as np

from robolab.core.controller import Controller


class PDController(Controller):
    """Proportional-derivative feedback on one scalar signal and its rate.

    u = -(kp * (obs[pos_index] - setpoint) + kd * obs[vel_index])

    For cartpole balancing: pos_index=1 (pole_angle), vel_index=3 (pole_rate),
    setpoint=0 (upright).
    """

    def __init__(self, kp, kd, pos_index, vel_index, setpoint=0.0):
        self.kp = kp
        self.kd = kd
        self.pos_index = pos_index
        self.vel_index = vel_index
        self.setpoint = setpoint 

    def act(self, observation):
        # 1. error = measured position minus setpoint
        # 2. u = +(kp*error + kd*rate) 
        # 3. return it as a 1-element array (action space is 1-dim)
        error = observation[self.pos_index] - self.setpoint
        rate = observation[self.vel_index]
        u = (self.kp * error + self.kd * rate)
        return np.array([u])