import numpy as np

from robolab.core.controller import Controller


class PurePursuitController(Controller):
    """Follow a list of (x, y) waypoints with a differential-drive car.

    Steering: PID on heading error (angle to target minus current yaw).
    Drive: constant cruise speed. action = [forward, turn].
    """

    def __init__(self, waypoints, kp, ki=0.0, kd=0.0,
                 cruise=0.5, reach_tol=0.2, dt=0.02):
        self.waypoints = np.asarray(waypoints, dtype=float)  # shape (N, 2)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.cruise = cruise
        self.cruise_initial = cruise
        self.reach_tol = reach_tol
        self.dt = dt
        self.reset()

    def reset(self):
        # Stateful -> zero the waypoint index and the PID memory.
        self.idx = 0
        self.int_err = 0.0
        self.prev_err = 0.0

    def act(self, observation):
        x, y, yaw = observation

        # 1. current target; if we're within reach_tol, advance to the next
        #    waypoint (but don't run past the last one).
        # 2. desired_yaw = atan2(ty - y, tx - x)
        # 3. err = desired_yaw - yaw, then WRAP to [-pi, pi]  (see hint)
        # 4. PID: accumulate int_err (+= err*dt), d_err = (err - prev_err)/dt,
        #    turn = kp*err + ki*int_err + kd*d_err;  remember to update prev_err
        # 5. return np.array([self.cruise, turn])

        tx = self.waypoints[self.idx,0]
        ty = self.waypoints[self.idx,1]
        if np.hypot(tx - x, ty - y) <= self.reach_tol and self.idx < len(self.waypoints) - 1:
            self.idx +=1
            tx = self.waypoints[self.idx,0]
            ty = self.waypoints[self.idx,1]
        desired_yaw = np.arctan2(ty - y, tx - x)
        err = desired_yaw - yaw 
        err = np.arctan2(np.sin(err), np.cos(err))
        self.int_err += err*self.dt
        d_err = (err - self.prev_err)/self.dt
        self.prev_err = err
        turn = self.kp*err + self.ki*self.int_err + self.kd*d_err

        if self.idx == len(self.waypoints) - 1:
            leg = np.hypot(tx - self.waypoints[self.idx - 1, 0],
                           ty - self.waypoints[self.idx - 1, 1])
            frac = np.hypot(tx - x, ty - y) / leg
            self.cruise = self.cruise_initial * min(frac, 1.0)

        return np.array([self.cruise, turn])
