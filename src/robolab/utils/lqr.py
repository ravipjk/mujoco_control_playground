import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are


def linearize(model, qpos0, qvel0, ctrl0, eps=1e-6):
    """Finite-difference the discrete dynamics x_{t+1} = A x_t + B u_t at an
    operating point. Returns (A, B). Works for ANY MuJoCo model.

    State x has dimension 2*nv + na, ordered [qpos-ish, qvel, act], which for
    the cartpole is exactly [cart_x, pole_angle, cart_vel, pole_rate].
    """
    data = mujoco.MjData(model)                 # scratch state; don't disturb the env
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    data.ctrl[:] = ctrl0
    mujoco.mj_forward(model, data)              # settle derived quantities

    nx = 2 * model.nv + model.na
    nu = model.nu
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    # mjd_transitionFD doesn't support RK4; swap to Euler just for the FD, then restore.
    saved_integrator = model.opt.integrator
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    try:
        mujoco.mjd_transitionFD(model, data, eps, True, A, B, None, None)
    finally:
        model.opt.integrator = saved_integrator
    return A, B


def lqr_gain(A, B, Q, R):
    """Optimal discrete-time LQR gain K for cost sum(x'Qx + u'Ru).
    Control law is u = -K (x - x_ref)."""
    P = solve_discrete_are(A, B, Q, R)          # the Riccati solution
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K