"""Phase 3: LQR keeps the pole up AND the cart centered."""
import numpy as np

import robolab
from robolab.envs.cartpole import CartpoleEnv
from robolab.controllers.lqr import LQRController
from robolab.utils.lqr import linearize, lqr_gain
from robolab.core.rollout import rollout
from robolab.utils.video import save_video


def main():
    env = CartpoleEnv(str(robolab.ASSETS_DIR / "cartpole.xml"),
                      ctrl_dt=0.01, init_angle=0.7)

    # Linearize about the upright equilibrium: cart centered, pole up, no motion.
    A, B = linearize(env.model, qpos0=[0.0, 0.0], qvel0=[0.0, 0.0], ctrl0=[0.0])

    # DESIGN CHOICE: what do you care about? (diagonal Q) and how lazy? (R)
    Q = np.diag([50, 10, 0.5, 0.7])   # [cart_x, pole_angle, cart_vel, pole_rate]
    R = np.array([[0.1]])
    K = lqr_gain(A, B, Q, R)
    print("LQR gain K =", K)

    controller = LQRController(K, x_ref=[0.0, 0.0, 0.0, 0.0])
    result = rollout(env, controller, render=True, camera="fixed")

    save_video(result["frames"], "outputs/04_lqr_cartpole.mp4", fps=round(1 / env.ctrl_dt))
    print(f"Steps balanced: {result['steps']}  (max {env.episode_length})")
    print(f"Total reward:   {result['total_reward']}")


if __name__ == "__main__":
    main()