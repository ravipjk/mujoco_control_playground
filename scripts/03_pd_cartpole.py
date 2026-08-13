"""Phase 3: PD feedback balances the cartpole."""
import robolab
from robolab.envs.cartpole import CartpoleEnv
from robolab.controllers.pd import PDController
from robolab.core.rollout import rollout
from robolab.utils.video import save_video


def main():
    env = CartpoleEnv(str(robolab.ASSETS_DIR / "cartpole.xml"), init_angle=0.7)
    controller = PDController(kp=10.0, kd=1.0, pos_index=1, vel_index=3, setpoint=0.0)
    result = rollout(env, controller, render=True, camera="fixed")

    save_video(result["frames"], "outputs/03_pd_cartpole.mp4", fps=round(1 / env.ctrl_dt))
    print(f"Steps balanced: {result['steps']}  (max {env.episode_length})")
    print(f"Total reward:   {result['total_reward']}")


if __name__ == "__main__":
    main()