"""Phase 5: watch the trained PPO policy balance, via the shared rollout."""
import robolab
from robolab.envs.cartpole import CartpoleEnv
from robolab.controllers.rl import PolicyController
from robolab.core.rollout import rollout
from robolab.utils.video import save_video

from stable_baselines3 import PPO


def main():
    env = CartpoleEnv(str(robolab.ASSETS_DIR / "cartpole.xml"), init_angle=0.7)
    model = PPO.load("outputs/ppo_cartpole")
    controller = PolicyController(model, deterministic=True)

    result = rollout(env, controller, render=True, camera="fixed")
    save_video(result["frames"], "outputs/09_ppo_cartpole.mp4", fps=round(1 / env.ctrl_dt))
    print(f"Steps: {result['steps']}   Total reward: {result['total_reward']}")


if __name__ == "__main__":
    main()