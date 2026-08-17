"""Phase 5: watch the trained PPO walker, via the shared rollout.

Run:  uv run python scripts/11_watch_walker.py
"""
import robolab
from robolab.envs.walker import WalkerRunEnv
from robolab.controllers.rl import PolicyController
from robolab.core.rollout import rollout
from robolab.utils.video import save_video

from stable_baselines3 import PPO


def main():
    env = WalkerRunEnv(str(robolab.ASSETS_DIR / "walker.xml"))
    model = PPO.load("outputs/ppo_walker")   # or a checkpoint: outputs/walker_ckpts/ppo_walker_XXXXXX_steps
    controller = PolicyController(model, deterministic=True)

    result = rollout(env, controller, render=True, camera="side", max_steps=1000)
    save_video(result["frames"], "outputs/11_ppo_walker.mp4", fps=round(1 / env.ctrl_dt))
    print(f"Steps: {result['steps']}   Total reward: {result['total_reward']:.1f}")


if __name__ == "__main__":
    main()
