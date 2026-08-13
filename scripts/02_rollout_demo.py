"""Phase 2 demo: one rollout runner, two (env, controller) pairs."""
import robolab
from robolab.core.env import MujocoRobotEnv
from robolab.core.rollout import rollout
from robolab.controllers.basic import RandomController, SinusoidController
from robolab.utils.video import save_video


def main():
    # 1) cartpole + RandomController
    env_cartpole = MujocoRobotEnv(str(robolab.ASSETS_DIR / "cartpole.xml"))
    controller_cartpole = RandomController(env_cartpole.action_space)
    result_cartpole = rollout(env_cartpole, controller_cartpole, render=True)

    save_video(result_cartpole["frames"], "outputs/02_rollout_demo_cartpole.mp4", fps=round(1/env_cartpole.ctrl_dt))
    print(f"Cartpole: Total Reward -> {result_cartpole["total_reward"]}")
    print(f"Cartpole: Total Steps -> {result_cartpole["steps"]}")

    # 2) car + SinusoidController, following the "car" body
    env_car = MujocoRobotEnv(str(robolab.ASSETS_DIR / "car.xml"))
    controller_car = SinusoidController(dt=env_car.ctrl_dt, amplitude=[0.5, 0.3], freq=[0.05, 0.2])
    result_car = rollout(env_car, controller_car, render=True, track_body="car")

    save_video(result_car["frames"], "outputs/02_rollout_demo_car.mp4", fps=round(1/env_car.ctrl_dt))
    print(f"Car: Total Reward -> {result_car["total_reward"]}")
    print(f"Car: Total Steps -> {result_car["steps"]}")


if __name__ == "__main__":
    main()