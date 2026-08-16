"""Phase 3: pure-pursuit PID drives the car around a square."""
import robolab
from robolab.envs.car import CarEnv
from robolab.controllers.pursuit import PurePursuitController
from robolab.core.rollout import rollout
from robolab.utils.video import save_video


def main():
    env = CarEnv(str(robolab.ASSETS_DIR / "car.xml"))

    waypoints = [(1.5, 0.0), (1.5, 1.5), (0.0, 1.5), (0.0, 0.0)]
    controller = PurePursuitController(waypoints, kp=1.0, ki=0.1, kd=0.5,
                                       cruise=0.5, dt=env.ctrl_dt)

    result = rollout(env, controller, render=True, track_body="car", max_steps=2000)
    save_video(result["frames"], "outputs/05_pursuit_car.mp4", fps=round(1 / env.ctrl_dt))
    print(f"Final waypoint index reached: {controller.idx} / {len(waypoints) - 1}")


if __name__ == "__main__":
    main()