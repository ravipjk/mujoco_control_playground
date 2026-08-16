import numpy as np

import robolab
from robolab.envs.walker import WalkerEnv
from robolab.controllers.cpg import CPGTrackingController
from robolab.core.rollout import rollout
from robolab.utils.video import save_video


def main():
    env = WalkerEnv(str(robolab.ASSETS_DIR / "walker.xml"))
    controller = CPGTrackingController(
        freq=1.2,
        offset=[0.3, -0.6, 0.3, 0.3, -0.6, 0.3],     # nominal posture (knees bent)
        amplitude=[0.2, 0.3, 0.2, 0.2, 0.3, 0.2],
        phase_offset=[0.0, 0.0, 0.0, np.pi, np.pi, np.pi],
        kp=8.0,
        kd=0.2,
        dt=env.ctrl_dt,
    )
    result = rollout(env, controller, render=True, camera="side", max_steps=1000)
    save_video(result["frames"], "outputs/07_cpg_pd_walker.mp4", fps=round(1 / env.ctrl_dt))
    print(f"Steps before falling: {result['steps']}")


if __name__ == "__main__":
    main()