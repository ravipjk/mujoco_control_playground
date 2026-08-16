import numpy as np

import robolab
from robolab.envs.walker import WalkerEnv
from robolab.controllers.cpg import CPGController
from robolab.core.rollout import rollout
from robolab.utils.video import save_video


def main():
    env = WalkerEnv(str(robolab.ASSETS_DIR / "walker.xml"))
    controller = CPGController(
        freq=1.5,
        amplitude=[0.5, 0.5, 0.3, 0.5, 0.5, 0.3],
        phase_offset=[0.0, 0.0, 0.0, np.pi, np.pi, np.pi],
        dt=env.ctrl_dt,
    )
    result = rollout(env, controller, render=True, camera="side", max_steps=1000)
    save_video(result["frames"], "outputs/06_cpg_walker.mp4", fps=round(1 / env.ctrl_dt))
    print(f"Steps before falling: {result['steps']}")


if __name__ == "__main__":
    main()