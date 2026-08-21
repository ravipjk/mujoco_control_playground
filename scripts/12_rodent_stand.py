"""Phase 5 addendum: load the dm_control rat and check it stands.

With zero control the 38 position servos hold their default joint angles, so a
healthy model should simply settle onto its feet and stay upright. This is our
smoke test that the freejoint + floor scene is physically sane before we train.

Run:  uv run python scripts/12_rodent_stand.py
"""
import numpy as np

import robolab
from robolab.envs.rodent import RodentStandEnv
from robolab.core.controller import Controller
from robolab.core.rollout import rollout
from robolab.utils.video import save_video


class ZeroController(Controller):
    """Commands zero to every actuator (position servos -> hold default pose)."""

    def __init__(self, n_act):
        self.n_act = n_act

    def act(self, observation):
        return np.zeros(self.n_act, dtype=np.float32)


def main():
    env = RodentStandEnv(str(robolab.ASSETS_DIR / "rodent_scene.xml"))
    print(f"nq={env.model.nq}  nv={env.model.nv}  nu={env.model.nu}")
    print(f"obs dim={env.observation_space.shape}  act dim={env.action_space.shape}")

    controller = ZeroController(env.model.nu)
    result = rollout(env, controller, max_steps=500, render=True, camera="side")

    save_video(result["frames"], "outputs/12_rodent_stand.mp4", fps=round(1 / env.ctrl_dt))
    print(f"Steps survived: {result['steps']} / 500")
    print(f"Final torso height: {env.data.body('torso').xpos[2]:.3f} m  (standing ~0.047)")


if __name__ == "__main__":
    main()
