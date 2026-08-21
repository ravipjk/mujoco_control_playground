"""Phase 5 addendum: watch a trained rat policy (PPO or SAC), deterministically.

Run:  uv run python scripts/14_watch_rodent.py ppo
      uv run python scripts/14_watch_rodent.py sac
      uv run python scripts/14_watch_rodent.py sac --env cpg
      uv run python scripts/14_watch_rodent.py sac --path outputs/rodent_sac_ckpts/sac_rodent_1000000_steps.zip

IMPORTANT: always watch the DETERMINISTIC policy. SAC trains with a stochastic,
high-entropy policy (ent_coef>0), so its mid-training montage looks far messier
than the deployable deterministic policy -- evaluate that one, not the rollouts.
"""
import argparse
import glob
import os

import robolab
from robolab.envs.rodent import RodentRunEnv, RodentCPGEnv
from robolab.controllers.rl import PolicyController
from robolab.core.rollout import rollout
from robolab.utils.video import save_video

from stable_baselines3 import PPO, SAC


def latest_model(algo, env_kind):
    """Most recent saved model or checkpoint for `algo` (ppo/sac) and env kind."""
    tag = "_cpg" if env_kind == "cpg" else ""
    finals = glob.glob(f"outputs/{algo}_rodent{tag}_*.zip")
    ckpts = glob.glob(f"outputs/rodent{tag}_{algo}_ckpts/{algo}_rodent{tag}_*_steps.zip")
    candidates = finals + ckpts
    if env_kind == "run":  # the plain glob also matches *_cpg_* -> drop those
        candidates = [c for c in candidates if "_cpg_" not in os.path.basename(c)]
    if not candidates:
        raise FileNotFoundError(f"No {algo} ({env_kind}) model found - train first.")
    return max(candidates, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("algo", nargs="?", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--env", default="run", choices=["run", "cpg"],
                    help="run = direct joint control; cpg = CPG-structured action space")
    ap.add_argument("--path", default=None, help="explicit .zip to load (else newest)")
    args = ap.parse_args()

    path = args.path or latest_model(args.algo, args.env)
    Algo = PPO if args.algo == "ppo" else SAC
    model = Algo.load(path, device="cpu")
    print(f"Loaded {args.algo.upper()} ({args.env}) from {path}")

    EnvCls = RodentCPGEnv if args.env == "cpg" else RodentRunEnv
    env = EnvCls(str(robolab.ASSETS_DIR / "rodent_scene.xml"))
    controller = PolicyController(model, deterministic=True)

    result = rollout(env, controller, render=True, camera="side", max_steps=1000)
    out = f"outputs/14_{args.algo}_{args.env}_rodent.mp4"
    save_video(result["frames"], out, fps=round(1 / env.ctrl_dt))
    print(f"Steps: {result['steps']}   Total reward: {result['total_reward']:.1f}   -> {out}")


if __name__ == "__main__":
    main()
