"""Phase 5 addendum: RESUME PPO rat training from a finished run (19 <- 13).

PPO is ON-policy, so there's no replay buffer to restore -- but it trains under
VecNormalize, and THAT wrapper's running RETURN-normalization stats are the
resume-critical state (the SAC replay buffer's analogue here). Reload the matching
vecnorm_rodent_<tag>.pkl or the value targets suddenly change scale and the critic
has to re-fit from scratch (explained_variance craters, clipping spikes).

Continues with reset_num_timesteps=False so TB curves / step counter carry on.

Run: uv run python scripts/19_resume_rodent_ppo.py --steps 2000000
     uv run python scripts/19_resume_rodent_ppo.py --path outputs/ppo_rodent_<tag>.zip
Watch: uv run python scripts/14_watch_rodent.py ppo --env run
"""
import argparse
import glob
import os
import time

import robolab
from robolab.envs.rodent import RodentRunEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from robolab.utils.montage import LearningMontageCallback
from robolab.utils.reward_logging import RewardTermLoggerCallback

XML = str(robolab.ASSETS_DIR / "rodent_scene.xml")


def make_env():
    return RodentRunEnv(XML)


def newest_final():
    cands = [p for p in glob.glob("outputs/ppo_rodent_*.zip")
             if not p.endswith("_steps.zip")]
    if not cands:
        raise FileNotFoundError(
            "No outputs/ppo_rodent_*.zip final model found. Run script 13 first, "
            "or pass --path to a checkpoint under outputs/rodent_ppo_ckpts/.")
    return max(cands, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000,
                    help="ADDITIONAL timesteps to train")
    ap.add_argument("--path", default=None,
                    help="explicit model .zip to resume (default: newest final)")
    ap.add_argument("--vecnorm", default=None,
                    help="explicit VecNormalize .pkl (default: derived from model tag)")
    args = ap.parse_args()

    path = args.path or newest_final()
    tag = time.strftime("%Y%m%d_%H%M%S")

    vec_env = make_vec_env(make_env, n_envs=8, vec_env_cls=SubprocVecEnv)

    # ppo_rodent_<tag>.zip  ->  vecnorm_rodent_<tag>.pkl (script 13's naming).
    vpath = args.vecnorm or (os.path.basename(path)
                             .replace("ppo_rodent_", "vecnorm_rodent_")
                             .replace(".zip", ".pkl"))
    vpath = vpath if os.path.isabs(vpath) or os.path.dirname(vpath) else os.path.join("outputs", vpath)
    if os.path.exists(vpath):
        vec_env = VecNormalize.load(vpath, vec_env)
        vec_env.training = True       # keep updating the running stats
        vec_env.norm_reward = True
        print(f"Loaded VecNormalize stats from {vpath}")
    else:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=0.99)
        print(f"WARNING: no VecNormalize pkl at {vpath} -> fresh return-normalization "
              f"(critic will need to re-fit the value scale).")

    model = PPO.load(path, env=vec_env, device="cpu",
                     tensorboard_log="outputs/tb_rodent_ppo")
    print(f"Resuming PPO from {path}")

    ckpt = CheckpointCallback(save_freq=50_000, save_path="outputs/rodent_ppo_ckpts",
                              name_prefix=f"ppo_rodent_resume_{tag}")

    montage = LearningMontageCallback(
        env_fn=make_env,
        record_freq=250_000,
        camera="side",
        max_steps=300,
        fps=50,
        out_path=f"outputs/rodent_ppo_resume_{tag}.mp4",
        verbose=1,
    )

    reward_log = RewardTermLoggerCallback(
        term_keys=["forward", "alive", "control_cost", "upright", "contact",
                   "hand", "energy", "smooth", "phase", "rhythm",
                   "bad_contacts", "hand_contacts", "vx"]
    )

    model.learn(total_timesteps=args.steps, reset_num_timesteps=False,
                callback=CallbackList([ckpt, montage, reward_log]))
    model.save(f"outputs/ppo_rodent_{tag}")
    vec_env.save(f"outputs/vecnorm_rodent_{tag}.pkl")
    print(f"Done. Saved model + vecnorm to outputs/ppo_rodent_{tag}.*")


if __name__ == "__main__":
    main()
