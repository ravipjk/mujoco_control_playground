"""Phase 5 addendum: PPO learns a locomotion gait for the dm_control rat.

Run:   uv run python scripts/13_train_rodent_ppo.py
Watch: uv run tensorboard --logdir outputs/tb_rodent_ppo

The rat is far harder than the planar walker: 38 actuators, a 145-dim
observation, a free-floating body, and contact-rich feet. Expect it to need
more timesteps and more reward-shaping iteration. Checkpoints are saved so you
can stop early (Ctrl-C) and still watch/resume.
"""
import time

import robolab
from robolab.envs.rodent import RodentRunEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from robolab.utils.montage import LearningMontageCallback
from robolab.utils.reward_logging import RewardTermLoggerCallback

tag = time.strftime("%Y%m%d_%H%M%S")

XML = str(robolab.ASSETS_DIR / "rodent_scene.xml")


def make_env():
    return RodentRunEnv(XML)


def main():
    # Each env in its own process -> real parallel CPU physics.
    # (If this errors on Windows, drop vec_env_cls to fall back to DummyVecEnv.)
    vec_env = make_vec_env(make_env, n_envs=8, vec_env_cls=SubprocVecEnv)

    # Normalize the RETURN (not the obs), so value targets are ~unit-scale and
    # the critic can actually fit them -> explained_variance climbs, clipping
    # calms down. norm_obs stays False so the policy sees raw obs and the watch
    # script needs no changes. The inner Monitor still logs raw ep_rew_mean.
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=0.99)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,           # per env -> batch = 8*2048 = 16384 transitions
        batch_size=512,
        n_epochs=10,
        gamma=0.99,             # locomotion default; raise toward 0.999 to reward
                                # longer-horizon posture (as the walker needed)
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        learning_rate=3e-4,
        target_kl=0.03,         # run 2's approx_kl ran hot (~0.85); early-stop the
                                # epoch loop when the policy moves too far per update
        policy_kwargs=dict(net_arch=[256, 256]),  # more capacity for 145->38 mapping
        verbose=1,
        device="cpu",           # net is small; MuJoCo physics (CPU) is the bottleneck
        tensorboard_log="outputs/tb_rodent_ppo",
    )

    ckpt = CheckpointCallback(save_freq=50_000, save_path="outputs/rodent_ppo_ckpts",
                              name_prefix="ppo_rodent")

    montage = LearningMontageCallback(
        env_fn=make_env,
        record_freq=250_000,    # ~20 snapshots over a 5M run
        camera="side",
        max_steps=300,          # ~6 s clip per snapshot at ctrl_dt=0.02
        fps=50,
        out_path=f"outputs/rodent_ppo_learning_{tag}.mp4",
        verbose=1,
    )

    # Log each reward component (surfaced by RodentRunEnv in info) separately.
    reward_log = RewardTermLoggerCallback(
        term_keys=["forward", "alive", "control_cost", "upright", "contact",
                   "hand", "energy", "smooth", "phase", "rhythm",
                   "bad_contacts", "hand_contacts", "vx"]
    )

    model.learn(total_timesteps=5_000_000,
                callback=CallbackList([ckpt, montage, reward_log]))
    model.save(f"outputs/ppo_rodent_{tag}")
    vec_env.save(f"outputs/vecnorm_rodent_{tag}.pkl")  # return-normalization stats
    print(f"Done. Saved to outputs/ppo_rodent_{tag}.zip")


if __name__ == "__main__":
    main()
