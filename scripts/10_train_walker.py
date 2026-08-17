"""Phase 5: PPO learns a walking gait for the planar biped.

Run:  uv run python scripts/10_train_walker.py
Watch curves:  uv run tensorboard --logdir outputs/tb_walker
"""
import robolab
from robolab.envs.walker import WalkerRunEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from robolab.utils.montage import LearningMontageCallback

import time
tag = time.strftime("%Y%m%d_%H%M%S")

XML = str(robolab.ASSETS_DIR / "walker.xml")


def make_env():
    return WalkerRunEnv(XML)


def main():
    # SubprocVecEnv = each env in its own process -> real parallel CPU physics.
    # (If this errors on Windows, drop the vec_env_cls arg to fall back to
    #  DummyVecEnv, which runs the envs sequentially in one process.)
    vec_env = make_vec_env(make_env, n_envs=8, vec_env_cls=SubprocVecEnv)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,           # per env -> batch = 8*2048 = 16384 transitions
        batch_size=512,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        learning_rate=3e-4,
        verbose=1,
        device="cpu",           # tiny net; MuJoCo physics (CPU) is the bottleneck
        tensorboard_log="outputs/tb_walker",
    )

    # Save a checkpoint every ~25k steps so a long run isn't lost / you can peek early.
    ckpt = CheckpointCallback(save_freq=25_000, save_path="outputs/walker_ckpts",
                              name_prefix="ppo_walker")

    # Record a learning-montage: a labeled clip of the policy every 100k steps,
    # stitched into one video so you can watch the gait evolve.
    montage = LearningMontageCallback(
        env_fn=make_env,
        record_freq=100_000,
        camera="side",
        max_steps=300,          # ~6 s clip per snapshot at ctrl_dt=0.02
        fps=50,
        out_path=f"outputs/walker_learning_{tag}.mp4",
        verbose=1,
    )

    model.learn(total_timesteps=2_000_000, callback=CallbackList([ckpt, montage]))
    model.save("outputs/ppo_walker")
    print("Done. Saved to outputs/ppo_walker.zip")


if __name__ == "__main__":
    main()
