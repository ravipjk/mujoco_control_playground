"""Phase 5 addendum: SAC learns a locomotion gait for the dm_control rat.

SAC is off-policy and sample-efficient: it reuses a replay buffer and does a
gradient step every env step, so it typically needs FEWER environment steps than
PPO to reach a gait (but more compute per step, and it runs with a single env
rather than many parallel ones). Good contrast to scripts/13 (PPO).

Run:   uv run python scripts/15_train_rodent_sac.py
Watch: uv run tensorboard --logdir outputs/tb_rodent_sac

Note: buffer_size dominates RAM (each transition stores the 145-dim obs twice).
300k here is a safe default; raise it if you have memory to spare.
"""
import time

import robolab
from robolab.envs.rodent import RodentRunEnv

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from robolab.utils.montage import LearningMontageCallback
from robolab.utils.reward_logging import RewardTermLoggerCallback

tag = time.strftime("%Y%m%d_%H%M%S")

XML = str(robolab.ASSETS_DIR / "rodent_scene.xml")


def make_env():
    return RodentRunEnv(XML)


def main():
    # SAC is off-policy: a single env feeding a replay buffer is the norm.
    env = make_env()

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=300_000,        # replay capacity (RAM-heavy)
        learning_starts=10_000,     # pure exploration warm-up
        batch_size=256,
        tau=0.005,                  # target-network smoothing
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",            # auto-tuned max-entropy temperature
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        device="auto",              # SAC does a grad step every env step; GPU can help
        tensorboard_log="outputs/tb_rodent_sac",
    )

    ckpt = CheckpointCallback(save_freq=50_000, save_path="outputs/rodent_sac_ckpts",
                              name_prefix="sac_rodent")

    montage = LearningMontageCallback(
        env_fn=make_env,
        record_freq=100_000,        # SAC needs fewer steps -> record more often
        camera="side",
        max_steps=300,
        fps=50,
        out_path=f"outputs/rodent_sac_learning_{tag}.mp4",
        verbose=1,
    )

    reward_log = RewardTermLoggerCallback(
        term_keys=["forward", "alive", "control_cost", "upright", "contact",
                   "hand", "energy", "smooth", "phase", "rhythm",
                   "bad_contacts", "hand_contacts", "vx"]
    )

    # Fewer env steps than PPO thanks to sample efficiency (but slower per step).
    model.learn(total_timesteps=1_000_000,
                callback=CallbackList([ckpt, montage, reward_log]))
    model.save(f"outputs/sac_rodent_{tag}")
    print(f"Done. Saved to outputs/sac_rodent_{tag}.zip")


if __name__ == "__main__":
    main()
