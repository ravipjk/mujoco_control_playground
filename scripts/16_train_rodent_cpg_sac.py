"""Phase 5 addendum: SAC on a CPG-structured action space (structure the action
space instead of shaping the reward).

The policy no longer commands 38 joint targets directly. It commands a Central
Pattern Generator (an integrated phase clock) + a feedback residual:

    ctrl_j = center_j + amp_gain*half_j * a_j * sin(phi + psi_j) + res_gain*half_j * r_j

Action = [dfreq, amp(nu), phase(nu), residual(nu)] = 1 + 3*nu dims. The periodic
part of every joint is forced to be a shared-clock sinusoid, so aperiodic
degenerate gaits (belly wiggle, scramble) are largely out of the search space.

The env's stance-gate config lives in scripts/_rodent_cfg.py so the resume script
(18) fine-tunes on the EXACT same reward -- flip biped/quadruped THERE, once.

Run:    uv run python scripts/16_train_rodent_cpg_sac.py
        uv run python scripts/16_train_rodent_cpg_sac.py --save-buffer-every 50000
Resume: uv run python scripts/18_resume_rodent_cpg_sac.py --steps 1000000
Watch:  uv run python scripts/14_watch_rodent.py sac --env cpg
TB:     uv run tensorboard --logdir outputs
"""
import argparse
import time

from _rodent_cfg import make_cpg_env, GATE_USE_HANDS, CPG_TERM_KEYS

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from robolab.utils.buffer import ReplayBufferSaver
from robolab.utils.montage import LearningMontageCallback
from robolab.utils.reward_logging import RewardTermLoggerCallback

tag = time.strftime("%Y%m%d_%H%M%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-buffer-every", type=int, default=0, metavar="N",
                    help="also save the replay buffer every N steps to a single "
                         "rolling file (so a Ctrl-C'd run resumes seamlessly). "
                         "0 (default) = only save the buffer at end-of-run.")
    ap.add_argument("--gamma", type=float, default=0.99, metavar="G",
                    help="discount factor. 0.99 (default) -> ~2 s effective horizon; "
                         "0.999 -> ~20 s (whole episode) to value longevity/sustained "
                         "gait. Buffer-safe, so it can also change on a resume.")
    ap.add_argument("--steps", type=int, default=1_000_000, metavar="N",
                    help="total training timesteps (default 1,000,000).")
    ap.add_argument("--ent-coef", default="auto", metavar="C",
                    help="SAC entropy temperature alpha (default 'auto'). The earlier "
                         "wind-up (alpha 6-90) that diverged the critic turned out to be "
                         "a SYMPTOM of an ill-conditioned critic (raw, 20-40x-scaled "
                         "velocity obs), now fixed by in-env obs normalization. With that "
                         "root addressed, auto-alpha is restored for exploration (and it "
                         "damps value overestimation). Pass a float (e.g. 0.1) to pin it.")
    args = ap.parse_args()
    # Accept a float ("0.1") or the literal "auto" for the temperature.
    ent_coef = args.ent_coef if args.ent_coef == "auto" else float(args.ent_coef)
    env = make_cpg_env()
    print(f"gate_use_hands={GATE_USE_HANDS}  gamma={args.gamma}  steps={args.steps:,}  "
          f"ent_coef={ent_coef}  "
          f"obs dim={env.observation_space.shape}  act dim={env.action_space.shape}")

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        # Discount factor from --gamma (default 0.99 -> ~2 s horizon). Try --gamma 0.999
        # (~20 s = whole episode) to make the full future alive+forward stream count, so
        # a "lunge now, topple in 3 s" strategy is penalized and longevity is valued;
        # watch the critic (value targets ~10x larger -> noisier/slower value learning).
        gamma=args.gamma,
        train_freq=1,
        gradient_steps=1,
        # Auto-tuned alpha (default). We first suspected the alpha wind-up (6-90) was
        # diverging the critic, but pinning it did NOT stop the divergence -- the real
        # cause was ill-conditioned inputs (raw velocity obs 20-40x other channels),
        # now normalized in the env. With the critic well-conditioned, auto-alpha is
        # the right default again: real exploration to FIND locomotion, plus the entropy
        # term regularizes the actor away from exploiting Q-overestimation. auto-alpha is
        # also self-diagnosing here -- if it STILL winds up with a stable critic, the
        # target-entropy really is too aggressive for 153 dims. (Pin via --ent-coef.)
        ent_coef=ent_coef,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        device="auto",
        tensorboard_log="outputs/tb_rodent_cpg_sac",
    )

    # Tag the checkpoint prefix so runs never overwrite each other's checkpoints
    # (files become sac_rodent_cpg_<tag>_<steps>_steps.zip). The watch/probe globs
    # (sac_rodent_cpg_*_steps.zip) still match, and newest-by-mtime still wins.
    ckpt = CheckpointCallback(save_freq=50_000, save_path="outputs/rodent_cpg_sac_ckpts",
                              name_prefix=f"sac_rodent_cpg_{tag}")

    montage = LearningMontageCallback(
        env_fn=make_cpg_env,
        record_freq=100_000,
        camera="side",
        max_steps=300,
        fps=50,
        out_path=f"outputs/rodent_cpg_sac_learning_{tag}.mp4",
        verbose=1,
    )

    reward_log = RewardTermLoggerCallback(term_keys=CPG_TERM_KEYS)

    # Rolling buffer save shares the run's end-of-run buffer path, so whether the run
    # completes or is interrupted, the buffer for tag <tag> is always at the same file.
    callbacks = [ckpt, montage, reward_log]
    if args.save_buffer_every > 0:
        callbacks.append(ReplayBufferSaver(
            save_freq=args.save_buffer_every,
            save_path=f"outputs/sac_rodent_cpg_{tag}_buffer",
            verbose=1,
        ))

    model.learn(total_timesteps=args.steps,
                callback=CallbackList(callbacks))
    model.save(f"outputs/sac_rodent_cpg_{tag}")
    # SAC is off-policy: the replay buffer IS most of the learned experience. Save
    # it next to the model so a resume (script 18) can pick up seamlessly instead
    # of re-warming an empty buffer for ~learning_starts steps. Big file (~hundreds
    # of MB); saved once at the end, not every checkpoint.
    model.save_replay_buffer(f"outputs/sac_rodent_cpg_{tag}_buffer")
    print(f"Done. Saved model + replay buffer to outputs/sac_rodent_cpg_{tag}.*")


if __name__ == "__main__":
    main()
