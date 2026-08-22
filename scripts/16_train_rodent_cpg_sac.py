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

from robolab.utils.huber_sac import HuberSAC
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
    ap.add_argument("--lr", type=float, default=3e-4, metavar="LR",
                    help="learning rate for actor, critic AND alpha (SB3 shares one). "
                         "SB3's 3e-4 default is tuned for ~6-17 dim MuJoCo; our 153-dim "
                         "action space makes Q harder to approximate and the critic "
                         "over-estimates and diverges (Q ~ 45k @25k while alpha still ~1 "
                         "-> the divergence is inner-loop, not alpha). Lower it (e.g. 1e-4) "
                         "to slow the bootstrap feedback and keep the critic bounded.")
    ap.add_argument("--critic-loss", choices=["mse", "huber"], default="mse",
                    help="critic regression loss. 'mse' (SB3 default) has an UNBOUNDED "
                         "per-sample gradient (= TD residual), so the bootstrap "
                         "over-estimation loop self-amplifies and Q diverges. 'huber' "
                         "saturates that gradient at +/-(--huber-delta), capping the loop "
                         "gain so the critic can't blow up. Use 'huber' for the 153-dim "
                         "CPG action space where mse diverges at every tested lr.")
    ap.add_argument("--huber-delta", type=float, default=50.0, metavar="D",
                    help="Huber knee (only used with --critic-loss huber). Below D the "
                         "loss is 0.5*r^2 (== mse, accurate fitting); above D it is linear "
                         "(bounded gradient). Set ABOVE the healthy TD-residual RMS (~25 "
                         "here) and BELOW the divergent scale (100s-1000s). Raise it if you "
                         "raise --gamma (bigger returns -> bigger residuals).")
    ap.add_argument("--ent-coef", default="auto", metavar="C",
                    help="SAC entropy temperature alpha (default 'auto', tuned to hit "
                         "--target-entropy). Pass a float (e.g. 0.1) to PIN it instead.")
    ap.add_argument("--target-entropy", default="-2.0", metavar="H",
                    help="target policy entropy for auto-alpha, in nats PER ACTION DIM "
                         "(default -2.0). SB3's default is -1.0/dim (=-|A|), which is too "
                         "aggressive for this 153-dim action space: the policy (under "
                         "reward pressure) sits BELOW -1/dim, so the thermostat cranks "
                         "alpha forever (SAC_9: alpha -> ~1950, re-diverging the critic). "
                         "A MORE NEGATIVE target demands LESS randomness -> reachable "
                         "equilibrium -> alpha settles. Ignored unless --ent-coef auto. "
                         "Pass 'sb3' for SB3's -|A| default.")
    args = ap.parse_args()
    # Accept a float ("0.1") or the literal "auto" for the temperature.
    ent_coef = args.ent_coef if args.ent_coef == "auto" else float(args.ent_coef)
    env = make_cpg_env()
    # Target entropy is only consulted for auto-alpha. SB3 wants a TOTAL (summed over
    # dims); we expose it per-dim (dimension-independent) and multiply by |A| here.
    adim = env.action_space.shape[0]
    if ent_coef == "auto" and args.target_entropy != "sb3":
        target_entropy = float(args.target_entropy) * adim
    else:
        target_entropy = "auto"   # SB3's -|A|; also the (unused) value when alpha is pinned
    crit = f"{args.critic_loss}" + (f"(delta={args.huber_delta:g})" if args.critic_loss == "huber" else "")
    print(f"gate_use_hands={GATE_USE_HANDS}  gamma={args.gamma}  steps={args.steps:,}  "
          f"lr={args.lr:g}  ent_coef={ent_coef}  target_entropy={target_entropy}  "
          f"critic_loss={crit}  "
          f"obs dim={env.observation_space.shape}  act dim={env.action_space.shape}")

    # HuberSAC only differs from SAC in the critic loss (robust vs mse); pick per flag.
    Algo = HuberSAC if args.critic_loss == "huber" else SAC
    extra = dict(huber_delta=args.huber_delta) if args.critic_loss == "huber" else {}
    model = Algo(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
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
        # Auto-tuned alpha, but with a REACHABLE target entropy (--target-entropy).
        # Two roots of the chronic critic divergence, found in sequence:
        #   1. ill-conditioned inputs (raw velocity obs 20-40x other channels) -> fixed
        #      by in-env obs normalization (SAC_9 then STARTED healthy: critic ~64).
        #   2. auto-alpha still wound up (SAC_9: alpha -> ~1950) because SB3's default
        #      target -|A| = -153 is unreachable here -> the -alpha*logpi term re-inflated
        #      the critic. Fixed by demanding LESS entropy (more-negative target), so the
        #      thermostat finds an equilibrium and alpha settles (keeps adaptivity, unlike
        #      a hard pin). Set target via --target-entropy; pin alpha via --ent-coef.
        ent_coef=ent_coef,
        target_entropy=target_entropy,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        device="auto",
        tensorboard_log="outputs/tb_rodent_cpg_sac",
        **extra,
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
