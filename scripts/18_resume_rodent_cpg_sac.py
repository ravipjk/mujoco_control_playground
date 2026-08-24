"""Phase 5 addendum: RESUME CPG-SAC training from a finished run (18 <- 16).

Off-policy fine-tuning. Loads the newest outputs/sac_rodent_cpg_*.zip final model
(or --path), restores its replay buffer if the matching _buffer.pkl exists, rebuilds
the env from the SHARED config (scripts/_rodent_cfg.py) so the reward can't drift,
and continues with reset_num_timesteps=False so the TB curves and the global step
counter carry straight on from where the run stopped.

Two resume qualities:
  * SEAMLESS  -- buffer present: SAC keeps its off-policy memory, learning continues
                 immediately at the same data efficiency.
  * POLICY-ONLY -- no buffer (e.g. a run that predates buffer-saving): the network
                 weights + entropy temperature carry over, but the buffer restarts
                 empty, so expect a brief `learning_starts` re-warmup dip.

Run: uv run python scripts/18_resume_rodent_cpg_sac.py --steps 1000000
     uv run python scripts/18_resume_rodent_cpg_sac.py --path outputs/sac_rodent_cpg_<tag>.zip
Watch: uv run python scripts/14_watch_rodent.py sac --env cpg
"""
import argparse
import glob
import os
import re
import time

from _rodent_cfg import make_cpg_env, CPG_TERM_KEYS

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from robolab.utils.huber_sac import HuberSAC
from robolab.utils.buffer import ReplayBufferSaver
from robolab.utils.montage import LearningMontageCallback
from robolab.utils.reward_logging import RewardTermLoggerCallback


def newest_model():
    """Newest resumable model: whichever of the final models OR the per-run
    checkpoints was written last. So a completed run resumes from its final .zip,
    and a Ctrl-C'd run resumes from its newest checkpoint -- no --path needed."""
    finals = [p for p in glob.glob("outputs/sac_rodent_cpg_*.zip")
              if not p.endswith("_steps.zip")]
    ckpts = glob.glob("outputs/rodent_cpg_sac_ckpts/sac_rodent_cpg_*_steps.zip")
    cands = finals + ckpts
    if not cands:
        raise FileNotFoundError(
            "No outputs/sac_rodent_cpg_*.zip model or checkpoint found. Run script 16 "
            "first, or pass --path.")
    return max(cands, key=os.path.getmtime)


def find_buffer(model_path):
    """Locate the replay buffer for a model. Tries the paired <model>_buffer.pkl
    (final-model save), then the run's rolling buffer outputs/sac_rodent_cpg_<tag>_buffer.pkl
    (written by --save-buffer-every), keyed off the YYYYmmdd_HHMMSS tag in the name."""
    direct = model_path[:-4] + "_buffer.pkl"
    if os.path.exists(direct):
        return direct
    m = re.search(r"sac_rodent_cpg_(\d{8}_\d{6})", os.path.basename(model_path))
    if m:
        rolling = os.path.join("outputs", f"sac_rodent_cpg_{m.group(1)}_buffer.pkl")
        if os.path.exists(rolling):
            return rolling
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000,
                    help="ADDITIONAL timesteps to train (added on top of the run so far)")
    ap.add_argument("--path", default=None,
                    help="explicit model .zip to resume (default: newest model/checkpoint)")
    ap.add_argument("--save-buffer-every", type=int, default=0, metavar="N",
                    help="save the replay buffer every N steps to a rolling file so "
                         "THIS resume is itself Ctrl-C resumable. 0 = end-of-run only.")
    ap.add_argument("--huber-delta", type=float, default=None, metavar="D",
                    help="override the critic Huber knee for this resume (default: keep "
                         "the value saved in the model). Raise it if you also raise gamma "
                         "on the resume -- bigger returns -> bigger TD residuals.")
    ap.add_argument("--no-buffer", action="store_true",
                    help="POLICY-ONLY resume: do NOT load the saved replay buffer. Use "
                         "when the REWARD changed (e.g. a new track_sigma): the buffer's "
                         "stored rewards are now stale, so keep the trained weights but "
                         "refill the buffer under the new reward (brief re-warmup).")
    args = ap.parse_args()

    path = args.path or newest_model()
    tag = time.strftime("%Y%m%d_%H%M%S")

    env = make_cpg_env()
    # Load as HuberSAC (NOT plain SAC): the runs use the robust Huber critic loss that
    # stops the critic divergence, and .load's CLASS decides train() -- SAC.load would
    # silently revert the resume to MSE. huber_delta is restored from the model's saved
    # data (or defaults to 50 for a pre-Huber zip); --huber-delta overrides it below.
    model = HuberSAC.load(path, env=env, device="auto",
                          tensorboard_log="outputs/tb_rodent_cpg_sac")
    if args.huber_delta is not None:
        model.huber_delta = float(args.huber_delta)
    print(f"Resuming CPG-SAC from {path}  (critic=huber, delta={model.huber_delta:g})")

    buf = None if args.no_buffer else find_buffer(path)
    if args.no_buffer:
        print("--no-buffer: POLICY-ONLY resume (reward changed -> stale buffer skipped; "
              "weights carry over, buffer refills under the new reward).")
    elif buf:
        model.load_replay_buffer(buf)
        print(f"Loaded replay buffer ({model.replay_buffer.size()} transitions) "
              f"from {buf} -> SEAMLESS resume")
    else:
        print(f"WARNING: no replay buffer found for {os.path.basename(path)} "
              f"-> POLICY-ONLY resume (brief re-warmup while the buffer refills).")

    ckpt = CheckpointCallback(save_freq=50_000, save_path="outputs/rodent_cpg_sac_ckpts",
                              name_prefix=f"sac_rodent_cpg_resume_{tag}")

    montage = LearningMontageCallback(
        env_fn=make_cpg_env,
        record_freq=100_000,
        camera="side",
        max_steps=300,
        fps=50,
        out_path=f"outputs/rodent_cpg_sac_resume_{tag}.mp4",
        verbose=1,
    )

    reward_log = RewardTermLoggerCallback(term_keys=CPG_TERM_KEYS)

    callbacks = [ckpt, montage, reward_log]
    if args.save_buffer_every > 0:
        callbacks.append(ReplayBufferSaver(
            save_freq=args.save_buffer_every,
            save_path=f"outputs/sac_rodent_cpg_{tag}_buffer",
            verbose=1,
        ))

    # reset_num_timesteps=False: keep counting from the loaded model's step count so
    # TensorBoard curves append instead of restarting at 0.
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False,
                callback=CallbackList(callbacks))
    model.save(f"outputs/sac_rodent_cpg_{tag}")
    model.save_replay_buffer(f"outputs/sac_rodent_cpg_{tag}_buffer")
    print(f"Done. Saved model + replay buffer to outputs/sac_rodent_cpg_{tag}.*")


if __name__ == "__main__":
    main()
