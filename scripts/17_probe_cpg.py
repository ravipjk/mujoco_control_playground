"""Phase 5 diagnostic: what does the CPG policy actually DO?

Loads the newest CPG-SAC checkpoint and runs ONE deterministic episode with NO
rendering (cheap, safe to run alongside training). Per control step it recomputes
the quantities the env hides inside _apply_action / _get_reward:

  - f         = cpg_f_base + cpg_f_range * action[0]   -> the commanded clock (Hz)
  - amp_mean  = mean of 0.5*(action[1:1+nu]+1)          -> how hard the CPG drives
  - res_mean  = mean |action[1+2nu:1+3nu]|              -> how much feedback residual
  - coff_mean = mean |action[1+3nu:1+4nu]|              -> Q3 center-offset command
  - coff_ema  = L1 of env._coff_ema                     -> the ACTUAL slow setpoint shift
  - vx, gate, hand_prox, bad/hand contacts             -> from info (reward terms)

Then it prints summary stats + a downsampled f time-series so we can see whether
the policy parks at a steady tempo or thrashes the frequency knob step-to-step.

Uses make_cpg_env so the obs/action layout matches the trained model (augmented
per-paw obs + the 1+4*nu action space).

Run:  uv run python scripts/17_probe_cpg.py
"""
import glob
import os
import sys

import numpy as np

from stable_baselines3 import SAC

sys.path.insert(0, "scripts")
from _rodent_cfg import make_cpg_env


def latest_cpg_ckpt():
    finals = glob.glob("outputs/sac_rodent_cpg_*.zip")
    ckpts = glob.glob("outputs/rodent_cpg_sac_ckpts/sac_rodent_cpg_*_steps.zip")
    cands = finals + ckpts
    if not cands:
        raise FileNotFoundError("No CPG-SAC checkpoint found yet.")
    return max(cands, key=os.path.getmtime)


def main():
    path = latest_cpg_ckpt()
    model = SAC.load(path, device="cpu")
    print(f"Loaded {path}")

    env = make_cpg_env()
    nu = env.model.nu

    obs, _ = env.reset()
    x0 = float(env.data.body("torso").xpos[0])

    F, AMP, RES, COFF, CEMA, VX, GATE, BAD, HAND = [], [], [], [], [], [], [], [], []
    steps = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        a = np.asarray(action, dtype=np.float64)

        f = env.cpg_f_base + env.cpg_f_range * float(a[0])
        amp_mean = float(np.mean(0.5 * (a[1:1 + nu] + 1.0)))          # [0,1]
        res_mean = float(np.mean(np.abs(a[1 + 2 * nu:1 + 3 * nu])))    # [0,1]
        coff_mean = float(np.mean(np.abs(a[1 + 3 * nu:1 + 4 * nu])))   # [0,1] Q3 command

        obs, reward, terminated, truncated, info = env.step(action)
        # ACTUAL slow setpoint shift after the EMA (rad, summed |.| over joints).
        cema = float(np.sum(np.abs(env._coff_ema))) if env._coff_ema is not None else 0.0
        F.append(f); AMP.append(amp_mean); RES.append(res_mean)
        COFF.append(coff_mean); CEMA.append(cema)
        VX.append(info["vx"]); GATE.append(info["gate"])
        BAD.append(info["bad_contacts"]); HAND.append(info["hand_contacts"])
        steps += 1
        if terminated or truncated:
            break

    x1 = float(env.data.body("torso").xpos[0])
    F = np.array(F); VX = np.array(VX)

    def stat(name, arr, fmt="{:.3f}"):
        arr = np.asarray(arr)
        print(f"  {name:<14} mean={fmt.format(arr.mean())}  std={fmt.format(arr.std())}"
              f"  min={fmt.format(arr.min())}  max={fmt.format(arr.max())}")

    print(f"\nEpisode: {steps} steps   net torso x-displacement = {x1 - x0:+.3f} m\n")
    print("Per-step summary:")
    stat("freq f (Hz)", F)
    stat("amp_mean", AMP)
    stat("res_mean", RES)
    stat("coff_mean", COFF)
    stat("coff_ema |.|", CEMA)
    stat("vx (m/s)", VX)
    stat("gate", GATE)
    stat("bad_contacts", BAD, "{:.2f}")
    stat("hand_contacts", HAND, "{:.2f}")

    jitter = float(np.mean(np.abs(np.diff(F)))) if len(F) > 1 else 0.0
    print(f"\n  freq step-to-step mean|df| = {jitter:.3f} Hz  "
          f"(small => parks at a tempo; large => thrashes the knob)")

    print("\n  f time-series (every 25 steps):")
    print("   " + "  ".join(f"{v:.2f}" for v in F[::25]))


if __name__ == "__main__":
    main()
