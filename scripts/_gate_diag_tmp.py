"""TEMP gate-vs-vx diagnostic (delete after use).

Rolls out the trained CPG-SAC policy DETERMINISTICALLY and, per step, records the
forward speed vx alongside the stance gate g AND its four multiplicative factors
(s_up, s_h, s_feet, s_hands). Then bins by vx to see WHICH factor collapses g at
speed -- testing the "gate is anti-correlated with vx / fights fast locomotion"
hypothesis. Runs on CPU so it doesn't contend with the live GPU training.
"""
import sys
import numpy as np

from _rodent_cfg import make_cpg_env
from robolab.utils.huber_sac import HuberSAC

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "outputs/rodent_cpg_sac_ckpts/sac_rodent_cpg_resume_20260824_190414_5050000_steps.zip")
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6000

env = make_cpg_env()
model = HuberSAC.load(CKPT, env=None, device="cpu")
print(f"Loaded {CKPT} on CPU; rolling out {N_STEPS} deterministic steps.")


def factors(e):
    """Reconstruct g's four soft indicators from the CURRENT env state."""
    xmat8 = float(e.data.body("torso").xmat[8])
    z = float(e.data.body("torso").xpos[2])
    s_up = np.clip((xmat8 - e.gate_up_lo) / (e.gate_up_hi - e.gate_up_lo), 0.0, 1.0)
    s_h = np.clip((z - e.gate_z_lo) / (e.gate_z_hi - e.gate_z_lo), 0.0, 1.0)
    _, _, n_foot = e._floor_contact_stats()
    s_feet = np.clip(n_foot / 2.0, 0.0, 1.0)
    s_hands = e._front_paw_proximity()
    return s_up, s_h, s_feet, s_hands


def corr(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rollout(deterministic):
    rows = []  # (vx, v*, g, s_up, s_h, s_feet, s_hands)
    acts = []
    ep_lens = []
    ep_len = 0
    obs, _ = env.reset(seed=0)
    for t in range(N_STEPS):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(action)
        su, sh, sf, sha = factors(env)
        rows.append((info["vx"], info["v_target"], info["gate"], su, sh, sf, sha))
        acts.append(np.asarray(action, dtype=np.float64))
        ep_len += 1
        if term or trunc:
            ep_lens.append(ep_len)
            ep_len = 0
            obs, _ = env.reset()
    return np.array(rows), np.array(acts), np.array(ep_lens)


edges = np.array([-1e9, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1e9])
labels = ["<0", "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", ">0.5"]


vedges = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
vlabels = ["0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5"]


def report(A, acts, ep_lens, tag):
    vx, vstar, g, su, sh, sf, sha = (A[:, i] for i in range(7))
    print(f"\n===== {tag} =====")
    print(f"n={len(A)}  vx: mean={vx.mean():.3f} std={vx.std():.3f} "
          f"[{vx.min():.2f},{vx.max():.2f}]   g: mean={g.mean():.3f}   "
          f"v*: mean={vstar.mean():.3f}")
    if len(ep_lens):
        print(f"episodes={len(ep_lens)}  ep_len: mean={ep_lens.mean():.0f} "
              f"[{ep_lens.min()},{ep_lens.max()}]")
    print(f"action |a| mean={np.abs(acts).mean():.3f}  "
          f"dfreq mean={acts[:, 0].mean():+.3f}  duty mean={acts[:, 1].mean():+.3f}")
    # Bin by COMMAND v* -> does the policy move MORE when commanded faster?
    idx = np.digitize(vstar, vedges) - 1
    print("Binned by v* (COMMAND):  does vx track the command?")
    print(f"{'v* bin':>8} {'n':>5} {'v*':>6} {'vx':>6} {'g':>6} {'s_up':>6} "
          f"{'s_feet':>7} {'s_hands':>8}")
    for b, lab in enumerate(vlabels):
        m = idx == b
        if m.sum() == 0:
            continue
        print(f"{lab:>8} {m.sum():>5} {vstar[m].mean():>6.3f} {vx[m].mean():>6.3f} "
              f"{g[m].mean():>6.3f} {su[m].mean():>6.3f} {sf[m].mean():>7.3f} "
              f"{sha[m].mean():>8.3f}")


for det, tag in [(True, "GREEDY"), (False, "STOCHASTIC")]:
    A, acts, ep_lens = rollout(deterministic=det)
    report(A, acts, ep_lens, tag)
