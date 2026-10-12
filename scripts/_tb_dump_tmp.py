"""TEMP: dump key scalar trajectories from a TB run (delete after use)."""
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1] if len(sys.argv) > 1 else "outputs/tb_rodent_cpg_sac/SAC_18"
ea = EventAccumulator(run, size_guidance={"scalars": 0})
ea.Reload()
tags = ["rollout/ep_len_mean", "rollout/ep_rew_mean", "reward/vx", "reward/track",
        "reward/gate", "reward/energy", "train/critic_loss", "train/ent_coef"]
avail = set(ea.Tags().get("scalars", []))
series = {}
for t in tags:
    if t in avail:
        ev = ea.Scalars(t)
        series[t] = [(e.step, e.value) for e in ev]

# print ~12 evenly spaced rows across the run
steps = [s for s, _ in series.get("rollout/ep_len_mean", [])]
if not steps:
    print("no ep_len data; tags avail:", sorted(avail)); sys.exit()
import numpy as np
grid = np.linspace(steps[0], steps[-1], 12)


def val_at(t, step):
    s = series.get(t)
    if not s:
        return float("nan")
    arr = np.array(s)
    i = int(np.argmin(np.abs(arr[:, 0] - step)))
    return arr[i, 1]


hdr = f"{'step':>9} {'ep_len':>7} {'ep_rew':>8} {'vx':>7} {'track':>6} {'gate':>6} {'energy':>7} {'critic':>9} {'alpha':>7}"
print(hdr)
for g in grid:
    print(f"{int(g):>9} {val_at('rollout/ep_len_mean', g):>7.0f} "
          f"{val_at('rollout/ep_rew_mean', g):>8.1f} {val_at('reward/vx', g):>7.3f} "
          f"{val_at('reward/track', g):>6.2f} {val_at('reward/gate', g):>6.3f} "
          f"{val_at('reward/energy', g):>7.3f} {val_at('train/critic_loss', g):>9.1f} "
          f"{val_at('train/ent_coef', g):>7.3f}")
