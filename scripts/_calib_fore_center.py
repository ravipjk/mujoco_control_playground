"""throwaway calibration: find forelimb CPG center OFFSETS (ctrl in [-1,1]) that put
the front paws on the floor at the trunk's natural settled ride height. Coordinate
descent over a coarse grid, then a sparsify pass so only the joints that matter carry
an offset (keeps the nominal pose gentle). Prints a dict to paste into RodentCPGEnv."""
import numpy as np
import mujoco
import robolab
from robolab.envs.rodent import RodentCPGEnv

env = RodentCPGEnv(str(robolab.ASSETS_DIR / "rodent_scene.xml"))
m, d = env.model, env.data
env.reset(); env._floor_contact_stats()
hand_geoms = env._hand_geoms

fore_L = ["scapula_L_supinate", "scapula_L_abduct", "scapula_L_extend",
          "shoulder_L", "shoulder_sup_L", "elbow_L", "wrist_L", "finger_L"]
fore_R = ["scapula_R_supinate", "scapula_R_abduct", "scapula_R_extend",
          "shoulder_R", "shoulder_sup_R", "elbow_R", "wrist_R", "finger_R"]
fore = fore_L + fore_R
fore_act = [m.actuator(n).id for n in fore]
center = 0.5 * (m.actuator_ctrlrange[:, 0] + m.actuator_ctrlrange[:, 1]).astype(float)

hand_L = [g for g in hand_geoms if "_L" in (m.geom(g).name or "")]
hand_R = [g for g in hand_geoms if "_R" in (m.geom(g).name or "")]

# natural settled ride height (free, zero action)
a0 = np.zeros(env.action_space.shape[0], dtype=np.float32)
env.reset()
for _ in range(200):
    env.step(a0)
PIN_Z = float(d.body("torso").xpos[2])
print(f"natural settled torso z = {PIN_Z:.4f} (pinning trunk level here to calibrate)")


def settle(off, geoms, steps=250):
    """Pin trunk level at PIN_Z, hold ctrl=center+off, return min z over `geoms`."""
    mujoco.mj_resetData(m, d)
    ctrl = center + off
    for _ in range(steps):
        d.qpos[0:3] = [0.0, 0.0, PIN_Z]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[0:6] = 0.0
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
    d.qpos[0:3] = [0.0, 0.0, PIN_Z]; d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]; d.qvel[0:6] = 0.0
    mujoco.mj_forward(m, d)
    return min(float(d.geom_xpos[g][2]) for g in geoms)


grid = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
TARGET = 0.004
off = np.zeros(m.nu)

z0L = settle(off, hand_L); z0R = settle(off, hand_R)
print(f"paw z at midpoint center (offset=0): L={z0L:.4f} R={z0R:.4f}")


def calibrate_side(side_names, side_geoms):
    """Coordinate-descent this arm's ctrl offsets to plant THIS side's paw, then
    sparsify so only load-bearing joints keep an offset."""
    acts = [m.actuator(n).id for n in side_names]
    best = settle(off, side_geoms)
    for _pass in range(3):
        for a in acts:
            z_v = []
            for v in grid:
                o = off.copy(); o[a] = v
                z_v.append((settle(o, side_geoms), v))
            z, v = min(z_v)
            if z < best - 1e-4:
                best, off[a] = z, v
    for a in acts:
        if off[a] != 0.0:
            o = off.copy(); o[a] = 0.0
            if settle(o, side_geoms) <= TARGET + 0.002:
                off[a] = 0.0
    return settle(off, side_geoms)


bL = calibrate_side(fore_L, hand_L)
bR = calibrate_side(fore_R, hand_R)
print(f"paw z after calibration:            L={bL:.4f} R={bR:.4f}\n")
print("cpg_center_offsets = {")
for n, a in zip(fore, fore_act):
    if abs(off[a]) > 1e-9:
        print(f'    "{n}": {off[a]:+.2f},')
print("}")

# free-settle verification with the offset baked in
env2 = RodentCPGEnv(str(robolab.ASSETS_DIR / "rodent_scene.xml"))
env2.cpg_center_offsets = {n: float(off[a]) for n, a in zip(fore, fore_act) if abs(off[a]) > 1e-9}
# rebuild center with offset (mirror what __init__ will do)
c2 = 0.5 * (env2.model.actuator_ctrlrange[:, 0] + env2.model.actuator_ctrlrange[:, 1]).astype(float)
for n, v in env2.cpg_center_offsets.items():
    c2[env2.model.actuator(n).id] += v
env2._center = np.clip(c2, -1.0, 1.0)
env2.reset()
for _ in range(200):
    env2.step(a0)
hg = env2._hand_geoms
gm = env2.model
pzL = min(float(env2.data.geom_xpos[g][2]) for g in hg if "_L" in (gm.geom(g).name or ""))
pzR = min(float(env2.data.geom_xpos[g][2]) for g in hg if "_R" in (gm.geom(g).name or ""))
print(f"\nFREE-settle check with offset: front-paw z L={pzL:.4f} R={pzR:.4f} "
      f"(was ~0.020 at midpoint center)  torso z = {float(env2.data.body('torso').xpos[2]):.4f}")
