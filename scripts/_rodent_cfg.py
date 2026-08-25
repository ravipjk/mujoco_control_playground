"""Shared config for the CPG rodent locomotion experiment.

Single source of truth for the stance-gate reward config, imported by BOTH the
trainer (16_train_rodent_cpg_sac.py) and the resumer (18_resume_rodent_cpg_sac.py)
so a resume can never drift from the run it continues -- fine-tuning must use the
EXACT reward the original run was trained on.

(Filename starts with '_' so it's a plain importable helper, not a numbered,
runnable experiment script. Scripts import it because scripts/ is on sys.path
when you run `uv run python scripts/<name>.py`.)
"""
import robolab
from robolab.envs.rodent import RodentCPGEnv

XML = str(robolab.ASSETS_DIR / "rodent_scene.xml")

# True  = quadruped: the stance gate ALSO requires front-paw contact.
# False = biped: stance = upright + height + hind feet only (keeps the 2-leg gait).
GATE_USE_HANDS = True


def make_cpg_env():
    """RodentCPGEnv with the stance-gate experiment config applied."""
    env = RodentCPGEnv(XML)
    # LEVER A (2026-08-24): UNGATE the tracking reward. Data-proven bandaid -- the gate
    # diagnostic (DESIGN_NOTES #5) showed the stance gate single-handedly makes STANDING
    # out-score MOVING: it cut a moving state's track 2.37 -> 1.23 (-1.14/step), a hit
    # bigger than the entire cost of moving (energy+res+ctrl+smooth ~= -0.72), so the
    # policy learned to pin dfreq at MINIMUM and stand (vx~=0). Ungating restores moving's
    # 2.37 track -> moving becomes optimal. The old anti-topple justification is now
    # carried structurally by the SYMMETRIC tracking well (overshoot penalized) + fall
    # termination + upright, so the track-gate is redundant. Keep gate_res=True (the
    # residual should still be cheap while stumbling, expensive once planted); g is still
    # computed + logged for diagnostics and the residual gate.
    env.gate_forward = False    # was True -- track no longer multiplied by the stance gate
    env.gate_res = True         # residual cost *= g (cheap while getting up/stumbling)
    env.gate_use_hands = GATE_USE_HANDS
    # Gate FLOOR + soft hands: turn the quadruped gate from a wall (g~0 forever,
    # dead forward gradient) into a ramp. forward/residual are attenuated, not
    # annihilated, out of stance, and s_hands is a smooth paw-height proximity so
    # there's a gradient leading the front paws down into the quadruped stance.
    env.gate_floor = 0.5        # forward earns 50% out of stance, 100% in stance.
                                #   Raised 0.25 -> 0.5 in the DESIGN_NOTES #4 rebalance:
                                #   forward was reaching the policy at ~0 (gate ~0.42,
                                #   vx ~0), so keep more of its GRADIENT alive out of
                                #   stance to pull the rat toward moving.
    env.gate_hands_soft = True  # s_hands = front-paw-height ramp, not a contact count
    # Q1: also gate the PERIODIC rewards by stance, so rhythm/phase pay only once
    # planted (complement to gate_res: residual is cheap OUT of stance). Together
    # this is the "recover aperiodically, then stride periodically" curriculum.
    env.gate_rhythm = True   # (no-op now that rhythm/phase rewards are removed)
    # Long-ish episodes so each spans a full slow speed-command period (T in [20,40] s):
    # 2000 steps * ctrl_dt 0.02 = 40 s >= T_max, so even the slowest command completes
    # one full min->max->min sweep within an episode (a full sine period covers the
    # whole [v_lo,v_hi] excursion regardless of start phase; SAC's buffer stitches the
    # rest). Chosen over 60 s to get MORE resets/step -> more startup-transient reps,
    # hedging the launch-topple basin. Decoupled from gamma (0.995 -> ~4 s horizon).
    env.episode_length = 2000
    # NOTE: this env has an AUGMENTED obs (per-paw contact force) and a SYMMETRY-
    # REDUCED action space (97-dim: dfreq, duty, amp_sym(19), phase_tied(19),
    # res(38), coff_sym(19)), so its checkpoints are NOT compatible with the older
    # 153-dim CPG models -- fresh run only, no resume across the change.
    return env


# Reward-term channels to log (each surfaced by the env in info each step).
# smooth_{freq,amp,phase,res,coff} break the aggregate `smooth` into its action
# groups so we can see which sub-term dominates before cutting its weight.
CPG_TERM_KEYS = ["track", "v_target", "alive", "control_cost", "upright", "contact",
                 "hand", "energy", "smooth", "res_cost",
                 "center_cost", "bad_contacts", "hand_contacts", "foot_contacts",
                 "gate", "hand_prox", "vx",
                 "smooth_freq", "smooth_duty", "smooth_amp", "smooth_phase",
                 "smooth_res", "smooth_coff"]
