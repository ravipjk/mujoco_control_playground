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
    env.gate_forward = True     # forward reward *= stance gate g
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
    env.gate_rhythm = True
    # NOTE: this env has an AUGMENTED obs (per-paw contact force) and a SYMMETRY-
    # REDUCED action space (97-dim: dfreq, duty, amp_sym(19), phase_tied(19),
    # res(38), coff_sym(19)), so its checkpoints are NOT compatible with the older
    # 153-dim CPG models -- fresh run only, no resume across the change.
    return env


# Reward-term channels to log (each surfaced by the env in info each step).
# smooth_{freq,amp,phase,res,coff} break the aggregate `smooth` into its action
# groups so we can see which sub-term dominates before cutting its weight.
CPG_TERM_KEYS = ["forward", "alive", "control_cost", "upright", "contact",
                 "hand", "energy", "smooth", "phase", "rhythm", "res_cost",
                 "center_cost", "bad_contacts", "hand_contacts", "foot_contacts",
                 "gate", "hand_prox", "vx",
                 "smooth_freq", "smooth_duty", "smooth_amp", "smooth_phase",
                 "smooth_res", "smooth_coff"]
