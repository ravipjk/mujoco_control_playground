# Design Notes

A lightweight decision log / parking lot for architecture choices. Each entry
records *context*, the *decision*, its *status*, and *when to revisit* -- in the
spirit of ADRs (Architecture Decision Records), kept intentionally short.

The point: capture reasoning so future-you (and future agents) inherit the "why,"
and so good ideas are parked without being started prematurely.

---

## 1. Sensor / actuator (encoder / decoder) layer -- DEFERRED (seam now, layer later)

- **Date:** 2026-08-08
- **Status:** Deferred. The *seam* is being built now; the *machinery* is parked.

**Context**
- Toward a NeuroAI adaptive neural controller, we want one controller to work
  across platforms (car, cartpole, walker, rodent) and to support experiments
  where sensors/actuators are lesioned or added.
- Biologically this is the separation of afferent (sensory transduction) and
  efferent (motor mapping) pathways around a central computation. In later phases
  the encoder becomes *modeled science* (e.g. RatInABox place/grid/head-direction
  cells), not mere plumbing.

**Decision**
- Do **not** build explicit, composable `Sensor`/`Actuator` objects yet. We have
  one platform and zero controllers; abstracting from a single example tends to
  encode the wrong assumptions ("write it twice before you abstract").
- Do build the **seam** now, at near-zero cost:
  - `Controller` stays minimal and platform-agnostic: `act(obs) -> action`.
  - `MujocoRobotEnv` funnels *all* observation construction through one
    overridable `_get_obs()` and *all* action application through one
    `_apply_action()`. These two methods are the future sensor/actuator joint.
  - Expose named / structured observation & action spaces so a later encoder can
    reason about specific channels (e.g. "lesion the head-direction input").
- Compose future layers using Gymnasium's `ObservationWrapper` / `ActionWrapper`
  pattern (adopted in Phase 5) rather than a bespoke framework.

**Revisit when**
- A second platform exists (Phase 4 walker) or at the NeuroAI bridge (Phase 6),
  where we'll have 2-3 concrete cases to generalize the abstraction from.

**Consequences**
- Minimal cost now; a clean joint to slot the layer into later.
- Env code must route obs/action through the two seam methods -- no scattered
  `data.qpos` / `data.ctrl` access elsewhere.

---

## 2. Heading-invariant locomotion (egocentric reward + yaw randomization) -- PARKED

- **Date:** 2026-08-20
- **Status:** Parked (good idea, deliberately NOT started yet).

**Context**
- The rodent forward reward is `vx` along **world +x** (`RodentRunEnv._get_reward`),
  and reset randomizes joint angles + the CPG start phase but *not* the free-joint
  torso pose. So the rat always starts facing ~+x and only ever learns to go one way.
- Randomizing initial **yaw** for robustness is tempting, but with a world-frame
  forward reward it silently changes the task: the rat must first *turn* to +x, i.e.
  it bundles "learn to turn" into "learn to walk." That's a harder, different problem
  and would muddy the current gait work.

**Decision**
- Keep the world-frame reward and fixed initial heading **for now** (isolate the gait).
- When we want heading-invariance, do it properly as a pair:
  1. Make the forward reward **egocentric** -- reward speed along the torso's own
     forward axis (project COM velocity onto the body x-axis via `xmat`), not world x.
  2. *Then* randomize initial yaw (and later add explicit turn/heading-command tasks).
- Both are buffer-INcompatible (they change the reward), so they belong in a fresh run,
  not a seamless resume.

**Revisit when**
- The straight-line gait is stable/robust (topple rate low), and we want turning,
  target-following, or navigation (natural lead-in to the RatInABox / NeuroAI phase,
  where head-direction and egocentric frames matter anyway).

**Consequences**
- Unlocks turning/navigation and is the honest frame for a place/head-direction
  encoder later; costs a from-scratch run (no buffer reuse) and a harder task.

---

## 3. Forelimb "stiffness" fix: gated rhythm + per-paw obs + movable center -- ADOPTED

- **Date:** 2026-08-20
- **Status:** Adopted (implemented in `RodentRunEnv` / `RodentCPGEnv`; enabled via
  `make_cpg_env`). Ships as ONE fresh run -- buffer/obs/action all change.

**Context (diagnosis)**
- The CPG forelimbs looked "unnaturally stiff." A ROM probe of the 2.3M checkpoint
  showed the big joints actually use a *moderate* fraction of range (shoulder ~57%,
  elbow ~41%, wrist ~61%), comparable to the hindlimb (knee ~34%) -- so it is NOT a
  forelimb-specific mechanical block. Two curable roots instead:
  1. **Fixed center = frozen home pose.** `ctrl = center + osc + residual` with a
     *constant* `center`. When amp/residual are modest the limb servos back to the
     SAME pose every cycle -> wiggles around a nailed-down setpoint (reads as stiff),
     rather than *repositioning* the posture like a real rat. The residual is the only
     setpoint-shifting tool and it's bounded + taxed + fast, so it "fights" the
     oscillation instead of holding a pose.
  2. **Saturating rhythm reward.** `rhythm = tanh(limb_speed/scale)` saturates and
     `smooth`/`energy` penalize big moves, so nothing pays for larger, posture-varying
     excursions -> the policy settles for a small efficient wiggle.

**Decision (three coupled changes, one fresh run)**
- **Q1 -- gate the periodic rewards by stance** (`gate_rhythm`): multiply
  `phase`/`rhythm` by the SAME stance ramp `gate_floor + (1-gate_floor)*g` used for
  `forward`. Complements the residual cost (`gate_res`, cheap OUT of stance): one gate
  `g` wired to a reward (rhythm ~ g) vs a cost (residual ~ 1-g) = "recover
  aperiodically, then stride periodically." Also closes a real exploit (earning rhythm
  by flailing while fallen).
- **Per-paw contact-force obs**: append `[FL,FR,HL,HR]` ground normal force
  (weight-normalized) to the CPG obs. This is a genuine **afferent** signal the rat
  lacked (it couldn't feel its feet touch). Deliberately do NOT put the gate `g` in
  the obs: `g` is a derived reward construct, not a sensor, and (given contacts +
  height + orientation) is redundant. If `g`'s critic-variance benefit is ever wanted,
  the right home is an **asymmetric critic**, not the shared observation.
- **Q3 -- slow, policy-controlled center offset** (`coff` action group): a 4th action
  group low-passed (EMA) into a slow postural setpoint, `ctrl = center + coff_ema +
  osc + residual`. Gives the policy an honest channel to *reposition* the forelimb
  pose (directly attacks root #1), decoupled from the fast/taxed residual. Guardrails
  keep it POSTURAL not a parallel direct-drive: modest `coff_gain` (0.3), slow
  `coff_beta` (0.01, tau~2 s).

  - **Cost semantics -- CORRECTED (2026-08-21).** First pass taxed the offset
    *magnitude* (`Σ|coff_ema|`), i.e. a spring pulling the center back to the ctrl
    midpoint -- which FIGHTS the whole idea of a movable center. A first CPG run
    (1M->3M, gamma 0.99) confirmed the muddle: no locomotion (`vx~=0`), bimodal
    stand-still/collapse, with `coff` slammed near-max (`coff_mean~0.66`). Diagnosis
    correction: a large *steady* `coff` is FINE (it's the intended oscillation center),
    and it does NOT drown the CPG (osc happens *around* the center); the real
    rhythm-underminer was the sustained **residual** (`res_mean~0.68`), a separate
    issue. Fix: charge the **rate of change** of the setpoint
    (`coff_move_w * Σ|d coff_ema|/half`), not its level. Integrated over a move this
    equals the offset magnitude -> a ONE-TIME "cost to shift posture," then FREE to
    hold -- "pay to change the gait's posture, ride the gait for free." Holding a
    non-zero center within a stable gait costs ~0 (verified: cost decays to ~2e-5 while
    a large setpoint is held). `smooth_w_coff` set to 0 (the move-cost owns
    "reposition cost"; the EMA already rate-limits the command).

**Rejected / not done**
- **Q2 (fade `osc` by `g` at the actuation level):** dropped. Q1 removes the reward
  reason to keep amplitude high out of stance, so the policy can quiet its own limbs
  via `amp->0` -- align incentives > hard-code the dynamics. (Kept as a mental
  fallback only; not in the code.)
- **Widened initial-joint randomization:** the strong position servos wash out
  init-angle noise within a few steps (the servo target doesn't depend on it), so it's
  secondary to the CPG phase randomization already in place; not changed here.

**Revisit when**
- Post-run ROM probe: did excursions grow / posture vary (coff_ema nonzero and useful)?
  If limbs still look stiff, the next lever is the **saturating rhythm reward**
  (unsaturate `rhythm_scale` or add an explicit stride-ROM bonus) -- root #2, untouched
  here to keep this bundle attributable.

**Consequences**
- Fresh run only: obs dim (+4 per-paw), action dim (`1+4*nu`), and reward all change
  -> resume scripts 18/19 can't cross this boundary; probe 17 updated for `1+4*nu`.
- More of the reward is now multiplicative on `g` (forward, residual, rhythm/phase);
  `gate_floor` keeps a live gradient everywhere.
