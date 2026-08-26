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

---

## 4. Reward BALANCE is the real blocker: "don't move" is optimal -- NEXT UP

- **Date:** 2026-08-21
- **Status:** Diagnosed, not yet fixed. This is the next run's focus (handoff note --
  a new chat should start here).

**Context / evidence**
- Two CPG-SAC runs compared on TensorBoard: SAC_6 (the failed magnitude-cost run, 3M)
  and SAC_7 (the rate-based `coff` fix from #3, gamma 0.99, ~1M/2M). Their `ep_rew_mean`
  and *every* reward-term curve overlap almost exactly -- EXCEPT `center_cost` (the one
  thing that changed: -0.021 -> -0.003).
- Why the coff fix didn't move behavior: `center_cost` is negligible vs the return
  (~0.017/step diff against ep_rew ~ -950). It was a correct conceptual fix, not a
  behavioral lever. Everything that drives behavior was identical between the runs.
- The damning reads from SAC_7's per-term plots:
  - `forward` ~ **-0.05** (i.e. ~0, slightly NEGATIVE) despite `forward_w=5`, because
    `gate ~ 0.27` (rarely in full stance) and `vx ~ -0.03` (drifting backward). The
    forward incentive barely reaches the policy.
  - `contact` ~ **-1** and `smooth` ~ **-0.9** DOMINATE the return.
  - `res_cost` ~ **-0.25** in BOTH runs -> residual still used as a sustained parallel
    controller (secondary issue, untouched).
- Net: the reward BALANCE makes **not moving optimal** -- locomotion earns ~nothing
  (gated-down forward) and costs a lot (contact + smoothness). Result is the bimodal
  stand-still/collapse basin, deeply negative return (~ -950).

**Decision (planned rebalance -- fresh run)**
1. Make `forward` actually reach the policy: raise `gate_floor` (so forward isn't
   attenuated to ~0 out of stance) and/or `forward_w`; sanity-check the speed cap +
   gating aren't jointly strangling it.
2. Cut the dominating penalties: lower `contact_w` and `smooth_w` (they currently
   outweigh all positive signal).
3. Then discipline the residual: raise `res_cost_w` (or gate it less cheaply) to stop
   the sustained parallel-controller use.
- Change these deliberately and few-at-a-time for attribution; lean on the per-term
  reward logs. Buffer-INcompatible (reward changes) -> fresh run.

**Run status at handoff**
- SAC_7: gamma 0.99, rate-based coff cost, ~1M of 2M, tracking SAC_6 into the same bad
  basin (expected). Fine to let it finish as confirmation; don't expect it to walk.
- Code committed through `575c7e7` (local ahead of origin by a few commits). Script 16
  has `--gamma` (default 0.99) and `--steps` (default 1e6) flags. The gamma=0.999
  curiosity run has NOT been done yet.

**Revisit when**
- After the rebalanced run: is `forward` now clearly positive, `vx` > 0, episodes
  longer? Then return to gait quality (rhythm-reward saturation, ROM) per #3.

---

## 5. Next-run docket: stance-gate-on-track + coarse-to-fine tracking kernel -- PARKED

- **Date:** 2026-08-24
- **Status:** Diagnostic reading banked + TWO levers PARKED for the NEXT run (deliberately
  NOT changed in the current `sigma(v*)` run, to keep that change attributable). Let the
  `sigma(v*)` run finish first; then take these one variable at a time.

**Context (the read that surfaced it)**
- Comparing two logged snapshots under the velocity-tracking reward (the
  `track = track_w * exp(-(vx-v*)^2 / 2 sigma(v*)^2)` well, gated by
  `ramp = gate_floor + (1-gate_floor)*g`, `gate_floor=0.5`):
  - Case A: `vx=0.117`, `v*=0.266` (moving, near target) -> raw well ~2.15, but
    `gate~=0` -> logged `track=1.08`.
  - Case B: `vx=-0.016`, `v*=0.241` (basically standing) -> raw well ~0.96,
    `gate~=1` -> logged `track=0.96`.
- So a genuinely-moving state and a near-standstill state collapse to nearly the same
  `track` (1.08 vs 0.96) even though the *raw* well differs ~2.2x. The gate eats
  exactly the factor separating them (verified: implied ramp ~0.5 vs ~1.0, matching the
  logged `gate` values).

**Primitive to bank (general)**
- A logged reward term is only interpretable **alongside its multiplicative factors**.
  `track` here is a *gated, nonlinearly-averaged* per-step quantity, NOT
  `f(mean vx, mean v*)`. When two reward curves look "surprisingly equal," suspect a
  hidden multiplier (a gate/mask/saturating nonlinearity) between the variable you read
  and the reward you log. Secondary shrink: Jensen -- mean of `exp(-err^2)` over a noisy
  episode sits below `exp(-(mean err)^2)`, biting hardest near the well's peak.

**Lever A -- change/replace/REMOVE the stance gate on `track`**
- The stance gate on the forward/track reward was introduced (#3/#4) to kill the old
  "topple-forward for cheap `vx`" exploit under the **capped-forward** reward. But the
  new **symmetric tracking well already penalizes overshoot structurally**, so that
  exploit is largely gone. Meanwhile the gate now hands ~full tracking credit to a
  planted near-standstill (Case B) while docking real locomotion to 50% (Case A):
  "move" barely out-scores "stand" (1.08 vs 0.96) AND costs more energy. This is a
  plausible contributor to the persistent `vx~=0` stall.
- **Verdict (evaluated 2026-08-24): it's a bandaid that has flipped to net-negative.**
  It patches a *proxy gap* (topple-for-`vx`) rather than the objective, and the honest
  terms it duplicates -- `upright` + fall termination, and now the symmetric well's
  overshoot penalty -- already handle that exploit. It now mis-shapes (subsidizes
  standing) and confounds the `track` signal. Legitimate only as *temporary scaffolding*;
  the error is making it permanent at fixed `gate_floor=0.5`.
- **Candidate fixes to weigh (pick ONE to test):** (a) raise `gate_floor` toward 1
  (ungate `track`); (b) **anneal** `gate_floor` 0.5 -> 1 over training (scaffold early,
  graduate off it); (c) move the gate onto a *stability* term instead of the tracking
  term (withhold "safely on support" credit while ballistic, not speed credit);
  (d) drop the track-gate entirely and lean on `upright`/`contact`/`energy`/termination.
- **Principle banked:** reward the causal quantity, not its correlate; patch a proxy-gap
  through the terms that already encode the true constraint ("don't fall"), not by
  bolting a multiplier onto the proxy. A hand-shaped term is fine as scaffolding you
  *remove*, suspect as structure you *keep*.

**Lever B -- coarse-to-fine (sum-of-two-Gaussians) tracking kernel**
- Replace the single Gaussian well with a mixture, both centered at `v*`
  (`e = vx - v*`):
  `r = track_w * (w_w*exp(-e^2/2*sig_w^2) + w_n*exp(-e^2/2*sig_n^2)) / (w_w + w_n)`,
  with `sig_w >> sig_n`.
- **Why:** a single Gaussian forces a trade-off between **capture range** (far-off states
  feel a pull) and **precision** (sharp incentive to nail `v*`). The mixture decouples
  them: the WIDE shoulder keeps a live gradient far out (kills the standstill barrier for
  ALL `v*`), the NARROW peak sharpens the pull near target (precision). Sum of two
  same-center Gaussians is still unimodal -> monotone in `|e|`, so NO spurious optima.
- **Relation to `sigma(v*)`:** orthogonal axis, and the mixture can SUBSUME it. `sigma(v*)`
  varies one scale *across commands* to keep the barrier reachable; a single wide `sig_w`
  (~0.30) does that uniformly with no schedule. And it CURES `sigma(v*)`'s hidden cost:
  a wide well at high `v*` makes high-speed tracking sloppy (a 0.1 m/s error at `v*=0.5`,
  `sigma=0.3` costs only ~5% of peak) -- the narrow peak restores precision at every speed.
- **Suggested A/B:** same `track_w`/band, `sig_w=0.30`, `sig_n~=0.10`, `w_w:w_n ~= 0.5:0.5`
  normalized. Single Gaussian is the `w_n=0` special case (clean superset).
- **Caveats:** normalize the peak (`/(w_w+w_n)`) so `track` doesn't silently rescale vs
  `energy`; keep `sig_n` moderate (~0.10, not 0.02) -- an over-sharp peak = stiff,
  high-curvature TD targets that can re-inject critic variance we fought hard to kill.
- **Primitive banked:** represent a shaped reward as a **superposition of kernels at
  multiple scales** (the K=2 case of an RBF/multiresolution basis for reward design) --
  the same change-of-basis move used on the action space, applied to the reward landscape.

**Lever C -- remove/relocate the `hand` bonus (SAME disease as the gate)**
- `hand` pays **+0.25 per front-paw geom on the floor** -- a flat carrot for *contact*,
  not for *propulsion*. Cheapest way to harvest it: plant the forelimbs as **passive
  struts** and hold them (zero propulsive work, full bonus, lower topple risk). That is
  exactly the **"forelimbs as stabilizers, not propulsors"** gait we're fighting, and it
  matches the probe (FORE `amp_pos`~0.39 low, FORE `|res|`~0.78 high: front paws are
  *positioned* by the aperiodic residual, not oscillated to push).
- **Same bandaid class as the gate:** both reward a *correlate of support/contact*
  instead of the *causal push-off*. Under `track`+`energy`, real quadruped propulsion
  should emerge only if planting-and-pushing hits `v*` more cheaply; the contact carrot
  short-circuits that by paying for mere contact. (See #6: this is really a *morphological
  PRIOR* -- "use four limbs" -- injected through the reward channel; its right home is the
  body/action-space coupling, not a per-contact payout.)
- **Test:** remove/anneal `hand`, lean on `track`+`energy`. Either forelimbs start
  contributing propulsion (only reason left to plant them is to go faster/cheaper) or they
  lift into a biped-ish gait -- both informative. The bonus currently makes "plant & rest"
  a free lunch that blocks that fork.

**Cheap trims to bundle LATER (low-risk, do after the big levers)**
- `upright` (0.2*xmat[8]): termination already forces `xmat[8]>0.5`, so we *pay* for
  posture we already *require*; worse, it rewards a VERTICAL torso while a running rat
  carries a low near-horizontal spine -> gently fights natural posture. Candidate: -> 0.
- `alive` (0.5): the honest "don't fall" is lost future return on termination; a flat
  per-step bonus mainly inflates the stand-still basin (#4). Keep as *early* exploration
  scaffolding, **anneal** it out.
- `contact` (-0.75/non-paw geom): belly/bone drag does friction WORK, so the now-live
  `energy` term (0.13, no longer ~off) likely already penalizes it. Test-for-redundancy,
  then trim.

**FIRST post-run action -- gate diagnostic (decide Lever A from data)**
- Live observation (2026-08-24): in the command-line dumps, `gate ~ 0` precisely at the
  HIGHER `vx` the policy reaches -> `g` looks ANTI-correlated with speed, i.e. the stance
  gate is docking the fast locomotion we want (direct support for the Lever-A hypothesis).
- Diagnostic (run on the finished policy BEFORE any change): roll out and log per step
  `vx, v*, g` AND g's four factors `s_up, s_h, s_feet, s_hands` (+ their raw inputs:
  torso `xmat[8]`, torso `z`, `n_foot`, lowest front-paw height). Then bin by `vx` (or
  `v*`) and see WHICH factor collapses `g` at speed:
  - `s_hands`->0 : forelimbs lifting at speed (hand-plant requirement fights a fast gait),
  - `s_feet`->0  : hind aerial/flight phase (gate demands both hind paws down),
  - `s_h`/`s_up`->0 : torso bobbing lower / pitching during push-off.
- The culprit factor picks the fix: e.g. `s_hands` the killer -> the gate's hand
  requirement is the problem (ties to Lever C `hand`); `s_feet` the killer -> the "both
  hind down" demand is incompatible with a real gait's flight phase. Quantify corr(`g`,`vx`).

**PRELIMINARY gate probe (5.05M checkpoint, 2026-08-24) -- course-correction**
- Ran the probe on the live `sigma(v*)` checkpoint (5.05M/6M) AND the SAC_16/4M model, greedy
  + stochastic (`scripts/_gate_diag_tmp.py`, CPU so it doesn't touch training).
- **Both policies are STALLED at `vx~=0`** (sigma(v*): greedy vx mean 0.000, max 0.19, `g~=0.99`;
  SAC_16: vx max ~0.25, `g~=0.75`). The velocity-tracking regime is collapsing to a
  **planted step-in-place basin** (limbs move, body doesn't translate; gate ~1 so it collects
  near-full `track` credit for standing). **`sigma(v*)` has NOT lifted `vx` off 0 by 5.05M.**
- Gate-vs-`vx` is **inconclusive on these checkpoints because they don't move**: weak and
  contradictory (`sigma(v*)`: corr(vx,gate)=-0.10 via `s_up`/`s_hands`; SAC_16: corr=+0.29,
  forward=more upright). The dramatic `gate->0`-at-speed came from a genuinely MOVING gait
  (SAC_15-era, vx 0.4-0.76) that PREDATES the velocity-tracking obs (added `cmd_norm`), so it
  can't be loaded into the current env (obs-dim mismatch) for a clean at-speed characterization.
- **Implication:** the pressing blocker is the reward BALANCE collapsing to standstill (gate
  subsidizes standing + energy pushes back on motion), not (yet) the gate docking a fast gait.
  Strengthens #5's Lever-A/energy hypotheses. Full gate-at-speed probe needs a moving policy
  (this run recovering, or testing the gate change directly).

**DECISIVE follow-up (2026-08-24) -- gate quantitatively inverts stand-vs-move**
- Reconciled the eval-vs-dump gap: the GREEDY policy pins `dfreq=-0.992` (CPG frequency at
  MINIMUM -> oscillator off -> stands; `ep_len=2000`, never falls, `vx~=0` at EVERY command
  incl v*=0.46 -> NOT command-following). Training dumps are ~instantaneous snapshots of the
  STOCHASTIC collection catching exploration transients into a moving+pitched state (where
  `gate->0`). Both true: converged policy stands; exploration briefly moves and gets gated.
- **Per-step reward, the two equilibria (moving row = the live dump v*=0.35, vx=0.196, g=0.04):**
  - STANDING (v*~0.29, vx~0, g~1): track~0.99 + alive .5 + upright .2 + hand .25, ~0 costs
    -> ~1.94/step.
  - MOVING (dump): track 1.23 (GATED; ungated 2.37) + .5 + .106 + .25 - energy .231 - res .354
    - ctrl .093 - smooth .041 -> ~1.37/step.
  - **Standing beats moving by ~0.6/step -> the policy correctly learned to STAND.**
- **The gate is the swing factor:** it cuts the moving state's track 2.37 -> 1.23 (-1.14),
  a hit LARGER than the moving state's entire energy+res+ctrl+smooth cost (-0.72). Ungate the
  moving state -> ~2.5 > 1.94 -> MOVING becomes optimal. The gate single-handedly inverts the
  preference. This is the quantitative confirmation of Lever A.
- **sigma(v*) backfired (keep, but it's not the fix):** widening sigma RAISED standing's reward
  (at v*=0.29, vx=0: track 0.045 at sigma=0.1 -> 0.99 at sigma~0.195), DEEPENING the standstill
  basin. Not harmful (moving UNGATED still dominates), but it cannot escape the stall because
  the GATE, not the well width, is the blocker.
- **Verdict:** promote **Lever A (gate)** to THE next-run change; it is now data-proven, not
  hypothesized. sigma(v*) is a keep-or-revert side issue. Probe: `scripts/_gate_diag_tmp.py`.

**Item #2 post-run -- cross-term reward interaction matrices (see also #6)**
- Marginal vector FIRST: per-term mean/std of the signed contributions in `_reward_terms`
  (a near-zero-variance term can't shape -- how `center_cost` was exposed in #4).
- Matrix 1 (cheap, observational): Pearson+Spearman corr of per-step term contributions, over
  BOTH on-policy and action-perturbed/buffer states (on-policy only sees the current basin).
- Matrix 2 (design-time adjacency): which terms share inputs / multiply (gate x track,
  energy/smooth/control share action mag, contact/hand share contacts) -- explains Matrix 1.
- Matrix 3 (the real one -- gradient conflict): mean cos(grad_a r_i, grad_a r_j) over sampled
  states via finite-diff on the action (O(action_dim) sim steps/state). `<0` = compete
  (Pareto tension), `>0` = redundant/synergistic. Feeds F/B/P: a hand term whose grad chronic-
  ally opposes `track`/`energy` = bandaid; one parallel to another w/ low added variance =
  redundant (test contact ~ energy). Cosine is scale-invariant (unlike Matrix 1).
- Matrix 4 (expensive, causal): d(metric_k)/d(weight_j) via short retrains -- gold standard,
  already approximated by the one-lever-per-run protocol; skip unless a weight is contentious.
- CAVEAT: all are policy-conditioned (describe the landscape from the current basin); recompute
  after any behavior change.

**Revisit / sequencing**
- Trigger: the `sigma(v*)` run resolves -- does `vx` lift off 0? Read outcomes:
  - Still stalls -> **Lever A** (gate) is the highest-confidence single lever. Because the
    gate and **Lever C** (`hand`) are the SAME bandaid class (support/contact correlate,
    not push-off) and jointly explain the low, hind-driven, forelimb-strut gait, consider
    "de-bandaid the support terms" as the theme: gate first, then `hand` (isolate each for
    attribution), or evaluate them as a paired change if impatient.
  - Locomotes but tracks high speeds loosely -> **Lever B** (mixture kernel), which also
    subsumes `sigma(v*)`.
  - `hand`/forelimb still strut after the gate -> **Lever C** (`hand`).
- Cheap trims (`upright`->0, `alive` anneal, `contact` redundancy) are low-risk; bundle
  them AFTER the big levers, still one attributable change at a time.
- Both are buffer-INcompatible (reward shape changes) -> fresh run or `--no-buffer` resume.
- Change ONE lever per run for attribution; lean on the per-term reward logs (watch
  `gate`, `track`, `vx`, forelimb `amp_pos`/`|res|`, and critic loss).

**Consequences**
- No code change yet; parking-lot entry so the reasoning isn't lost. Keeps the current
  run a clean one-variable experiment (`sigma(v*)` only).

---

## 6. Reward audit lens: Fundamental / Bandaid / PRIOR -- behavior = reward x priors

- **Date:** 2026-08-24
- **Status:** Framework / reusable primitive (no code). The lens used to build the #5
  docket; recorded so future audits reuse it.

**The core idea**
- Expressed behavior is the product of TWO things, not one:
  `behavior = f(external reward landscape, internal priors)`.
- The **reward landscape** is the objective + costs the animal actually experiences.
- The **priors** are everything the animal brings *before* reward: its **body**
  (morphology, mass, joint limits, tendon/muscle mechanics), its **brain**
  (spinal CPGs, reflexes, neural architecture, tuning curves), and **evolution/development**
  (innate motor primitives, default posture/tone, instincts). Priors are *inductive
  biases / structure*, NOT objectives.
- Failure mode we keep hitting: trying to buy a PRIOR with REWARD. That produces a
  "bandaid" -- a hand payout for a *correlate* of the behavior a prior would have
  produced structurally, which the policy then satisfies via a free-lunch equilibrium
  (stand & collect `alive`/`gate`; plant & collect `hand`).

**Three-way classification (audit every term this way)**
1. **F = Fundamental reward.** Part of the true objective/cost the animal experiences.
   Belongs in the reward. Test: would a real rat's fitness/drive actually depend on it?
2. **B = Bandaid.** A hand-crafted patch on the gap between a proxy and the objective;
   creates free-lunch equilibria. Remove or anneal; prefer fixing the proxy or installing
   the missing prior.
3. **P = Prior.** A true fact about the animal's body/brain/evolution. Its right home is
   **structure** -- the body model (passive stiffness, joint ranges, actuator low-pass /
   activation dynamics), the **action-space basis** (CPG, bilateral symmetry, duty warp),
   the **network**, or the **initial-state distribution** -- NOT a reward term. If it is
   currently a reward, that's a prior injected through the wrong channel: RELOCATE it.

**This project's terms, classified**
- `track` -> **F** (objective). Its *gate* -> **B**, and the gate's intent ("propulsion
  needs support") is a **P** expressed as reward.
- `energy` -> **F** (real metabolic cost; also grounded in a metabolic **P** -- correctly
  a reward because the animal *does* experience it as a drive).
- termination (fall/flip) -> **F** (the constraint; the honest "don't fall").
- `hand` -> **B** now; underlying "be a quadruped / use four limbs" is a morphological
  **P** -> belongs in body + limb coupling, not a contact payout.
- `upright` -> **B**/**P**: default erect-ish posture is a **P** (skeleton + muscle tone +
  vestibular reflexes) -> belongs in passive body stiffness + termination, not a payout.
- `alive` -> **B**/scaffolding: survival is a **P** (instinct) but in RL it is encoded by
  discounting + termination, not a per-step bonus.
- `contact` (anti belly/bone skid) -> **B**, largely redundant with `energy` (**F**) once
  `energy_w` is live.
- `smooth` -> **P** mis-homed: real muscles low-pass activation (finite bandwidth). Its
  honest home is **actuator dynamics** (first-order activation / low-pass on `ctrl`), not a
  reward penalty. Keep as reward for now (cheap), but the principled move is to relocate.
- `control_cost` -> weak **F/P** (neural/metabolic cost of drive); low priority.
- `res_cost` / `center_move` -> **P** as reward: "the CPG is primary, feedback is
  corrective, posture-shift is costly." Could instead be structural (limit residual
  authority / EMA rate in the action space). Already carefully re-derived (#3); fine for
  now.
- ALREADY correctly placed as **P = structure** (not reward): the **CPG** action space
  (spinal rhythmicity), the **bilateral-symmetry basis** (body symmetry), the **duty
  warp** (stance/swing asymmetry), the **rat MJCF** morphology, and **phase/joint
  randomization** on reset (a prior over initial conditions). These are the model doing
  its job -- priors in their right channel.

**Decision rule (for any candidate term)**
- True objective/cost the animal experiences? -> **F**, keep in reward.
- Fact about its body/brain/evolution? -> **P**, relocate to structure; don't pay for it.
- Neither -- a proxy patch that enables a free lunch? -> **B**, remove/anneal; fix the
  proxy or install the prior instead.

**Primitive banked (general, transferable)**
- When a behavior won't emerge, you have TWO channels: reshape the **reward** or install
  the **prior** (structure/inductive bias). Installing the prior is usually more robust
  and more biologically honest than paying for a proxy of it. Bandaids are the symptom of
  reaching for the reward channel when the prior channel was the right one. Audit: *reward
  the causal quantity; encode body/brain facts as structure; delete the rest.*

---

## 7. Asymmetric tracking reward -- kill the standstill freebie

- **Date:** 2026-08-25
- **Status:** DONE (implemented in `RodentRunEnv._get_reward`). Fresh run under way to test.

**Context**
- Ungating `track` (#5, Lever A) did NOT produce locomotion. SAC_16 (gated) and the
  ungated SAC_18 run converged to the SAME stable standstill (ep_len 2000, `dfreq`
  pinned at min, `vx~=0`); their TB traces were near-identical except the `track`
  *number*. That falsifies "the gate created the standstill basin."
- Root cause: a plain Gaussian well centred at `v*` pays real credit AT `vx=0`, because
  `vx=0` sits only ~1-1.6 sigma from `v*`. Measured freebie: `track~=0.75-1.8` of 3 for
  ZERO effort (worst at low `v*`, where the well is closest to 0). Gated or ungated,
  standing collected it -> STANDING was a local optimum. The gate only ever docked
  *moving*, never *standing*, so removing it changed scoring, not the basin.
- Corollary: the earlier "wider sigma / sum-of-two-Gaussians" idea would BACKFIRE -- a
  wider shoulder pumps MORE credit into `vx=0`, feeding the freebie. Capture range and
  no-freebie pull sigma in opposite directions; a symmetric well centred at `v*` can't win both.

**Decision**
- Make `track` ASYMMETRIC / zero-at-standstill by subtracting the well's own value at
  `vx=0`:  `r_track = track_w * (exp(-(vx-v*)^2/2s^2) - a0)`, `a0 = exp(-v*^2/2s^2)`.
  Now `vx=0 -> 0`, peak `track_w*(1-a0)` at `v*`, and moving BACKWARD (`vx<0`) or
  OVERSHOOTING past `2v*` goes NEGATIVE (both penalized). Live uphill gradient off standstill.
- NOT renormalized by `1/(1-a0)`: that restores a fixed peak but blows up as `v*->0`
  (`a0->1`). Accept the gently sagging low-`v*` peak (`track_w*(1-a0)`) to avoid the
  amplification. F/B/P: the symmetric well's fat tail at 0 was an accidental free lunch
  (a **B** artifact on the **F** objective); this deletes it without adding a bandaid.
- Two-part trap named: (i) the freebie (local optimum) + (ii) a "stumbling valley"
  (stance->walk goes downhill first via slip/`res` cost). Zeroing the freebie tilts BOTH
  -- partial forward motion now out-scores standing, turning the barrier into a gradient.

**Revisit when**
- Fresh run is the clean test (all prior runs used the freebie reward): does the
  corrected reward, from scratch, produce a gait? Watch `track` going POSITIVE (moving)
  vs hovering ~0 (parked), and `vx` lifting off 0. If it still parks, the freebie wasn't
  the whole story (look harder at the stumbling valley / exploration).
- If a resume from the good-stance model is tried instead, pair with an `--ent-coef`
  re-warm: the parked policy is near-deterministic (alpha~=0.006) and must be pushed to
  sample the forward motions that feel the new gradient.
