import mujoco
import numpy as np
from gymnasium import spaces

from robolab.core.env import MujocoRobotEnv


class RodentStandEnv(MujocoRobotEnv):
    """dm_control rat, wrapped as a free-floating body on a floor.

    We added a <freejoint/> to the torso, so the state layout is now:

      qpos = [ x, y, z,  qw, qx, qy, qz,  <67 joint angles> ]   (len 74)
               |__pos__| |____quat_____|
      qvel = [ vx, vy, vz,  wx, wy, wz,   <67 joint vels>   ]   (len 73)
               |__lin__|  |__angular__|

    So: torso world height is qpos[2]; torso orientation is the quaternion
    qpos[3:7]; internal joints start at qpos[7] / qvel[6].

    This base env is just "stand there": zero (or held) control should keep the
    rat upright (the 38 actuators are position servos, so ctrl=0 holds the
    default pose). Reward is 0 here; StandEnv only exists to (a) define the
    observation and (b) define what "fell over" means. RodentRunEnv adds the
    locomotion reward on top.

    Measured facts (default pose, zero control):
      torso z starts ~0.065 m, settles ~0.047 m; torso "up-z" stays ~1.0.
    """

    def _get_obs(self):
        # TODO(you): proprioceptive observation.
        # Absolute world x, y (qpos[0:2]) are NOT useful for a policy (they say
        # *where on the floor* the rat is, which shouldn't change how it walks),
        # so drop them. Keep everything else:
        #   - qpos[2:]  -> torso height, orientation quat, all joint angles
        #   - qvel[:]   -> all 73 velocities (torso lin/ang vel + joint vels)
        # Concatenate them into one float32 vector and return it.
        return np.concatenate([self.data.qpos[2:], self.data.qvel[:]]).astype(np.float32)

    def _reset_model(self):
        # TODO(you): mj_resetData (called by the base reset) already puts qpos at
        # the model default -> the upright standing pose. For robustness during
        # RL, add a little noise so the policy doesn't overfit one exact pose:
        #   self.data.qpos[7:] += small uniform noise   (joints only; leave the
        #                                                 free-joint pose alone)
        #   self.data.qvel[:]  += small uniform noise
        # Use self.np_random (Gymnasium's seeded RNG) so runs are reproducible.
        # Keep the noise tiny (e.g. +-0.02) so it starts upright, not collapsing.
        self.data.qpos[7:] += self.np_random.uniform(low=-0.02, high=0.02, size=self.data.qpos[7:].shape)
        self.data.qvel[:] += self.np_random.uniform(low=-0.02, high=0.02, size=self.data.qvel[:].shape)
        

    def _get_terminated(self):
        # TODO(you): end the episode if the rat has fallen or flipped.
        #
        # Height: torso world height -> self.data.body("torso").xpos[2]
        #   Standing is ~0.047 m; call it "fallen" below ~0.03 m.
        #
        # Orientation (the new/tricky bit): self.data.body("torso").xmat is the
        # 3x3 rotation matrix (world <- body) flattened row-major (length 9).
        # Its 3rd COLUMN is the torso's local +z axis expressed in world coords;
        # the world-z component of that axis is xmat[8]. Interpret it as:
        #     xmat[8] = +1 -> perfectly upright
        #              =  0 -> lying on its side
        #              = -1 -> upside down
        # Call it "flipped" when xmat[8] < ~0.5 (more than 60 deg off vertical).
        #
        # Return bool(fallen or flipped)  -- native bool, not numpy.bool_.
        fallen = self.data.qpos[2] < 0.03
        flipped = self.data.body("torso").xmat[8] < 0.5
        return bool(fallen or flipped)


class RodentRunEnv(RodentStandEnv):
    """Rat set up for RL locomotion. Inherits the observation and the fall/flip
    termination from RodentStandEnv; adds a forward-progress reward.

    Forward progress is measured as center-of-torso displacement per control
    step (x_after - x_before) / ctrl_dt -- the low-variance, frame-unambiguous
    signal, so we cache x before stepping.
    """

    # --- reward weights: your seam to TUNE (run 4 "biological-cost" set) --------
    contact_w = 0.75    # penalty per NON-paw geom on the floor. Was 1.5, but the
                        #   TB rebalance (DESIGN_NOTES #4) showed contact ~ -1.17/step
                        #   CO-DOMINATED the return (tied with smooth) while forward
                        #   earned ~0 -> "don't move" was optimal. Halved so attempting
                        #   locomotion isn't drowned. (The old "0.5 too soft -> migrated
                        #   the skid" was the pre-gate direct-action env; the stance gate
                        #   + planted-forelimb center now carry the anti-skid pressure.)
    hand_w = 0.0       # BONUS per front-paw (hand) geom on the floor -- a carrot
                        #   for the missing behavior (it never planted its hands)
    energy_w = 0.13     # penalty per unit mechanical power (|force*vel| summed) --
                        #   a cost-of-transport proxy. RAISED 5e-3 -> 0.13 for the
                        #   velocity-tracking reward: at 5e-3 the penalty was ~1% of the
                        #   reward (effectively OFF -> legs free to wheelspin). 0.13 is
                        #   power-probe calibrated (median ~6.9 W at the old gait) so
                        #   energy is ~30% of track_w -> it now actually selects
                        #   efficient, low-slip gaits (min mean power == min CoT at a
                        #   fixed commanded speed).
    smooth_w = 3e-3     # penalty per unit sum-squared action delta. Lowered
                        #   0.01 -> 3e-3 because it DIRECTLY antagonizes the new
                        #   rhythm/phase terms (smoothness punishes the very
                        #   oscillation rhythm rewards) -- at 0.01 it would just
                        #   suppress the experiment into a null result.

    # (Removed with the reward re-derivation: the capped-forward reward
    # [forward_w, forward_speed_cap] -> replaced by velocity tracking below; and the
    # CPG-inspired rhythm/phase shaping [phase_w, rhythm_w, rhythm_scale] -> redundant
    # with the CPG action space and the source of the max-frequency wheelspin.)

    # --- STANCE GATE (state-based curriculum) -----------------------------------
    # A smooth signal g in [0,1] = "how well is the rat in a supported stance?".
    # It multiplies the forward reward (you only get paid to move once you're up)
    # and, in the CPG env, the residual cost (aperiodic control is cheap while
    # getting up / stumbling, expensive once planted -> hand off to the CPG). All
    # OFF by default so other scripts/envs are untouched; a training script opts in.
    gate_forward = False   # if True: forward *= g
    gate_use_hands = False # if True: g also requires FRONT-paw contact (-> quadruped);
                           #   False keeps the biped valid (feet-only stance).
    gate_up_lo, gate_up_hi = 0.5, 0.9    # xmat[8] ramp: 0.5->0 (termination) .. 0.9->1
    gate_z_lo, gate_z_hi = 0.03, 0.05    # torso height ramp: fallen .. standing

    # --- gate FLOOR + soft hands (make the quadruped gate a ramp, not a wall) ----
    # The GATE_USE_HANDS=True run stalled with g~0 forever, because:
    #  (1) forward *= g annihilates the forward GRADIENT (not just the value) whenever
    #      g==0 -> the policy never learns moving is good. gate_floor fixes this by
    #      ATTENUATING forward (and the CPG residual cost) out of stance instead of
    #      zeroing it: forward *= gate_floor + (1-gate_floor)*g. Still a premium for
    #      being in stance, but a live gradient everywhere.
    #  (2) s_hands from a CONTACT COUNT is 0 until a paw touches -> no slope to climb.
    #      gate_hands_soft swaps it for a smooth "how close are the front paws to the
    #      floor" ramp, so lowering the front end is rewarded before contact and leads
    #      into the quadruped stance. Probe: front paws sit ~0.016-0.025 lifted (rat
    #      settles back on its haunches) and ~0.002-0.006 planted.
    # Both default OFF (gate_floor=0, hands_soft=False) -> unchanged for other runs.
    gate_floor = 0.0        # forward *= (gate_floor + (1-gate_floor)*g); 0 = hard gate
    gate_hands_soft = False # True: s_hands = paw-height ramp; False: contact count
    hand_z_lo, hand_z_hi = 0.004, 0.020  # front-paw z: planted (s=1) .. lifted (s=0)

    # --- Q1: gate the PERIODIC rewards by stance (complementary to the residual) --
    # rhythm/phase reward oscillating limbs REGARDLESS of stance, so the policy can
    # earn them by flailing while fallen (limb_speed>0 even lying down). Multiply
    # them by the SAME stance ramp used for forward/residual so periodic locomotion
    # only pays once planted -- the inverse of the residual, which is cheap OUT of
    # stance (res_cost *= g). One gate g wired to a reward (rhythm ~ g) vs a cost
    # (residual ~ 1-g) gives the "recover aperiodically, then stride" curriculum.
    # OFF by default (unchanged for non-CPG runs); make_cpg_env opts in.
    gate_rhythm = False     # if True: phase_reward, rhythm_reward *= stance ramp

    # --- velocity-command tracking (replaces the capped-forward reward) ----------
    # Each episode draws a slow sinusoidal target speed v*(t) sweeping [v_lo, v_hi]
    # with a random period in [cmd_T_lo, cmd_T_hi]; the reward is a Gaussian well of
    # width sigma(v*) peaking at track_w when vx == v*. Sweeping the target forces a
    # stable, efficient gait at EVERY speed -- gait transitions then EMERGE from the
    # energy term picking the cheapest gait per speed (rather than being hand-shaped).
    v_lo = 0.1          # m/s: slowest commanded speed (stays in locomotion, not standing)
    v_hi = 0.5          # m/s: fastest commanded speed
    cmd_T_lo = 20.0     # s: shortest sweep period (slow vs the ~2-4 s value horizon ->
    cmd_T_hi = 40.0     #    quasi-static, so the critic sees a ~constant target locally)
    track_w = 5.0       # peak tracking reward (matches old forward_w*cap = 5*0.6 scale)
    # SPEED-DEPENDENT tolerance sigma(v*), linear from track_sigma_lo at v_lo to
    # track_sigma_hi at v_hi. Rationale: the standstill (vx=0) barrier equals the full
    # command v*, so a FIXED narrow sigma leaves high commands stranded ~5 sigma out on
    # the dead Gaussian tail (no gradient -> the SAC_16 vx~0 stall). Widening sigma with
    # v* keeps the standstill error at ~1-1.7 sigma across the whole band (a live
    # guiding gradient everywhere) while preserving tight low-speed precision. This is
    # Weber-like (tolerance scales with target magnitude); anneal/tighten later for
    # precision once it locomotes. (See DESIGN_NOTES: SAC_16 tracking-barrier fix.)
    track_sigma_lo = 0.10   # m/s tolerance at v_lo (standstill barrier ~1.0 sigma -> steepest)
    track_sigma_hi = 0.30   # m/s tolerance at v_hi (standstill barrier ~1.7 sigma -> still live)

    def _stance_gate(self, n_hand, n_foot):
        """Smooth stance-quality g in [0,1] = product of soft indicators. Uses the
        contact counts already computed in _get_reward (no second contact pass)."""
        xmat8 = float(self.data.body("torso").xmat[8])
        z = float(self.data.body("torso").xpos[2])
        s_up = np.clip((xmat8 - self.gate_up_lo) / (self.gate_up_hi - self.gate_up_lo), 0.0, 1.0)
        s_h = np.clip((z - self.gate_z_lo) / (self.gate_z_hi - self.gate_z_lo), 0.0, 1.0)
        s_feet = np.clip(n_foot / 2.0, 0.0, 1.0)   # both hind paws down -> 1
        g = s_up * s_h * s_feet
        if self.gate_use_hands:
            # Soft ramp gives a gradient toward planting the hands; the binary count
            # is 0 until touch, so the optimizer has nothing to follow down.
            s_hands = (self._front_paw_proximity() if self.gate_hands_soft
                       else np.clip(n_hand / 2.0, 0.0, 1.0))
            g = g * s_hands
        return float(g)

    def _front_paw_proximity(self):
        """Smooth [0,1]: 1 when the lowest front paw is at/below hand_z_lo (planted),
        0 when lifted to hand_z_hi. Unlike the binary front-paw contact count, this
        gives the policy a GRADIENT to follow the front end down into stance."""
        if not hasattr(self, "_hand_geoms"):
            self._floor_contact_stats()  # populates the geom masks + _hand_geoms
        z_paw = min(float(self.data.geom_xpos[g][2]) for g in self._hand_geoms)
        return float(np.clip((self.hand_z_hi - z_paw)
                             / (self.hand_z_hi - self.hand_z_lo), 0.0, 1.0))

    def _floor_contact_stats(self):
        """One pass over ground contacts -> (n_bad, n_hand, n_foot).

        n_bad  = distinct NON-paw geoms touching the floor (the skid cheat:
                 forelimb bones, trunk, legs). Paws (toe/foot/hand/finger) are
                 legal support; the tail ("vertebra_*") is tolerated.
        n_hand = distinct FRONT-paw (hand/finger) geoms touching the floor -- the
                 behavior we want to REWARD, since it never plants its hands.
        n_foot = distinct HIND-paw (toe/foot) geoms touching the floor -- the
                 support the stance gate needs to tell "on its feet" from "down".

        Distinct geoms (not raw contact points) so a bone lying flat counts once,
        keeping the weights interpretable.
        """
        # Lazily precompute the floor id and per-geom masks.
        if not hasattr(self, "_floor_id"):
            self._floor_id = self.model.geom("floor").id
            paw = ("toe", "foot", "hand", "finger")
            tolerated = ("vertebra",)  # tail brush is natural, don't penalize it
            self._ok_contact = np.zeros(self.model.ngeom, dtype=bool)
            self._is_hand = np.zeros(self.model.ngeom, dtype=bool)
            self._is_foot = np.zeros(self.model.ngeom, dtype=bool)
            for gid in range(self.model.ngeom):
                name = self.model.geom(gid).name or ""
                self._ok_contact[gid] = any(tok in name for tok in paw + tolerated)
                self._is_hand[gid] = ("hand" in name) or ("finger" in name)
                self._is_foot[gid] = ("toe" in name) or ("foot" in name)
            # Front-paw geom ids, cached for the soft paw-height gate signal.
            self._hand_geoms = [gid for gid in range(self.model.ngeom)
                                if self._is_hand[gid]]

        bad, hands, feet = set(), set(), set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self._floor_id:
                other = c.geom2
            elif c.geom2 == self._floor_id:
                other = c.geom1
            else:
                continue  # a self-collision, not a ground contact
            if self._is_hand[other]:
                hands.add(other)
            elif self._is_foot[other]:
                feet.add(other)
            elif not self._ok_contact[other]:
                bad.add(other)
        return len(bad), len(hands), len(feet)

    def _paw_contact_forces(self):
        """Per-paw ground normal force [FL, FR, HL, HR], normalized by body weight.

        A real afferent signal (cutaneous / load mechanoreception) the rat was
        previously missing -- it literally couldn't feel its feet touch. Continuous
        and load-graded (not a boolean) so the policy gets *how hard* each paw is
        loaded. Normalizing by body weight (m*g) makes it ~[0,1] per paw and
        model-scale-free. Distinct from the stance gate g (a derived reward
        construct); this is raw sensing, so it belongs in the observation."""
        if not hasattr(self, "_paw_group"):
            self._floor_id_pf = self.model.geom("floor").id
            # group id: 0=front-L, 1=front-R, 2=hind-L, 3=hind-R, -1=not a paw
            self._paw_group = np.full(self.model.ngeom, -1, dtype=int)
            for gid in range(self.model.ngeom):
                nm = self.model.geom(gid).name or ""
                side = 0 if "_L" in nm else (1 if "_R" in nm else -1)
                if side < 0:
                    continue
                if ("hand" in nm) or ("finger" in nm):
                    self._paw_group[gid] = 0 + side
                elif ("toe" in nm) or ("foot" in nm):
                    self._paw_group[gid] = 2 + side
            gz = -float(self.model.opt.gravity[2])
            self._weight = float(np.sum(self.model.body_mass)
                                 * (gz if gz > 1e-6 else 9.81))
        forces = np.zeros(4, dtype=np.float64)
        if self.data.ncon:
            buf = np.zeros(6, dtype=np.float64)
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if c.geom1 == self._floor_id_pf:
                    other = c.geom2
                elif c.geom2 == self._floor_id_pf:
                    other = c.geom1
                else:
                    continue
                grp = self._paw_group[other]
                if grp >= 0:
                    mujoco.mj_contactForce(self.model, self.data, i, buf)
                    forces[grp] += abs(buf[0])   # buf[0] = normal-force magnitude
        return (forces / (self._weight + 1e-9)).astype(np.float32)

    def _reset_model(self):
        # Keep the base noise-injected reset, and clear the smoothness memory so
        # the first step of a new episode isn't charged a spurious action jump.
        super()._reset_model()
        self._prev_action = None
        # Per-episode velocity command: a slow sinusoid v*(t) the policy must TRACK.
        # Draw the sweep CENTRE in [v_lo, v_hi]; the amplitude is then the MAXIMAL
        # symmetric excursion that still fits the band -- the distance to the nearer
        # edge -- so v_mid +/- v_amp stays inside [v_lo, v_hi] BY CONSTRUCTION while
        # sweeping as wide as that centre allows (central v_mid -> full-band sweep;
        # near-edge -> narrower, hugging that edge). Only the centre is random; wide
        # per-episode sweeps drive walk<->trot transitions within an episode.
        # Leading-underscore names mark these as per-episode STATE (like _phase),
        # distinct from the fixed v_lo/v_hi/cmd_T_* knobs.
        self._cmd_v_mid = self.np_random.uniform(self.v_lo, self.v_hi)
        self._cmd_v_amp = min(self._cmd_v_mid - self.v_lo, self.v_hi - self._cmd_v_mid)
        self._cmd_T = self.np_random.uniform(self.cmd_T_lo, self.cmd_T_hi)
        self._cmd_phi = self.np_random.uniform(0.0, 2.0 * np.pi)

    def _target_speed(self):
        """Commanded forward speed v*(t) at the CURRENT sim time (the slow sinusoid whose
        parameters were drawn this episode in _reset_model). The getattr(..., default)
        guards the ONE construction-time call: the base __init__ builds observation_space
        via _get_obs -> _target_speed BEFORE any reset() has drawn the command, so _cmd_*
        don't exist yet. Defaults (amp 0 -> flat, T = cmd_T_lo -> no divide-by-zero) make
        that early call return the mid-band speed instead of raising AttributeError."""
        v_mid = getattr(self, "_cmd_v_mid", 0.5 * (self.v_lo + self.v_hi))
        v_amp = getattr(self, "_cmd_v_amp", 0.0)
        T = getattr(self, "_cmd_T", self.cmd_T_lo)
        phi = getattr(self, "_cmd_phi", 0.0)
        return v_mid + v_amp * np.sin(2.0 * np.pi * self.data.time / T + phi)

    def _track_sigma(self, v_star):
        """Speed-dependent Gaussian-well width sigma(v*): linear from track_sigma_lo at
        v_lo to track_sigma_hi at v_hi (v* clipped to the band). Wider at high speed so
        the standstill barrier stays ~1-1.7 sigma (a live gradient); tight at low speed
        for precision. See the track_sigma_lo/hi class-attr note."""
        t = (float(v_star) - self.v_lo) / (self.v_hi - self.v_lo)
        t = min(1.0, max(0.0, t))
        return self.track_sigma_lo + t * (self.track_sigma_hi - self.track_sigma_lo)

    def step(self, action):
        # Record the torso's world-x BEFORE the physics substeps run, so
        # _get_reward (called at the end of super().step) can form a displacement.
        self._x_before = float(self.data.body("torso").xpos[0])
        obs, reward, terminated, truncated, info = super().step(action)
        # Surface the individual reward terms (stashed in _get_reward) so a
        # callback can log each one to TensorBoard separately -- turning "is it
        # actually moving?" from arithmetic into a glance.
        info.update(self._reward_terms)
        return obs, reward, terminated, truncated, info

    def _smooth_weights(self):
        """Per-action-dimension weight for the smoothness penalty. Default: a flat
        smooth_w on every dim. Subclasses with a HETEROGENEOUS action space (e.g.
        the CPG env's [freq, amp, phase, residual] groups) override this so one
        group -- like the single frequency knob, otherwise 1 term drowned by ~114
        -- can be regularized on its own scale."""
        return np.full(self.action_space.shape[0], self.smooth_w, dtype=np.float64)

    def _smoothness_cost(self, action):
        """Sum of per-dim WEIGHTED, NORMALIZED squared action deltas.

        Each delta is divided by that dimension's action-space half-range before
        squaring, so the term is dimensionless and comparable across actuators no
        matter how a model defines its ctrlrange (a no-op here -- the rat's ranges
        are all [-1,1] -- but correct for car/walker/etc., and it future-proofs the
        term). Also owns the _prev_action bookkeeping (first step of an episode
        costs 0)."""
        action = np.asarray(action, dtype=np.float64)
        if getattr(self, "_action_half", None) is None:
            low = np.asarray(self.action_space.low, dtype=np.float64)
            high = np.asarray(self.action_space.high, dtype=np.float64)
            half = 0.5 * (high - low)
            half[~np.isfinite(half) | (half <= 1e-9)] = 1.0
            self._action_half = half
        w = self._smooth_weights()
        if self._prev_action is None:
            # Stash the per-dim contributions (zeros on the first step) so a
            # subclass with grouped actions can log the smooth penalty per group
            # -- the aggregate hides whether e.g. the freq knob or the per-joint
            # terms dominate, which is exactly what you need to tune it.
            self._smooth_per_dim = np.zeros_like(w)
            cost = 0.0
        else:
            d = (action - self._prev_action) / self._action_half
            self._smooth_per_dim = w * np.square(d)
            cost = float(np.sum(self._smooth_per_dim))
        self._prev_action = action.copy()
        return cost

    def _get_reward(self, action):
        x_after = float(self.data.body("torso").xpos[0])
        vx = (x_after - self._x_before) / self.ctrl_dt   # m/s along +x

        # --- reward terms: your seam to TUNE ---------------------------------
        # Diagnosis from run 1: alive(1) + upright(~1) paid ~2/step just for
        # standing, while forward added ~0.4 -> the policy banked survival, not
        # locomotion. Rebalanced so forward out-pays standing, and shrinking the
        # posture terms because TERMINATION already forces upright (xmat[8]>0.5)
        # and height>0.03 -- so we don't need to also *pay* for posture.
        alive = 0.0                                          # was 1.0 (a floor, not the payout)
        control_cost = 0.000 * np.sum(np.square(action))
        upright_bonus = 0.0 * self.data.body("torso").xmat[8]  # was 1.0 (mostly rewarded posing)

        # Contact shaping. n_bad = non-paw geoms on the floor (the skid cheat);
        # n_hand = front paws on the floor (the missing behavior). Run 3 showed a
        # soft contact_w=0.5 only MIGRATED the skid (forearm -> upper arm) and the
        # hands still never touched. Run 4: push harder (contact_w=1.5) AND add a
        # carrot for actually planting the front paws. n_foot feeds the stance gate.
        n_bad, n_hand, n_foot = self._floor_contact_stats()
        contact_cost = self.contact_w * n_bad
        hand_bonus = self.hand_w * n_hand

        # Stance gate: you only get PAID to move forward once you're up on support.
        # Gating (not deleting) forward removes the early "topple forward for cheap
        # vx" exploit and stops taxing the stand-up transient. g is also reused by
        # the CPG env to gate the residual cost.
        g = self._stance_gate(n_hand, n_foot)
        # Velocity-TRACKING reward (replaces the old capped-forward term): a Gaussian
        # "well" peaking when vx matches the commanded speed v* (width track_sigma),
        # penalizing OVERSHOOT the same as undershoot -> no incentive to spin the legs
        # faster than commanded (the old wheelspin), no speed cap needed.
        #
        # ASYMMETRIC / zero-at-standstill: a plain Gaussian centred at v* still pays
        # ~0.75-1.8 (of track_w=3) at vx=0, because vx=0 sits only ~1-1.6 sigma from v*.
        # That freebie made STANDING a local optimum -- the SAC_16/18 runs converged to a
        # stable standstill (ep_len 2000, dfreq pinned min, vx~0) and collected it.
        # Fix: subtract the well's value AT vx=0, a0 = exp(-v*^2 / 2 sigma^2). Now
        # standing pays exactly 0, vx=v* pays the peak track_w*(1-a0), and moving
        # BACKWARD (vx<0) or OVERSHOOTING past 2v* goes NEGATIVE -- both penalized, and
        # there is a live uphill gradient out of standstill toward v*. NOT renormalized
        # by 1/(1-a0): that restores a fixed peak but blows up as v*->0 (a0->1); we take
        # the sagging low-v* peak instead to avoid that amplification.
        v_star = self._target_speed()
        sigma = self._track_sigma(v_star)   # speed-dependent tolerance (wide at high v*)
        a0 = float(np.exp(-np.square(v_star) / (2.0 * np.square(sigma))))
        gauss = float(np.exp(-np.square(vx - v_star) / (2.0 * np.square(sigma)))) + float(np.exp(-np.square(vx - v_star) / (2.0 * np.square(sigma / 4.0))))
        r_track = self.track_w * (gauss - a0)
        if self.gate_forward:
            # FLOOR the gate: attenuate tracking out of stance, never annihilate it,
            # so a locomotion gradient always exists (gate_floor=0 -> old hard gate).
            ramp = self.gate_floor + (1.0 - self.gate_floor) * g
            r_track *= ramp

        # Cost of transport (a "biological cost" the rigid servo body doesn't pay
        # for free): mechanical power = sum |actuator_force * actuator_velocity|.
        # A persistent skid does a lot of friction work, so this taxes it
        # independently of which body part is touching.
        power = float(np.sum(np.abs(self.data.actuator_force * self.data.actuator_velocity)))
        energy_cost = self.energy_w * power

        # Action smoothness: penalize how much the command jumped since last step.
        # Real muscles low-pass their activation; this discourages the rigid,
        # high-frequency wiggle and nudges toward rhythmic, continuous motion.
        # Delegated to _smoothness_cost so the penalty is (a) normalized per action
        # dim (dimensionless -> uniform across a model's actuators) and (b) per-group
        # weightable by subclasses (the CPG env boosts the frequency knob).
        smooth_cost = self._smoothness_cost(action)

        # (Removed: the old CPG-inspired rhythm + phase shaping. `rhythm` rewarded raw
        # limb SPEED, which directly drove the max-frequency wheelspin; `phase` is now
        # guaranteed structurally by the CPG action space. Both were redundant/harmful
        # once the reward was re-derived to {track, energy, stability}.)

        # Stash raw per-step components; step() copies these into `info`.
        self._reward_terms = {
            "track": float(r_track),
            "v_target": float(v_star),
            "alive": float(alive),
            "control_cost": float(-control_cost),
            "upright": float(upright_bonus),
            "contact": float(-contact_cost),
            "hand": float(hand_bonus),
            "energy": float(-energy_cost),
            "smooth": float(-smooth_cost),
            "bad_contacts": float(n_bad),
            "hand_contacts": float(n_hand),
            "foot_contacts": float(n_foot),
            "gate": float(g),
            "hand_prox": float(self._front_paw_proximity()),  # front-end-down signal
            "vx": float(vx),
        }
        return float(r_track + alive - control_cost + upright_bonus
                     - contact_cost + hand_bonus - energy_cost - smooth_cost)


class RodentCPGEnv(RodentRunEnv):
    """Rat with a CPG-structured action space (structure the action space instead
    of shaping the reward).

    Instead of commanding 38 joint targets directly, the policy commands a Central
    Pattern Generator plus a feedback residual. An internal phase phi is INTEGRATED
    across control steps (this is what makes the rhythm persist and stay smooth
    despite action noise), and each actuator's command is

        ctrl_j = center_j + amp_gain*half_j * a_j * sin(phi + psi_j)   (CPG term)
                          + res_gain*half_j * r_j                       (feedback)

    So the ACTION is [ dfreq(1), a_j(nu), psi_j(nu), r_j(nu) ]  (len 1 + 3*nu):
      - dfreq  : modulates the oscillator frequency (adjustable clock)
      - a_j    : per-joint amplitude / "how much the CPG drives this joint" (0..1)
      - psi_j  : per-joint phase offset in [-pi, pi]
      - r_j    : per-joint aperiodic residual for balance/feedback

    Why this attacks the problem at the right level: the periodic part of every
    joint is FORCED to be a shared-clock sinusoid, so the aperiodic degenerate
    solutions (belly wiggle, frantic scramble) are largely removed from the search
    space by construction. Rhythm is a structural guarantee, not a reward we beg
    for. The clock (sin phi, cos phi) is appended to the observation so the policy
    can modulate in-phase. The existing `smooth` penalty now acts on the CPG
    PARAMETERS -> it rewards a *stable* gait rather than fighting oscillation.
    """

    # --- smoothness weights PER ACTION GROUP (see _smooth_weights) --------------
    # The base env's flat smooth_w puts the 1 frequency term among 114 per-joint
    # terms, so the clock was effectively unregularized -> it thrashed (probe:
    # mean|df| ~0.66 Hz/step, f swinging the full 1-5 Hz range). Give the frequency
    # its own, much larger weight so a STEADY tempo is rewarded, while keeping
    # amp/phase/residual at the base smooth_w. (Phase can be boosted too if the
    # per-limb coordination still jitters -- left at base for now.)
    # Smooth weights CUT in the DESIGN_NOTES #4 rebalance: `smooth` was co-dominant
    # (~ -1.11/step) with contact, so escaping the "don't move" basin needs it down.
    # The new per-group split (smooth_freq/amp/phase/res/coff in info) revealed the
    # penalty is almost ENTIRELY the freq knob (~30x the whole per-joint sum), so
    # freq is where the cut has to land -- halving only the per-joint terms would
    # barely dent the aggregate. smooth_w_freq 2.0 -> 1.0 (still discourages wild
    # tempo swings, but no longer dominates the return); per-joint 3e-3 -> 1.5e-3
    # (they directly antagonize the rhythm/phase reward). Watch smooth_freq + tempo
    # next run for jitter now that the fixed-alpha fix should calm the action noise.
    smooth_w_freq = 0.5     # weight on the (normalized) frequency-knob delta (was 2.0)
    smooth_w_duty = 0.5     # weight on the (normalized) duty/warp-knob delta (slow, like freq)
    smooth_w_amp = 0.01   # per-joint amplitude delta (was 3e-3)
    smooth_w_phase = 0.0 # per-joint phase-offset delta (was 3e-3)
    smooth_w_res = 0.01   # per-joint residual delta (was 3e-3)

    # --- SYMMETRY-REDUCED ACTION SPACE (bilateral change of basis) --------------
    # The 153-dim action (1 + 4*38) drove the SAC critic to over-estimate and
    # DIVERGE (max-bias grows with action dim). We shrink it by exploiting the
    # rat's sagittal MIRROR symmetry and by tying left/right phase, WITHOUT losing
    # the residual's full authority (feedback must break symmetry to catch a
    # stumble). New layout (see _build_symmetry_maps / _apply_action):
    #   [ dfreq(1), duty(1), amp_sym(19), phase_tied(19), res(38), coff_sym(19) ]
    #   = 97 dims (down from 153, ~1.6x).
    # Principle: impose symmetry on the REFERENCE (amp/phase/coff -> straight,
    # mirror-symmetric intended gait) but NOT on the CORRECTIONS (res stays full
    # 38 so it can generate roll/lateral/one-sided catches and stand-ups).
    #   - amp_sym / coff_sym: 14 L/R pair commons + 5 sagittal-midline
    #     (extend/atlas/mandible); the 5 antisym-midline (bend/twist) are pinned to
    #     0 (they oscillate/offset only to TURN, a Phase-2 DOF).
    #   - phase_tied: one phase per pair with the partner locked a HALF-CYCLE away
    #     (contralateral antiphase, the walk/trot relation) + 5 midline phases.
    # The map is a fixed linear expansion built once at construction, so it stays a
    # clean change of basis (synergy factoring can slot in later by shrinking it).
    use_symmetry = True      # False -> fall back to the legacy 153-dim per-joint action
    # Duty-factor phase WARP (temporal symmetry break). One global knob beta in
    # [duty_lo, duty_hi] redistributes time within each cycle: the first half-wave
    # occupies fraction beta of real time, the second (1-beta). beta=0.5 == uniform
    # (the neutral at duty action 0), beta>0.5 stretches stance. Monotonic
    # reparameterization of the shared clock -> period is preserved.
    use_duty_warp = True
    duty_lo, duty_hi = 0.4, 0.6   # beta range; action in [-1,1] maps linearly onto this

    # --- residual USAGE cost: a SOFT constraint on the aperiodic escape hatch ----
    # The probe showed the policy defeats the CPG by maxing the residual (res_mean
    # ~0.8) and using it as a parallel controller. Rather than HARD-capping the
    # ceiling (res_gain -> tiny), keep headroom (res_gain 0.2) and TAX residual USE
    # so it stays a quiet balance-trim spent only on real corrections. Two shapes:
    #   "l1"  : cost = w * sum|r|          -> sparsity (mostly-off, occasional big),
    #                                         but taxes a big EMERGENCY spike too.
    #   "avg" : cost = w * sum(EMA|r|)     -> taxes SUSTAINED use, not brief spikes.
    # The emergency-vs-cheat distinction is DURATION, not magnitude, so "avg" is the
    # principled default: a one-off stumble catch barely moves the EMA, while
    # residual-as-locomotion pays every step. (Downside: the cost lags the action,
    # muddying per-step credit -- the value function has to bridge it.)
    res_cost_w = 0.02        # overall weight on residual usage (seam to TUNE)
    res_cost_mode = "avg"    # "avg" (recommended) or "l1"
    res_ema_beta = 0.02      # EMA rate for "avg"; tau ~ 1/beta = 50 steps = 1 s
    gate_res = False         # if True: residual cost *= stance gate g (cheap while
                             #   getting up / stumbling, taxed once planted)

    # --- Q3: slow, policy-controlled CENTER OFFSET (a movable postural setpoint) --
    # `center` is otherwise FIXED, so the only way to shift a joint's setpoint is the
    # fast, taxed residual -- which is why the forelimbs looked stiff: they always
    # servo back to the same frozen pose, wiggling around it rather than repositioning
    # like a real rat. Give the policy a 4th action group `coff` that a slow EMA turns
    # into a postural setpoint separate from the fast residual:
    #     ctrl = center + coff_ema + osc + residual
    # Guardrails so it stays POSTURAL (not a slow parallel direct-drive controller
    # that could resurrect the belly-crawl): modest authority (coff_gain*half) and a
    # slow EMA (too sluggish to do stepping).
    #
    # COST SEMANTICS (fixed): the offset is a movable OSCILLATION CENTER, so HOLDING a
    # non-zero setpoint must be FREE -- within a stable gait the center legitimately
    # sits at whatever angle is that joint's oscillation midpoint. We therefore charge
    # the RATE of change of the setpoint (cost to *reposition* the center), not its
    # magnitude. Integrated over a move, sum|d coff_ema| = |final offset|, so this is
    # effectively a ONE-TIME charge to shift the center, then zero cost to ride it --
    # "pay to change gait posture, free to hold it." (The earlier magnitude penalty was
    # a spring back to the midpoint, which fought the whole point of a movable center.)
    coff_gain = 0.3          # setpoint authority as a fraction of each joint half-range
    coff_beta = 0.01         # EMA rate; tau ~ 1/beta = 100 steps = 2 s (slow = postural)
    coff_move_w = 0.05       # tax on the SETPOINT's per-step change (repositioning cost)
    smooth_w_coff = 0.0      # coff move-cost owns "cost to reposition"; don't double-tax
                             #   the raw command via smoothness (the EMA rate-limits it)

    # --- planted forelimb nominal stance (CPG center offset) --------------------
    # The CPG oscillates around _center. At the ctrlrange MIDPOINT the forelimbs
    # hang BENT ~2 cm off the floor at the natural ride height (probe), so planting
    # the paws is a costly deviation at the edge of the action envelope -> the policy
    # leaves the arms parked bent (the "front paws never touch" failure). Shift the
    # forelimb center toward elbow EXTENSION so the neutral pose already reaches the
    # floor: planting becomes the default, lifting the deviation. Only the elbow
    # matters (extends the forearm to drop the paw); +0.6 of [-1,1] still leaves
    # room to oscillate for stepping. Calibrated offline by scripts/_calib_fore_center.py
    # (pin trunk at settled height, lower paws to the floor). {} => old midpoint center.
    cpg_center_offsets = {"elbow_L": 0.6, "elbow_R": 0.6}

    # --- observation normalization: a fixed afferent SENSOR GAIN (DESIGN_NOTES #1) --
    # The critic diverged (Q -> 1e13) in every run. Root cause (measured via an obs
    # probe under the SAC_8 policy): the observation is raw qpos/qvel, and the VELOCITY
    # channels are 20-40x every other input (torso angular-vel p99 ~12, joint-vel p99
    # ~12, max ~38; positions/quat/clock/paws are all <=~2). Feeding that badly-scaled
    # vector to an unbounded Q-head lets values blow up. We rescale the velocity dims to
    # ~unit HERE, in the afferent seam, rather than with an external VecNormalize wrapper
    # -- off-policy SAC wants a FIXED transform (the replay buffer stays consistent) and
    # a self-contained env (watch/probe/resume/montage all get normalized obs for free).
    # Only angvel + joint-vels are divided (linvel ~0.5 and the rest are already ~unit).
    obs_vel_scale = 10.0     # divisor for velocity obs channels (p99 ~12 -> ~1.2)

    def _obs_scale_vec(self):
        """Fixed per-dim divisor to bring the observation to ~unit scale, built once
        from the model's nq/nv split (no magic indices). Obs layout (see _get_obs):
          [ qpos[2:] (z, quat, joint angles) | qvel (linvel, angvel, joint vels) |
            clock(2) | paws(4) ].
        Only the angular-vel and joint-vel blocks are scaled; everything else is 1."""
        if getattr(self, "_obs_scale", None) is None:
            nq, nv = self.model.nq, self.model.nv
            n = (nq - 2) + nv + 6            # base (qpos[2:]+qvel) + clock(2) + paws(4)
            scale = np.ones(n, dtype=np.float32)
            vbase = nq - 2                    # index where the qvel block starts
            scale[vbase + 3:vbase + 6] = self.obs_vel_scale   # torso angular velocity
            scale[vbase + 6:vbase + nv] = self.obs_vel_scale  # joint velocities
            self._obs_scale = scale
        return self._obs_scale

    def __init__(self, xml_path, cpg_f_base=1.5, cpg_f_range=1.5,
                 cpg_amp_gain=0.6, cpg_res_gain=0.2, **kwargs):
        self._phase = 0.0  # exists before super().__init__ calls _get_obs()
        super().__init__(xml_path, **kwargs)

        nu = self.model.nu
        low = self.model.actuator_ctrlrange[:, 0].astype(np.float64).copy()
        high = self.model.actuator_ctrlrange[:, 1].astype(np.float64).copy()
        half = 0.5 * (high - low)
        center = 0.5 * (high + low)
        # Actuators with no ctrlrange set (low==high, i.e. "unlimited") would give
        # half=0 and a clip range of [0,0]; treat those as ~[-1,1] and unclipped.
        degenerate = half <= 1e-9
        half[degenerate] = 1.0
        center[degenerate] = 0.0
        low[degenerate] = -np.inf
        high[degenerate] = np.inf
        # Shift the neutral pose toward the calibrated planted forelimb stance, then
        # keep it inside the ctrlrange so the oscillation still swings around a legal
        # target (see cpg_center_offsets above).
        for name, val in self.cpg_center_offsets.items():
            center[self.model.actuator(name).id] += val
        center = np.clip(center, low, high)
        self._ctrl_low, self._ctrl_high = low, high
        self._half, self._center = half, center

        self.cpg_f_base = cpg_f_base
        self.cpg_f_range = cpg_f_range
        self.cpg_amp_gain = cpg_amp_gain
        self.cpg_res_gain = cpg_res_gain
        self._res_ema = None   # per-joint EMA of |residual|, (re)set on reset
        self._coff_ema = None  # per-joint slow center OFFSET (Q3), (re)set on reset

        # Redefine the action space. Either the symmetry-reduced layout
        # [dfreq, duty, amp_sym(19), phase_tied(19), res(38), coff_sym(19)] = 97,
        # or the legacy per-joint layout [dfreq, amp(nu), phase(nu), res(nu),
        # coff(nu)] = 1 + 4*nu. Both paths set the same _sl_* group slices so the
        # smoothness/reward code is layout-agnostic.
        if self.use_symmetry:
            self._build_symmetry_maps()
        else:
            self._build_legacy_maps()

    def _build_legacy_maps(self):
        """Legacy 153-dim per-joint action space (no symmetry, no duty warp)."""
        nu = self.model.nu
        self._S = None
        self._psi_gather = None
        self._sl_dfreq = slice(0, 1)
        self._sl_duty = None
        self._sl_amp = slice(1, 1 + nu)
        self._sl_phase = slice(1 + nu, 1 + 2 * nu)
        self._sl_res = slice(1 + 2 * nu, 1 + 3 * nu)
        self._sl_coff = slice(1 + 3 * nu, 1 + 4 * nu)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(1 + 4 * nu,), dtype=np.float32)

    def _build_symmetry_maps(self):
        """Build the fixed bilateral change-of-basis (see use_symmetry docstring) and
        set the reduced action space + per-group slices.

        Populates a fixed linear expansion so _apply_action stays a clean change of
        basis:
          self._S          (nu, n_sym) symmetric expansion: reduced params -> 38 joints
          self._psi_gather (nu,) int   which reduced-phase feeds each joint (-1 = none)
          self._psi_offset (nu,) float additive psi offset (+1.0 == half cycle, for R)
          self._sl_*       action-vector slices for each group; self.action_space
        """
        model = self.model
        nu = model.nu
        names = [model.actuator(i).name for i in range(nu)]

        def side(nm):
            if nm.endswith("_L") or "_L_" in nm:
                return "L"
            if nm.endswith("_R") or "_R_" in nm:
                return "R"
            return None

        def canon(nm):  # side-agnostic key so L/R partners collide
            return (nm.replace("_L_", "_@_").replace("_R_", "_@_")
                      .replace("_L", "_@").replace("_R", "_@"))

        pairs = {}                 # canon -> {"L": idx, "R": idx}
        sym_mid, anti_mid = [], []
        for i, nm in enumerate(names):
            s = side(nm)
            if s is not None:
                pairs.setdefault(canon(nm), {})[s] = i
            elif nm.endswith("_bend") or nm.endswith("_twist"):
                anti_mid.append(i)     # lateral bend / axial twist -> turning (antisym)
            else:
                sym_mid.append(i)      # extend / atlas / mandible -> sagittal (symmetric)

        pair_list = sorted((d["L"], d["R"]) for d in pairs.values())  # stable by L index
        n_pairs, n_symmid = len(pair_list), len(sym_mid)
        n_sym = n_pairs + n_symmid          # amp/coff symmetric-subspace dim
        n_phase = n_pairs + n_symmid        # one phase per pair + per sym-midline
        self._sym_dims = (n_pairs, n_symmid, len(anti_mid))

        # Symmetric expansion: pair-common -> BOTH sides equally; sym-midline
        # passthrough; antisym-midline rows stay 0 (pinned off -> no oscillation/offset).
        S = np.zeros((nu, n_sym), dtype=np.float64)
        for p, (li, ri) in enumerate(pair_list):
            S[li, p] = S[ri, p] = 1.0
        for k, mi in enumerate(sym_mid):
            S[mi, n_pairs + k] = 1.0
        self._S = S

        # Phase gather: L takes phase p @ offset 0, R takes phase p @ +1.0 (half cycle
        # == contralateral antiphase); sym-midline take their own phase; antisym-midline
        # are inert (amp pinned 0) -> gather -1, psi 0.
        gather = np.full(nu, -1, dtype=np.int64)
        offset = np.zeros(nu, dtype=np.float64)
        for p, (li, ri) in enumerate(pair_list):
            gather[li] = gather[ri] = p
            offset[ri] = 1.0
        for k, mi in enumerate(sym_mid):
            gather[mi] = n_pairs + k
        self._psi_gather = gather
        self._psi_offset = offset

        i = 0
        self._sl_dfreq = slice(i, i + 1); i += 1
        self._sl_duty = slice(i, i + 1); i += 1
        self._sl_amp = slice(i, i + n_sym); i += n_sym
        self._sl_phase = slice(i, i + n_phase); i += n_phase
        self._sl_res = slice(i, i + nu); i += nu
        self._sl_coff = slice(i, i + n_sym); i += n_sym
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(i,), dtype=np.float32)

    def _duty_beta(self, duty):
        """Map a duty action in [-1,1] to the stance fraction beta in [duty_lo,duty_hi].
        duty=0 -> midpoint (0.5 for the default range == uniform, no warp)."""
        t = 0.5 * (float(duty) + 1.0)
        return self.duty_lo + t * (self.duty_hi - self.duty_lo)

    def _warp_phase(self, theta, beta):
        """Duty-factor reparameterization of phase (temporal symmetry break). Maps the
        uniform angle theta so the first half-wave [0,pi) occupies fraction beta of the
        cycle and the second [pi,2pi) the rest. Monotonic & 2pi-periodic -> preserves
        the period, only redistributes time within it."""
        two_pi = 2.0 * np.pi
        u = np.mod(theta, two_pi) / two_pi
        v = np.where(u < beta,
                     0.5 * u / beta,
                     0.5 + 0.5 * (u - beta) / (1.0 - beta))
        return two_pi * v

    def _decode_action(self, action):
        """Split the (reduced or legacy) action into physical per-joint quantities:
        (dfreq, beta|None, amp_pos[nu], psi[nu], coff[nu], res[nu]). This is the ONLY
        place the two action layouts differ; the physics in _apply_action is shared."""
        if self.use_symmetry:
            dfreq = float(action[self._sl_dfreq][0])
            beta = (self._duty_beta(action[self._sl_duty][0])
                    if self.use_duty_warp else None)
            amp_red = np.asarray(action[self._sl_amp], dtype=np.float64)
            amp_pos = self._S @ (0.5 * (amp_red + 1.0))       # 19->38; antisym-mid -> 0
            phase_red = np.asarray(action[self._sl_phase], dtype=np.float64)
            g = self._psi_gather
            psi = np.where(g >= 0, phase_red[np.clip(g, 0, None)] + self._psi_offset, 0.0)
            coff = self._S @ np.asarray(action[self._sl_coff], dtype=np.float64)  # 19->38
            res = np.asarray(action[self._sl_res], dtype=np.float64)
        else:
            dfreq = float(action[0])
            beta = None
            amp_pos = 0.5 * (np.asarray(action[self._sl_amp], dtype=np.float64) + 1.0)
            psi = np.asarray(action[self._sl_phase], dtype=np.float64)
            res = np.asarray(action[self._sl_res], dtype=np.float64)
            coff = np.asarray(action[self._sl_coff], dtype=np.float64)
        return dfreq, beta, amp_pos, psi, coff, res

    def _get_obs(self):
        base = super()._get_obs()
        ph = getattr(self, "_phase", 0.0)
        # Append the CPG clock (sin/cos phi) + per-paw ground normal force (a real
        # afferent signal, not the derived gate g). Order: [base, clock(2), paws(4)].
        paws = self._paw_contact_forces()
        obs = np.concatenate([base, [np.sin(ph), np.cos(ph)], paws]).astype(np.float32)
        # Sensor-gain normalization: divide the velocity channels to ~unit scale so the
        # critic's inputs are well-conditioned (see obs_vel_scale / _obs_scale_vec).
        scaled = obs / self._obs_scale_vec()
        # Expose the (unit-normalized) commanded speed so the policy can TRACK it -- a
        # target it can't see would make the reward non-stationary. Append AFTER the
        # scale-divide (cmd is already ~unit) so _obs_scale_vec's length stays keyed to
        # base+clock+paws; appending BEFORE would be a length mismatch.
        cmd_norm = (self._target_speed() - self.v_lo) / (self.v_hi - self.v_lo)
        return np.concatenate([scaled, [cmd_norm]]).astype(np.float32)

    def _reset_model(self):
        super()._reset_model()
        # Randomize the STARTING phase of the CPG clock each episode. Otherwise every
        # episode begins the fragile stand-up transient at the exact same point in the
        # gait cycle (phase 0), so the policy only ever learns to launch from one phase
        # -> narrow startup basin (the ~40% instant-topples). Randomizing forces it to
        # be able to ENTER the gait at any phase, widening that basin. Buffer-safe: this
        # only changes the reset distribution, not the reward or the action->ctrl map,
        # so a seamless resume can keep using the existing replay buffer.
        self._phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        self._res_ema = None    # forget last episode's residual history
        self._coff_ema = None   # forget last episode's slow center offset
        self._coff_delta = None  # no repositioning charged on the first step

    def _apply_action(self, action):
        nu = self.model.nu
        dfreq, beta, amp_pos, psi, coff, res = self._decode_action(action)

        # Advance the integrated phase at the (policy-modulated) frequency.
        f = self.cpg_f_base + self.cpg_f_range * dfreq
        self._phase = float((self._phase + 2.0 * np.pi * f * self.ctrl_dt)
                            % (2.0 * np.pi))

        # Q3: low-pass the requested center offset into a SLOW postural setpoint, so
        # the policy can hold a shifted pose without the fast residual fighting it.
        # Stash the per-step CHANGE of the setpoint so _get_reward can tax repositioning
        # (not the held level) -- see coff_move_w.
        target_off = self.coff_gain * self._half * coff
        if self._coff_ema is None:
            self._coff_ema = np.zeros(nu, dtype=np.float64)
        new_ema = ((1.0 - self.coff_beta) * self._coff_ema
                   + self.coff_beta * target_off)
        self._coff_delta = new_ema - self._coff_ema
        self._coff_ema = new_ema

        # Per-joint oscillator angle (psi in [-1,1] -> pi*psi rad), optionally
        # duty-warped (temporal symmetry break). amp_pos is already the nonneg [0,1]
        # amplitude (the symmetric expansion pins antisym-midline joints to 0).
        theta = self._phase + np.pi * psi
        if beta is not None:
            theta = self._warp_phase(theta, beta)
        osc = self.cpg_amp_gain * self._half * amp_pos * np.sin(theta)
        residual = self.cpg_res_gain * self._half * res
        ctrl = self._center + self._coff_ema + osc + residual
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)

    def _smooth_weights(self):
        # Heterogeneous action -> weight each GROUP separately (via the _sl_* slices,
        # so this works for both the reduced and legacy layouts): the single freq
        # (and duty) knob otherwise gets drowned by the per-joint terms, leaving the
        # clock effectively unregularized.
        w = np.empty(self.action_space.shape[0], dtype=np.float64)
        w[self._sl_dfreq] = self.smooth_w_freq
        if self._sl_duty is not None:
            w[self._sl_duty] = self.smooth_w_duty
        w[self._sl_amp] = self.smooth_w_amp
        w[self._sl_phase] = self.smooth_w_phase
        w[self._sl_res] = self.smooth_w_res
        w[self._sl_coff] = self.smooth_w_coff
        return w

    def _get_reward(self, action):
        # Inherit all of RodentRunEnv's terms (it also fills self._reward_terms),
        # then subtract the residual-USAGE tax that's unique to this action space.
        base = super()._get_reward(action)

        nu = self.model.nu
        res = np.abs(np.asarray(action[self._sl_res], dtype=np.float64))
        if self.res_cost_mode == "avg":
            # Low-pass |residual| per joint: a brief spike barely moves the EMA, but
            # sustained "residual as locomotion" saturates it and pays every step.
            if self._res_ema is None:
                self._res_ema = np.zeros(nu, dtype=np.float64)
            self._res_ema = ((1.0 - self.res_ema_beta) * self._res_ema
                             + self.res_ema_beta * res)
            usage = float(np.sum(self._res_ema))
        else:  # "l1": instantaneous magnitude -> sparsity, but taxes spikes too
            usage = float(np.sum(res))
        res_cost = self.res_cost_w * usage
        if self.gate_res:
            # g was stashed by super()._get_reward. Use the SAME floor as forward so
            # the residual is cheap while getting up / stumbling but never entirely
            # untaxed (gate_floor=0 -> old behavior: fully free out of stance).
            g = self._reward_terms.get("gate", 1.0)
            res_cost = res_cost * (self.gate_floor + (1.0 - self.gate_floor) * g)

        # Q3: tax the RATE OF CHANGE of the center setpoint, not its level -- holding a
        # non-zero oscillation center is FREE (that's the point of a movable center);
        # only *repositioning* it costs. Integrated over a move this equals the offset
        # magnitude, i.e. a one-time "cost to shift posture," then free to ride.
        move = getattr(self, "_coff_delta", None)
        if move is not None:
            center_move = float(np.sum(np.abs(move) / self._half))
        else:
            center_move = 0.0
        center_cost = self.coff_move_w * center_move

        # Break the aggregate `smooth` penalty into its action GROUPS so TB shows
        # whether the freq knob (smooth_w_freq=2.0, 1 term) or the per-joint terms
        # (114 terms) drive it -- the decomposition needed to tune smooth honestly.
        # _smooth_per_dim was stashed by _smoothness_cost during super()._get_reward.
        spd = getattr(self, "_smooth_per_dim", None)
        if spd is not None and len(spd) == self.action_space.shape[0]:
            self._reward_terms["smooth_freq"] = float(-np.sum(spd[self._sl_dfreq]))
            if self._sl_duty is not None:
                self._reward_terms["smooth_duty"] = float(-np.sum(spd[self._sl_duty]))
            self._reward_terms["smooth_amp"] = float(-np.sum(spd[self._sl_amp]))
            self._reward_terms["smooth_phase"] = float(-np.sum(spd[self._sl_phase]))
            self._reward_terms["smooth_res"] = float(-np.sum(spd[self._sl_res]))
            self._reward_terms["smooth_coff"] = float(-np.sum(spd[self._sl_coff]))

        self._reward_terms["res_cost"] = float(-res_cost)
        self._reward_terms["center_cost"] = float(-center_cost)
        return float(base - res_cost - center_cost)
