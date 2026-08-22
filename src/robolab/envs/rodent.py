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
    contact_w = 1.5     # penalty per NON-paw geom on the floor (0.5 was too soft:
                        #   it only migrated the skid from forearm -> upper arm)
    hand_w = 0.25       # BONUS per front-paw (hand) geom on the floor -- a carrot
                        #   for the missing behavior (it never planted its hands)
    energy_w = 5e-3     # penalty per unit mechanical power (|force*vel| summed) --
                        #   a cost-of-transport proxy. Lowered 1e-2 -> 5e-3 for
                        #   run 5: energy+smooth together caused the "stand still"
                        #   outcome in run 4; relieve it so movement can pay off.
    smooth_w = 3e-3     # penalty per unit sum-squared action delta. Lowered
                        #   0.01 -> 3e-3 because it DIRECTLY antagonizes the new
                        #   rhythm/phase terms (smoothness punishes the very
                        #   oscillation rhythm rewards) -- at 0.01 it would just
                        #   suppress the experiment into a null result.

    # --- EXPERIMENT (run 5): CPG-inspired rhythm + phase shaping ----------------
    # Hypothesis: rewarding a trot-like phase relationship between the limb swing
    # joints will NOT produce real locomotion -- it will be gamed. The point is to
    # SEE the failure mode. "Phase" is inherently a property over time, but our
    # reward is per-step, so we proxy it with the normalized correlation of limb
    # swing-joint VELOCITIES (same sign -> in phase, opposite -> anti phase).
    phase_w = 0.5       # weight on the trot phase-match score (normalized to [-1,1])
    rhythm_w = 0.3      # weight on "the limbs are actually oscillating" (0..1)
    rhythm_scale = 10.0 # rad/s scale: summed limb speed that saturates the tanh

    # --- forward reward + speed cap ---------------------------------------------
    # We want a good GAIT, not max speed. Reward forward progress linearly but
    # SATURATE it at a biologically reasonable walk/trot speed: vx above the cap
    # earns no extra reward, which also removes the incentive to gallop/bound (the
    # regime where "always plant paws" shaping would fight the natural aerial
    # phase). Backward motion (vx<0) is still penalized (min, not clip-to-0).
    forward_w = 5.0            # weight on (capped) forward speed
    forward_speed_cap = 0.6    # m/s; reward flat above this (~60 cm/s)

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

    def _swing_velocities(self):
        """(vFL, vFR, vHL, vHR): fore-aft swing-joint velocities of the four limbs.

        Forelimbs use `shoulder_*`, hindlimbs use `hip_*_extend` -- the joints whose
        axis is the body-y (pitch) axis, i.e. the ones that swing a limb front-back.
        """
        if not hasattr(self, "_swing_dofs"):
            names = ("shoulder_L", "shoulder_R", "hip_L_extend", "hip_R_extend")
            self._swing_dofs = [self.model.jnt_dofadr[self.model.joint(n).id]
                                for n in names]
        qv = self.data.qvel
        return (float(qv[self._swing_dofs[0]]), float(qv[self._swing_dofs[1]]),
                float(qv[self._swing_dofs[2]]), float(qv[self._swing_dofs[3]]))

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
        if self._prev_action is None:
            cost = 0.0
        else:
            d = (action - self._prev_action) / self._action_half
            cost = float(np.sum(self._smooth_weights() * np.square(d)))
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
        alive = 0.5                                          # was 1.0 (a floor, not the payout)
        control_cost = 0.001 * np.sum(np.square(action))
        upright_bonus = 0.2 * self.data.body("torso").xmat[8]  # was 1.0 (mostly rewarded posing)

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
        forward = self.forward_w * min(vx, self.forward_speed_cap)  # saturate at cap
        if self.gate_forward:
            # FLOOR the gate: attenuate forward out of stance, never annihilate it,
            # so a locomotion gradient always exists (gate_floor=0 -> old hard gate).
            forward = forward * (self.gate_floor + (1.0 - self.gate_floor) * g)

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

        # CPG-inspired rhythm + phase (the experiment). Trot target: diagonals in
        # phase, left-right anti phase. corr(a,b) = normalized velocity product in
        # [-1,1] (~sign agreement); ~0 when a joint is still, so the rat MUST move
        # its limbs to earn anything -- which is exactly what should get gamed.
        vFL, vFR, vHL, vHR = self._swing_velocities()

        def corr(a, b):
            return (a * b) / (abs(a) * abs(b) + 1e-6)

        phase_score = (corr(vFL, vHR) + corr(vFR, vHL)   # diagonals -> in phase
                       - corr(vFL, vFR) - corr(vHL, vHR)  # L/R pairs -> anti phase
                       ) / 4.0                            # -> [-1, 1]
        phase_reward = self.phase_w * phase_score

        speed = abs(vFL) + abs(vFR) + abs(vHL) + abs(vHR)
        rhythm_reward = self.rhythm_w * float(np.tanh(speed / self.rhythm_scale))

        if self.gate_rhythm:
            # Q1: pay for rhythm/phase only in stance, using the SAME ramp as
            # forward (floor -> attenuate, not annihilate, so a gradient survives).
            # Reuses g computed above; complements the residual cost (cheap OUT of
            # stance) so the two control modes hand off across the gate.
            ramp = self.gate_floor + (1.0 - self.gate_floor) * g
            phase_reward = phase_reward * ramp
            rhythm_reward = rhythm_reward * ramp

        # Stash raw per-step components; step() copies these into `info`.
        self._reward_terms = {
            "forward": float(forward),
            "alive": float(alive),
            "control_cost": float(-control_cost),
            "upright": float(upright_bonus),
            "contact": float(-contact_cost),
            "hand": float(hand_bonus),
            "energy": float(-energy_cost),
            "smooth": float(-smooth_cost),
            "phase": float(phase_reward),
            "rhythm": float(rhythm_reward),
            "bad_contacts": float(n_bad),
            "hand_contacts": float(n_hand),
            "foot_contacts": float(n_foot),
            "gate": float(g),
            "hand_prox": float(self._front_paw_proximity()),  # front-end-down signal
            "vx": float(vx),
        }
        return float(forward + alive - control_cost + upright_bonus
                     - contact_cost + hand_bonus - energy_cost - smooth_cost
                     + phase_reward + rhythm_reward)


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
    smooth_w_freq = 2.0     # weight on the (normalized) frequency-knob delta
    smooth_w_amp = 3e-3     # per-joint amplitude delta
    smooth_w_phase = 3e-3   # per-joint phase-offset delta
    smooth_w_res = 3e-3     # per-joint residual delta

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

    def __init__(self, xml_path, cpg_f_base=3.0, cpg_f_range=2.0,
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

        # Redefine the action space: [dfreq, amp(nu), phase(nu), residual(nu),
        # coff(nu)] -- the trailing coff group is the Q3 slow center offset.
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(1 + 4 * nu,), dtype=np.float32)

    def _get_obs(self):
        base = super()._get_obs()
        ph = getattr(self, "_phase", 0.0)
        # Append the CPG clock (sin/cos phi) + per-paw ground normal force (a real
        # afferent signal, not the derived gate g). Order: [base, clock(2), paws(4)].
        paws = self._paw_contact_forces()
        return np.concatenate([base, [np.sin(ph), np.cos(ph)], paws]).astype(np.float32)

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
        dfreq = float(action[0])
        amp = np.asarray(action[1:1 + nu], dtype=np.float64)
        psi = np.asarray(action[1 + nu:1 + 2 * nu], dtype=np.float64)
        res = np.asarray(action[1 + 2 * nu:1 + 3 * nu], dtype=np.float64)
        coff = np.asarray(action[1 + 3 * nu:1 + 4 * nu], dtype=np.float64)

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

        amp_pos = 0.5 * (amp + 1.0)  # [-1,1] -> [0,1] nonneg amplitude
        osc = self.cpg_amp_gain * self._half * amp_pos * np.sin(self._phase + np.pi * psi)
        residual = self.cpg_res_gain * self._half * res
        ctrl = self._center + self._coff_ema + osc + residual
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)

    def _smooth_weights(self):
        # Heterogeneous action = [dfreq(1), amp(nu), psi(nu), res(nu), coff(nu)].
        # Weight each group separately so the single frequency term isn't drowned by
        # the 4*nu per-joint terms -- this is what makes `smooth` regularize tempo.
        nu = self.model.nu
        w = np.empty(1 + 4 * nu, dtype=np.float64)
        w[0] = self.smooth_w_freq
        w[1:1 + nu] = self.smooth_w_amp
        w[1 + nu:1 + 2 * nu] = self.smooth_w_phase
        w[1 + 2 * nu:1 + 3 * nu] = self.smooth_w_res
        w[1 + 3 * nu:1 + 4 * nu] = self.smooth_w_coff
        return w

    def _get_reward(self, action):
        # Inherit all of RodentRunEnv's terms (it also fills self._reward_terms),
        # then subtract the residual-USAGE tax that's unique to this action space.
        base = super()._get_reward(action)

        nu = self.model.nu
        res = np.abs(np.asarray(action[1 + 2 * nu:1 + 3 * nu], dtype=np.float64))
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

        self._reward_terms["res_cost"] = float(-res_cost)
        self._reward_terms["center_cost"] = float(-center_cost)
        return float(base - res_cost - center_cost)
