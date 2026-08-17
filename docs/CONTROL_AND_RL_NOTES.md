# Control & RL Notes

Background/reference material for the playground, written to complement the
hands-on phases. Assumes a solid PID background and introductory RL exposure.

Contents:
1. LQR, deepened
2. Reinforcement Learning (history, framework, applications)
3. The through-line: PID -> LQR -> RL

---

## Part 1 - LQR, retroactively deepened

### Where it came from

LQR is a product of the **optimal control** revolution of ~1955-1965, driven by
the space race and Cold War (missiles, rockets, autopilots). Three threads
converged:

- **Richard Bellman (1957) - Dynamic Programming.** The **principle of
  optimality**: an optimal plan has the property that, whatever the first move,
  the rest must be optimal from the resulting state. This yields the **Bellman
  equation**, the recursive backbone of *both* LQR and RL.
- **Lev Pontryagin (1956) - the Maximum Principle.** The Soviet school's
  continuous-time optimality conditions.
- **Rudolf Kalman (1960) - "Contributions to the Theory of Optimal Control."**
  Gave us LQR in the form used here. The same year Kalman also gave us the
  **Kalman filter** (optimal state estimation). Together they form **LQG**
  (Linear-Quadratic-Gaussian): estimate the state optimally (Kalman filter),
  then act optimally on the estimate (LQR). Control and estimation are
  mathematical mirror images (duality). Apollo flew LQG-style methods; the
  Kalman filter ran on the Apollo Guidance Computer.

### Why LQR is special (why it's closed-form)

General optimal control is infinite-dimensional (optimize over whole
trajectories). LQR is the special case where it collapses to an exact solution,
because of a precise marriage:

| Ingredient | Choice in LQR | Consequence |
|---|---|---|
| Dynamics | **Linear** (x_{t+1}=A x + B u) | superposition holds |
| Cost | **Quadratic** (x'Qx + u'Ru) | smooth, convex, one minimum |
| Result | | optimal policy is **linear**: u = -K x |

The optimal controller for a linear-quadratic problem is just a **constant gain
matrix** - a matrix multiply, no lookup table. The Riccati equation folds the
infinite-horizon Bellman recursion into a single algebraic equation for the
cost-to-go matrix P, from which K drops out.

The **cost-to-go** V(x) = x'Px is the total future cost from state x if you act
optimally forever. That "how good is this state" function is the **value
function** - the same object RL spends all its effort *learning* because it
can't solve for it in closed form.

### LQR vs. PID intuition

On the cartpole:
- **PD**: u = -(kp*theta + kd*theta_dot) - feedback on the pole only, gains guessed.
- **LQR**: u = -(k1*x + k2*theta + k3*x_dot + k4*theta_dot) - feedback on all
  four states, gains derived.

So LQR is the principled multivariable generalization of PD tuning. A hand-tuned
PD is *some* LQR solution - for a particular implied Q, R. LQR makes that choice
explicit and optimal.

Honest gap: classic LQR has **no integral action**, so it doesn't reject
constant disturbances/biases the way PID's *I* term does. Practitioners add
integral states to the model to recover it (servo LQR / LQR with integral
action).

### Practical facets

- **Q and R are the whole design.** Big Q_i = care about state i; big R =
  penalize effort. **Bryson's rule**: set each diagonal weight to
  1/(max acceptable value)^2 to normalize differing units.
- **Finite vs infinite horizon.** Infinite-horizon -> constant K (steady-state
  Riccati). Finite-horizon -> time-varying K_t from a backward recursion (e.g.
  a fixed-duration maneuver like a rocket landing burn).
- **Local validity.** Linearization about an operating point means guarantees
  only *near* it. Great at 0.7 rad, hopeless from horizontal.

### Real-world LQR/LQG

Aircraft autopilots / fly-by-wire, spacecraft attitude control, rocket landing
(related convex/optimal control), self-balancing robots (Segway), quadrotor
stabilization, hard-disk head positioning, active suspension, cruise control,
camera gimbals, and countless process-control loops.

### The limitation that opens the door to RL

LQR needs three things you often don't have:
1. a **model** (A, B) - hard for real robots with contact/friction;
2. **linearity** - hopeless for a biped mid-stride (contacts -> nonlinear,
   discontinuous dynamics);
3. a **quadratic cost** - "walk forward without falling" isn't naturally quadratic.

Remove those (unknown nonlinear dynamics, arbitrary reward) and you've defined
the RL problem.

---

## Part 2 - Reinforcement Learning

### Three historical threads (Sutton & Barto framing)

1. **Optimal control (1950s-60s)** - Bellman, dynamic programming, MDPs. Same
   math as LQR, but for general (nonlinear, stochastic) systems where you can't
   solve Bellman in closed form, so you *approximate* it. RL is **approximate
   dynamic programming**.
2. **Trial-and-error learning (psychology)** - Thorndike's *Law of Effect*
   (1911): actions followed by satisfaction are repeated. The "reinforcement" in
   the name; maps onto dopamine in the brain.
3. **Temporal-difference (TD) learning** - Sutton (1988), building on Arthur
   Samuel's self-learning **checkers** program (1959). TD bootstraps a value
   estimate from later value estimates without waiting for the final outcome.

### The framework - Markov Decision Process (MDP)

- **State** s (the observation)
- **Action** a
- **Transition** P(s'|s,a) (the dynamics; unknown to the agent = MuJoCo)
- **Reward** r(s,a) (you design this)
- **Discount** gamma in [0,1)

Goal: a **policy** pi(a|s) maximizing expected **return** G_t = sum_k gamma^k r_{t+k}.

Central objects (same ideas as LQR's cost-to-go, flipped cost->reward):
- **Value** V^pi(s) - expected return from s under pi. (LQR's x'Px, learned.)
- **Action-value** Q^pi(s,a) - expected return from taking a in s, then pi.

They obey the **Bellman equation** - the same recursion as LQR. RL estimates
these when you can't compute them.

### The family tree

| Axis | Options | PPO / SAC |
|---|---|---|
| Model | model-free vs model-based | both **model-free** |
| What's learned | value / policy / actor-critic | both **actor-critic** |
| Data usage | on-policy vs off-policy | **PPO on-policy**, **SAC off-policy** |

- **Value-based** (Q-learning, DQN): learn Q, act greedily. Great for discrete
  actions, awkward for continuous torque.
- **Policy-gradient** (REINFORCE, Williams 1992): directly nudge policy toward
  higher return. Natural for continuous control.
- **Actor-critic**: policy (actor) improved using a learned value fn (critic) to
  cut variance. Dominant modern design.

**On-policy vs off-policy:**
- **PPO (OpenAI 2017)**: on-policy, learns from current-policy data then discards
  it. Robust, forgiving default; "clipped" objective prevents too-large updates.
  Stable, parallelizes well, but sample-hungry.
- **SAC (Haarnoja 2018)**: off-policy, reuses old data from a **replay buffer**,
  so far more **sample-efficient** (key for real robots). Uses **maximum-entropy
  RL**: rewarded for being as random as possible while still succeeding -> great
  exploration/robustness. More hyperparameters, can be finickier.

Plan: PPO first (robust win, learn to read curves), SAC when sample efficiency
matters on the walker.

### Deep-RL timeline

- **1992 - TD-Gammon** (Tesauro): NN + TD + self-play -> near-champion backgammon;
  invented strategies experts adopted.
- **2013/2015 - DQN** (DeepMind): one architecture plays 49 Atari games from
  pixels, human-level on many. Tricks: experience replay + target networks.
- **2016 - AlphaGo** beats Lee Sedol; **AlphaGo Zero (2017)** learns from scratch
  by self-play; **AlphaZero** generalizes to chess/shogi; **MuZero (2020)** learns
  without being told the rules (model-based).
- **2017 - PPO**, **2018 - SAC**: continuous-control workhorses, benchmarked on
  **MuJoCo** (this playground's engine; hopper/walker/cheetah/humanoid are the
  field's standard tests).
- **2019 - OpenAI Five** (Dota 2), **AlphaStar** (StarCraft II): RL at scale in
  real-time strategy.
- **2019 - Dexterous manipulation**: OpenAI robot hand solves a Rubik's Cube;
  sim-to-real via domain randomization.
- **2022 - Tokamak plasma control** (DeepMind + EPFL, Nature): RL controls the
  magnetic coils of a real fusion reactor.
- **2020s - RLHF**: Reinforcement Learning from Human Feedback aligns LLMs
  (e.g. ChatGPT) - a reward model on human preferences, optimized with **PPO**.

### Application areas

Games (benchmarks + strategy discovery), robotics (locomotion, manipulation,
drones, quadrupeds), datacenter cooling (Google/DeepMind), chip floorplanning
(Google TPU), fusion, recommendation/ads, autonomous driving (planning/sim),
operations/inventory, finance (execution), traffic control, and LLM alignment
(RLHF/RLAIF).

### The NeuroAI hook

The **reward-prediction-error** at the heart of TD learning matches the firing
of **dopamine neurons** (Schultz, Dayan, Montague, 1997) - the brain appears to
run a TD-like algorithm; the basal ganglia look like a biological actor-critic.
So for Phase 6, RL is part of the same story as the neural controllers: CPG
(spinal rhythm) + RL (dopaminergic learning) + place cells (RatInABox).

---

## Part 3 - The through-line: PID -> LQR -> RL

| | PID | LQR | RL (PPO/SAC) |
|---|---|---|---|
| Needs a model? | No | **Yes** (linear A,B) | **No** (learns from interaction) |
| Handles nonlinear? | Poorly | Locally only | **Yes** |
| Objective | implicit (kill error) | quadratic cost | **any reward you design** |
| Optimality | none guaranteed | **provably optimal** (LQ) | approximately optimal |
| Output | fixed gains | fixed gain matrix K | learned policy (often a NN) |
| Cost | trivial compute | one Riccati solve | **millions of trials** |
| Seen in playground | car steering | cartpole balance | Phase 5 |

Core trade: RL **relaxes the assumptions** (no model, nonlinear, arbitrary
reward) at the **price of sample efficiency and guarantees**. LQR gives a
certificate and a matrix in milliseconds *if* the world is linear and known. RL
gives a controller for a world you can only poke at - but needs a million pokes
and can't prove it's right.
