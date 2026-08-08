# AGENTS.md

Guidance for AI agents (and humans) working in this repository. This is a living
document; expand it as the project grows.

## What this project is

`robolab` (the "MuJoCo Control Playground") is a hands-on playground for dropping
**controllers the user designs** into **walkers and wheeled robots** simulated in
[MuJoCo](https://mujoco.org/). Its purposes, in priority order:

1. **Learning by building** — control theory, dynamical systems, reinforcement
   learning, implemented from fundamentals.
2. **A testing ground** for a separate NeuroAI project (spatial navigation &
   memory; continuous-attractor / path-integration models).

Core design principle: the **environment (robot + physics)** and the
**controller** are decoupled. A controller just maps an observation to an action,
so hand-tuned (PD/LQR), dynamical-systems (CPG), RL, and neural controllers are
all interchangeable behind one small interface.

## How to work here: the pedagogical contract

This is the most important section. The user is learning, so **teaching takes
priority over speed**.

- **Guide; don't just do.** For anything that is a *learning step* (control loops,
  controllers, environment logic, algorithms), explain the concept and the goal,
  then let the user implement it (see *Teaching method* below for how).
- **Distinguish infra from learning.** Tedious boilerplate (installing packages,
  wiring config) may be done for the user, but *call it out* and explain the
  reasoning. When in doubt, ask whether they want to implement it themselves.
- **Always explain the "why."** State the motivation and trade-offs behind design
  choices, not just the "what."
- **Go step by step.** One concept at a time. Check understanding before moving
  on; invite the user to predict outcomes.
- **Meet the user's background.** Strong in neuroscience/domain; still building
  software-engineering fundamentals. Don't shy away from more optimal/advanced
  designs, but teach the fundamentals behind them.
- **Ask when ambiguous.** Prefer a clarifying question over guessing on decisions
  that meaningfully change direction.

### Teaching method (how to run a learning step)

- Favor retrieval over recognition: give a spec + an API "toolbox" (names only),
  not fill-in-the-blank code the user can just uncomment.
- Discovery before answers: point the user to `help()`, `dir()`, docs, and REPL
  exploration so they find the API themselves.
- Predict, then run: ask the user to predict the outcome before executing.
- Graduated hints on request: (1) conceptual nudge, (2) code structure in prose,
  (3) literal answer -- and only give level 3 after a genuine attempt.
- Review like a mentor: point at what to reconsider; don't silently rewrite the
  user's code.
- Treat errors as feedback to reason through together, not chores to fix for them.
- Know when to exit teaching mode: if the user explicitly says to just implement
  it, comply -- but still explain the "why" afterward.

## Tech stack & environment

- **Platform:** native Windows, NVIDIA GPU (RTX 4080 Laptop). Default shell:
  PowerShell.
- **Python:** 3.12, managed by [`uv`](https://docs.astral.sh/uv/) in a project
  `.venv`. Do not rely on the system Python (3.13).
- **Key deps:** `mujoco`, `gymnasium`, `stable-baselines3`, `torch` (CUDA 12.4
  build), `numpy`, `scipy`, `matplotlib`, `imageio`.
- **Run things via uv**, e.g. `uv run python scripts/...`. This guarantees the
  project's environment is used.

## Repository layout

```
src/robolab/
  core/         controller interface, MuJoCo env base, rollout runner
  envs/         concrete robots (cartpole, car, walker, ...) + registry
  controllers/  hand-designed / dynamical / RL / neuro controllers
  utils/        plotting, video, logging helpers
  assets/       self-contained MuJoCo model .xml files
scripts/        runnable demos (numbered by learning phase)
notebooks/      exploratory / pedagogical notebooks
tests/          sanity checks
```

## Conventions

- Prefer **self-contained MJCF** assets (no hidden `<include>` files) for clarity.
- Reference assets via `robolab.ASSETS_DIR`, never hard-coded paths.
- Keep the `Controller` side free of MuJoCo imports (that's what keeps them
  swappable).
- <!-- TODO: expand as conventions emerge (naming, testing, style, commits). -->

## Learning roadmap (high level)

Phase 0 setup -> 1 hello MuJoCo -> 2 core abstractions -> 3 control (PD/LQR, PID)
-> 4 dynamical systems (CPG) -> 5 RL (PPO/SAC) -> 6 NeuroAI bridge (RatInABox +
ring attractor). See the chat/plan for current status.

## Related repositories (siblings on this machine)

- `neuro_model_project` — the NeuroAI project this playground supports.
- `RatInABox` — place/grid/head-direction cell models (Phase 6 bridge).
- `dm_control`, `mujoco`, `mujoco_playground` — upstream references for models
  and APIs.
