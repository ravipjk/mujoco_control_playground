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
