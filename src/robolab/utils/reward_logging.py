"""A tiny SB3 callback that averages per-step reward components (surfaced by the
env in its `info` dict) and logs them to TensorBoard as separate curves.

This lets you watch WHICH part of the reward the policy is actually harvesting
(forward progress vs alive vs upright vs control cost) instead of inferring it
from ep_rew_mean / ep_len_mean.
"""
from stable_baselines3.common.callbacks import BaseCallback


class RewardTermLoggerCallback(BaseCallback):
    """Accumulates named scalars from each env `info` and records their mean to
    the SB3 logger under `reward/<key>`, flushed once per rollout.

    term_keys: the info keys to track (e.g. ["forward", "alive", "vx", ...]).
    """

    def __init__(self, term_keys, verbose=0):
        super().__init__(verbose)
        self.term_keys = list(term_keys)
        self._sums = {k: 0.0 for k in self.term_keys}
        self._count = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            self._count += 1
            for k in self.term_keys:
                v = info.get(k)
                if v is not None:
                    self._sums[k] += v
        return True

    def _flush(self):
        if self._count == 0:
            return
        for k in self.term_keys:
            self.logger.record(f"reward/{k}", self._sums[k] / self._count)
        self._sums = {k: 0.0 for k in self.term_keys}
        self._count = 0

    def _on_rollout_end(self):
        self._flush()

    def _on_training_end(self):
        self._flush()
