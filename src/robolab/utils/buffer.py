"""Periodic replay-buffer checkpointing for off-policy agents (SAC/TD3/DDPG).

SB3's built-in CheckpointCallback can save the buffer too, but it writes a NEW
timestamped file every time -- and an off-policy buffer is often hundreds of MB, so
keeping one per checkpoint balloons disk fast. This callback instead saves to a
SINGLE rolling path, OVERWRITING it each time, so you always have "the buffer as of
the latest save" for a seamless Ctrl-C resume, at fixed disk cost.

The rolling file is paired (by naming) with the run's model so the resume script can
find it; a small policy/buffer age mismatch is harmless for off-policy learning (the
buffer is just experience, not tied to the exact policy snapshot).
"""
from stable_baselines3.common.callbacks import BaseCallback


class ReplayBufferSaver(BaseCallback):
    """Every ``save_freq`` env steps, overwrite ``save_path`` with the current buffer.

    Args:
        save_freq: interval in env steps (matches CheckpointCallback semantics for a
            single env; assumes the 1-env setup used by scripts 16/18).
        save_path: path WITHOUT extension; SB3 appends ``.pkl`` (so it stays in sync
            with ``model.save_replay_buffer(<same path>)`` used at end-of-run).
    """

    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            self.model.save_replay_buffer(self.save_path)
            if self.verbose:
                n = self.model.replay_buffer.size()
                print(f"[ReplayBufferSaver] {self.save_path}.pkl "
                      f"({n} transitions) @ {self.num_timesteps} steps")
        return True
