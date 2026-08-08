"""Tiny video helper so every script/notebook records rollouts the same way."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np


def save_video(frames: Sequence[np.ndarray], path: str | Path, fps: int = 30) -> Path:
    """Write a list of RGB uint8 frames (H, W, 3) to an mp4 file.

    Parameters
    ----------
    frames : sequence of arrays shaped (H, W, 3), dtype uint8
        Rendered frames, e.g. from mujoco.Renderer.render().
    path : where to save (parent directories are created automatically).
    fps : playback frame rate.

    Returns
    -------
    Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    return path
