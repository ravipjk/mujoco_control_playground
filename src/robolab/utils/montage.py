"""Record a "learning montage": periodically roll out the current policy during
training, label each frame with progress, and save one combined video so the
stages of learning can be watched end to end.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stable_baselines3.common.callbacks import BaseCallback

from robolab.core.rollout import rollout
from robolab.controllers.rl import PolicyController
from robolab.utils.video import save_video


def _load_font(size):
    # Try a real TrueType font for readable text; fall back to PIL's bitmap font.
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_frame(frame, text, font=None):
    """Draw `text` (top-left, on a dark bar) onto an RGB uint8 frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = font or _load_font(max(14, img.height // 22))
    # dark background bar for legibility
    bbox = draw.textbbox((8, 6), text, font=font)
    draw.rectangle([0, 0, bbox[2] + 8, bbox[3] + 6], fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 0), font=font)
    return np.asarray(img)


class LearningMontageCallback(BaseCallback):
    """Every `record_freq` timesteps, roll out the current (deterministic) policy
    in a dedicated render env, label frames with timestep + episode reward, and
    accumulate them. On training end, write one montage video.

    env_fn: zero-arg callable returning a fresh env to render in.
    """

    def __init__(self, env_fn, record_freq=100_000, camera=None, track_body=None,
                 max_steps=300, fps=50, out_path="outputs/learning_montage.mp4",
                 verbose=0):
        super().__init__(verbose)
        self.env_fn = env_fn
        self.record_freq = record_freq
        self.camera = camera
        self.track_body = track_body
        self.max_steps = max_steps
        self.fps = fps
        self.out_path = out_path
        self.frames = []
        self._next_record = 0
        self._render_env = None

    def _on_training_start(self):
        self._render_env = self.env_fn()
        # On a RESUME (learn(reset_num_timesteps=False)) num_timesteps starts high,
        # but our schedule starts at _next_record=0. Without this, _on_step "catches
        # up" the 0-based schedule and dumps a burst of duplicate snapshots (one per
        # step) until it passes num_timesteps. Anchor the schedule to now so a resume
        # records a single baseline snapshot, then every record_freq thereafter.
        if self._next_record < self.num_timesteps:
            self._next_record = self.num_timesteps

    def _on_step(self):
        if self.num_timesteps >= self._next_record:
            self._record_snapshot()
            self._next_record += self.record_freq
        return True

    def _record_snapshot(self):
        controller = PolicyController(self.model, deterministic=True)
        result = rollout(self._render_env, controller, render=True,
                         camera=self.camera, track_body=self.track_body,
                         max_steps=self.max_steps)
        label = f"steps={self.num_timesteps:,}   reward={result['total_reward']:.0f}   len={result['steps']}"
        for f in result["frames"]:
            self.frames.append(annotate_frame(f, label))
        if self.verbose:
            print(f"[montage] snapshot @ {self.num_timesteps:,} steps -> {label}")

    def _on_training_end(self):
        if self.frames:
            save_video(self.frames, self.out_path, fps=self.fps)
            if self.verbose:
                print(f"[montage] saved {len(self.frames)} frames to {self.out_path}")
