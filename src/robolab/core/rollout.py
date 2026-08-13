import mujoco
import numpy as np

from robolab.utils.video import save_video


def rollout(env, controller, max_steps=None, render=False, camera=None, track_body=None):
    """Run one episode: the controller drives the env until done (or max_steps).

    Returns a dict:
      'rewards'      -> list of per-step rewards
      'total_reward' -> float sum
      'steps'        -> number of steps taken
      'frames'       -> list of RGB frames (empty unless render=True)

    Rendering (only if render=True): create one mujoco.Renderer(env.model);
    each step call update_scene(env.data, camera=...) then render().
    `camera` may be a model camera name/id or None (free cam). If `track_body`
    is given, follow that body with an MjvCamera (reuse your Phase 1 camera code).
    """
    obs, info = env.reset()
    controller.reset()

    rewards = [];  frames = [];  steps = 0

    if render:
        width, height = 640, 480
        renderer = mujoco.Renderer(env.model, height=height, width=width)

    if track_body is not None:
        cam = mujoco.MjvCamera()
        cam.lookat = np.array([0.0, 0.0, 0.0])
        cam.distance = 2.0
        cam.azimuth = 90.0
        cam.elevation = -20.0
    else:
        cam = camera if camera is not None else -1

    if max_steps is None:
        max_steps = env.episode_length

    for _ in range(max_steps):
        action = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        steps +=1

        if render:
            if track_body is not None: 
                car_pos = env.data.body(track_body).xpos
                cam.lookat[:] = car_pos

            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render())

        if terminated or truncated:
            break
    
    return {"rewards": rewards, "total_reward": float(sum(rewards)), "steps": steps, "frames": frames}


