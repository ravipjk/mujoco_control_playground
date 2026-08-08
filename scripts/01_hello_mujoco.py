"""Phase 1 - Hello, MuJoCo.

Load the wheeled `car`, drive it OPEN-LOOP (action depends only on time), and
record a video. Run with:  uv run python scripts/01_hello_mujoco.py
"""

import mujoco 
import numpy as np
import robolab
from robolab.utils.video import save_video

def main_car():
    model = mujoco.MjModel.from_xml_path(str(robolab.ASSETS_DIR / "car.xml"))

    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([0.0, 0.0, 0.0])
    cam.distance = 2.0
    cam.azimuth = 90.0
    cam.elevation = -20.0

    print(model.opt.timestep, model.nq, model.nv, model.nu)

    width, height = 640, 480
    renderer = mujoco.Renderer(model, height=height, width=width)

    steps = int(6.0 / model.opt.timestep)
    skip = max(1, round((1.0/30) / model.opt.timestep))

    frames:list[np.ndarray] = []

    for step in range(steps):
        t = step * model.opt.timestep
        data.ctrl[0] = 1.0
        data.ctrl[1] = 0.3 * np.sin(2.0*t)
        mujoco.mj_step(model, data)
        if step % skip ==0:
            car_pos = data.body("car").xpos
            cam.lookat = car_pos
            renderer.update_scene(data, cam)
            frames.append(renderer.render())

    print(data.qpos[:3])
    out = save_video(frames, "outputs/01_hello_mujoco.mp4", fps=30)
    print(f"Saved -> {out.resolve()}")
    
if __name__ == "__main__":
    main_car()
    