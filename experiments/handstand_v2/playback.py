"""Render a trusted exported policy with full MuJoCo dynamics to an animated GIF."""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import mujoco
from PIL import Image
import torch
import curriculum
from experiments.handstand_v2.environment import V2Environment
from experiments.morphology.learning.environment import Assets
from experiments.handstand_v2.run import ORIGINAL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=ORIGINAL)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    net, hp, cfg, _ = curriculum.load_source(ORIGINAL)
    net.load_state_dict(
        torch.load(a.checkpoint, map_location="cpu", weights_only=False)["network"]
    )
    env = V2Environment(Assets(a.output.parent / "playback_assets"), cfg)
    actor, _ = env.reset(1.0, 70400)
    renderer = mujoco.Renderer(env.m, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.3]
    camera.distance = 1.7
    camera.azimuth = 90
    camera.elevation = -15
    images = []
    while True:
        renderer.update_scene(env.d, camera=camera)
        images.append(Image.fromarray(renderer.render().copy()))
        with torch.no_grad():
            action = net.actor(torch.tensor(actor)).tanh().numpy()
        (actor, _), _, done, _ = env.step(action)
        if done:
            break
    images[0].save(
        a.output, save_all=True, append_images=images[1:], duration=20, loop=0
    )
    renderer.close()
    print(a.output)


if __name__ == "__main__":
    main()
