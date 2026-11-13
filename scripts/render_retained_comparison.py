"""Four exact-state attempts, a common fixed side camera, no invented motion."""

import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"),
)
import mujoco, numpy as np
from PIL import Image, ImageDraw
from scripts.unified_go2 import setup
from experiments.unified_go2.core import Environment
from scripts.render_adc_front_capture import marker


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg, data, bank, assets = setup(
        Path("output/retained_go2_preparation_v3"), a.output
    )
    e = Environment(assets, cfg)
    sources = [
        ("ADC", "output/retained_go2_validation_v1/videos/adc_standing_0_states.npz"),
        (
            "Previous student",
            "output/unified_go2_evaluation_initial_v2/videos/student_standing_0_states.npz",
        ),
        (
            "Corrected student",
            "output/retained_go2_validation_v3/videos/student_standing_0_states.npz",
        ),
        (
            "After numerical PPO smoke",
            "output/retained_go2_post_smoke_v1/videos/student_standing_0_states.npz",
        ),
    ]
    runs = [(label, np.load(path)["integration"]) for label, path in sources]
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    xs = []
    for label, states in runs:
        assert states.shape[1] == mujoco.mj_stateSize(e.m, spec)
        for state in states[::4]:
            mujoco.mj_setState(e.m, e.d, state, spec)
            xs.append(e.d.qpos[0])
    low, high = min(xs) - 0.7, max(xs) + 0.7
    camera = mujoco.MjvCamera()
    camera.azimuth = 90
    camera.elevation = 0
    camera.lookat[:] = [(low + high) / 2, 0, 0.5]
    camera.distance = max(2.8, (high - low) / (2 * np.tan(np.deg2rad(22.5)) * 4 / 3))
    renderer = mujoco.Renderer(e.m, height=300, width=400)
    frames = []
    individual = [[] for _ in runs]
    indices = list(range(0, max(len(x[1]) for x in runs), 2))
    for index in indices:
        canvas = Image.new("RGB", (800, 600))
        for panel, (label, states) in enumerate(runs):
            actual = min(index, len(states) - 1)
            mujoco.mj_setState(e.m, e.d, states[actual], spec)
            mujoco.mj_forward(e.m, e.d)
            renderer.update_scene(e.d, camera=camera)
            com = e.d.subtree_com[e.base].copy()
            marker(renderer.scene, com, 0.018, [1, 0, 0, 1])
            for contact in e.d.contact:
                if contact.dist <= 0:
                    bad = any(
                        b != 0 and b not in e.feet
                        for b in e.m.geom_bodyid[[contact.geom1, contact.geom2]]
                    )
                    marker(
                        renderer.scene,
                        contact.pos,
                        0.018,
                        [1, 0, 0, 1] if bad else [0, 1, 0, 1],
                    )
            im = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(im)
            cam = renderer.scene.camera[0]
            relative = com - cam.pos
            depth = relative @ cam.forward
            if depth > 0:
                focal = 300 * cam.frustum_near / (cam.frustum_top - cam.frustum_bottom)
                x = 200 + focal * (relative @ np.cross(cam.forward, cam.up)) / depth
                y = 150 - focal * (relative @ cam.up) / depth
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red", outline="white")
            draw.rectangle((0, 0, 400, 30), fill="black")
            draw.text((4, 3), label, fill="white")
            draw.text(
                (4, 16), f"Actual MuJoCo time {e.d.time:.2f}s | nominal", fill="white"
            )
            if index >= len(states):
                draw.text((4, 282), "EPISODE ENDED: final frame held", fill="red")
            individual[panel].append(im.convert("P", palette=Image.Palette.ADAPTIVE))
            canvas.paste(im, ((panel % 2) * 400, (panel // 2) * 300))
        frames.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE))
    renderer.close()
    for i, images in enumerate(individual):
        images[0].save(
            a.output / f"attempt_{i}.gif",
            save_all=True,
            append_images=images[1:],
            duration=40,
            loop=0,
        )
    frames[0].save(
        a.output / "comparison.gif",
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
    )
    with Image.open(a.output / "comparison.gif") as im:
        im.seek(im.n_frames - 1)
        im.load()
        print("Verified", im.n_frames, "native comparison frames", flush=True)


if __name__ == "__main__":
    main()
