"""Simulation-only viewer/MP4 helpers; render fields never change physics."""

from contextlib import ExitStack
import os
import sys
import numpy as np
import mujoco


def presentation_style(model):
    """Visual fields only: brighter lighting and distinct front/rear leg colors."""
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    model.vis.headlight.specular[:] = [0.1, 0.1, 0.1]
    for g in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[g])).name
        color = (
            [0.82, 0.84, 0.87, 1.0]
            if not model.geom_bodyid[g]
            else [0.48, 0.53, 0.62, 1.0]
        )
        if name.startswith(("FL_", "FR_")):
            color = [0.1, 0.5, 0.95, 1.0]
        if name.startswith(("RL_", "RR_")):
            color = [0.95, 0.42, 0.08, 1.0]
        model.geom_rgba[g] = color
        model.geom_matid[g] = -1


def ground_grid(scene):
    for axis in (0, 1):
        for value in np.arange(-1.5, 1.51, 0.25):
            a = np.array([-1.5, value, 0.001])
            b = np.array([1.5, value, 0.001])
            if axis:
                a[[0, 1]] = a[[1, 0]]
                b[[0, 1]] = b[[1, 0]]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array([0.48, 0.51, 0.56, 1.0]),
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.0, a, b)
            scene.ngeom += 1


class Presentation:
    def __init__(self, robot, viewer=False, video=None, fps=30, width=640, height=480):
        self.robot = robot
        presentation_style(robot.m)
        self.stack = ExitStack()
        self.frames = 0
        self.next_frame = 0.0
        self.fps = fps
        self.renderer = self.writer = self.viewer = None
        try:
            if viewer:
                if sys.platform == "linux" and not os.environ.get("DISPLAY"):
                    raise ValueError(
                        "No DISPLAY for --viewer; use MUJOCO_GL=egl with --video"
                    )
                from mujoco import viewer as mjviewer

                self.viewer = self.stack.enter_context(
                    mjviewer.launch_passive(robot.m, robot.d)
                )
                with self.viewer.lock():
                    ground_grid(self.viewer.user_scn)
                self.viewer.cam.distance = 1.6
                self.viewer.cam.azimuth = 130
                self.viewer.cam.elevation = -15
            if video is not None:
                try:
                    import imageio.v2 as imageio
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "Install scripts/handstand_video_requirements.txt for --video"
                    ) from exc

                robot.m.vis.global_.offwidth = max(robot.m.vis.global_.offwidth, width)
                robot.m.vis.global_.offheight = max(
                    robot.m.vis.global_.offheight, height
                )
                self.renderer = mujoco.Renderer(robot.m, height=height, width=width)
                self.stack.callback(self.renderer.close)
                self.writer = imageio.get_writer(
                    str(video), fps=fps, codec="libx264", macro_block_size=1
                )
                self.stack.callback(self.writer.close)
                self.camera = mujoco.MjvCamera()
                self.camera.distance = 1.6
                self.camera.azimuth = 130
                self.camera.elevation = -15
            self.capture(force=True)
        except BaseException:
            self.stack.close()
            raise

    def capture(self, force=False):
        r = self.robot
        if self.viewer and self.viewer.is_running():
            self.viewer.cam.lookat[:] = r.d.subtree_com[r.base]
        if self.renderer and (force or r.d.time + 1e-10 >= self.next_frame):
            self.camera.lookat[:] = r.d.subtree_com[r.base]
            self.renderer.update_scene(r.d, camera=self.camera)
            ground_grid(self.renderer.scene)
            self.writer.append_data(self.renderer.render().copy())
            self.frames += 1
            self.next_frame += 1 / self.fps

    def sync(self):
        if self.viewer:
            if not self.viewer.is_running():
                return False
            self.viewer.sync()
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        self.stack.close()
