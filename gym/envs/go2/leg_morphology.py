"""Opt-in axial leg shortening with constant density; see morphology docs."""

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from gym import GYM_ROOT_DIR

SCALES = (0.80, 0.85, 0.90, 0.95, 1.00)
SOURCE = Path(GYM_ROOT_DIR) / "resources/robots/go2/urdf/go2.urdf"


def _vector(text):
    return np.fromstring(text, sep=" ")


def _text(value):
    return " ".join(format(float(x), ".17g") for x in value)


def _rotation(rpy):
    r, p, y = _vector(rpy)
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def axial_inertia(inertia, transform, mass_ratio):
    """Transform the central second moment, then recover the inertia tensor."""
    second = np.trace(inertia) * np.eye(3) / 2 - inertia
    second = mass_ratio * transform @ second @ transform.T
    return np.trace(second) * np.eye(3) - second


def _collision_mesh(geometry, origin, transform, path):
    """Exact affine boxes; closed 128-sided approximation to affine cylinders."""
    geom = geometry[0]
    if geom.tag == "box":
        from itertools import product

        vertices = (
            np.array(list(product((-1, 1), repeat=3))) * _vector(geom.get("size")) / 2
        )
        faces = [
            (0, 1, 3),
            (0, 3, 2),
            (4, 6, 7),
            (4, 7, 5),
            (0, 4, 5),
            (0, 5, 1),
            (2, 3, 7),
            (2, 7, 6),
            (0, 2, 6),
            (0, 6, 4),
            (1, 5, 7),
            (1, 7, 3),
        ]
    elif geom.tag == "cylinder":
        n = 128
        angles = np.arange(n) * 2 * np.pi / n
        radius, half = float(geom.get("radius")), float(geom.get("length")) / 2
        circle = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
        vertices = np.r_[
            np.c_[circle, np.full(n, -half)], np.c_[circle, np.full(n, half)]
        ]
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.extend([(i, j, n + j), (i, n + j, n + i)])
        for i in range(1, n - 1):
            faces.extend([(0, i + 1, i), (n, n + i, n + i + 1)])
    else:
        raise ValueError(f"Unsupported leg collision primitive: {geom.tag}")
    vertices = (
        vertices @ _rotation(origin.get("rpy", "0 0 0")).T
        + _vector(origin.get("xyz", "0 0 0"))
    ) @ transform.T
    with path.open("x") as f:
        for vertex in vertices:
            f.write("v " + _text(vertex) + "\n")
        for face in faces:
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")
    geometry.remove(geom)
    ET.SubElement(geometry, "mesh", filename=str(path.resolve()))
    origin.set("xyz", "0 0 0")
    origin.set("rpy", "0 0 0")


def generate(scale, destination):
    """Axially shorten thigh/calf assemblies, preserving transverse dimensions.

    A=diag(1,1,scale) in each segment frame. Fixed calf collision children
    inherit that segment's map through their original rigid transforms.
    Density is constant: m'=s*m, COM'=A*COM, C'=s*A*C*A.T.
    Unchanged motor/rotor bodies and feet are translated but never scaled.
    Scale 1 is an exact byte copy of the original URDF.
    """
    import mujoco

    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    destination = Path(destination)
    if destination.resolve() == SOURCE.resolve() or destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if scale == 1:
        shutil.copyfile(SOURCE, destination)
        return destination
    root = ET.parse(SOURCE).getroot()
    original = build_model(SOURCE)
    maps = {}
    axial = np.diag([1.0, 1.0, scale])
    for leg in ("FL", "FR", "RL", "RR"):
        maps[f"{leg}_thigh"] = axial
        maps[f"{leg}_calf"] = axial
        for part in ("calflower", "calflower1"):
            joint = root.find(f"joint[@name='{leg}_{part}_joint']")
            parent = joint.find("parent").get("link")
            rotation = _rotation(joint.find("origin").get("rpy", "0 0 0"))
            maps[f"{leg}_{part}"] = rotation.T @ maps[parent] @ rotation
    meshdir = destination.parent / (destination.stem + "_collisions")
    meshdir.mkdir(exist_ok=False)
    for link in root.findall("link"):
        name = link.get("name")
        if name not in maps:
            continue
        transform = maps[name]
        body = original.body(name)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, body.iquat)
        rotation = rotation.reshape(3, 3)
        tensor = rotation @ np.diag(body.inertia) @ rotation.T
        tensor = axial_inertia(tensor, transform, scale)
        old = link.find("inertial")
        if old is not None:
            link.remove(old)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=_text(transform @ body.ipos), rpy="0 0 0")
        ET.SubElement(
            inertial, "mass", value=str(float(original.body_mass[body.id]) * scale)
        )
        ET.SubElement(
            inertial,
            "inertia",
            **{
                key: format(tensor[i, j], ".17g")
                for key, i, j in (
                    ("ixx", 0, 0),
                    ("ixy", 0, 1),
                    ("ixz", 0, 2),
                    ("iyy", 1, 1),
                    ("iyz", 1, 2),
                    ("izz", 2, 2),
                )
            },
        )
        for i, collision in enumerate(link.findall("collision")):
            _collision_mesh(
                collision.find("geometry"),
                collision.find("origin"),
                transform,
                meshdir / f"{name}_{i}.obj",
            )
        # Go2 thigh/calf visual frames have zero rotation. DAE visuals are
        # discarded by the existing MuJoCo loader; retain correct URDF scaling.
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            origin.set("xyz", _text(transform @ _vector(origin.get("xyz"))))
            visual.find("geometry/mesh").set("scale", f"1 1 {scale}")
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        if parent in maps:
            origin = joint.find("origin")
            origin.set("xyz", _text(maps[parent] @ _vector(origin.get("xyz"))))
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def build_model(path):
    """Use the same physical model construction as the existing Go2 CPU task."""
    from gym.envs.go2.go2_config import Go2Cfg
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.utils.helpers import class_to_dict
    from types import SimpleNamespace

    def namespace(x):
        return (
            SimpleNamespace(**{k: namespace(v) for k, v in x.items()})
            if isinstance(x, dict)
            else x
        )

    cfg = namespace(class_to_dict(Go2Cfg()))
    cfg.asset.file = str(Path(path).resolve())
    cfg.sim_dt = 0.002
    backend = MuJocoCPUBackend()
    model = backend._load_model(cfg, discard_visual=True)
    backend._configure_model(model, cfg, "cpu")
    return model


def actuator_parameters(model):
    """Native joint ordering; effort and speed bounds from unchanged URDF."""
    import mujoco
    from gym.envs.base.urdf_limits import parse_urdf_limits

    limits = parse_urdf_limits(str(SOURCE))
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(1, model.njnt)
    ]
    effort, speed = np.array([limits[n] for n in names]).T
    return names, effort, speed
