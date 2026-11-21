"""Opt-in, simulation-only replay of external Go2 position-command trajectories.

NPZ schema 1 (no pickle): schema_version, mujoco_version, actions (N,12),
initial_state (mjSTATE_INTEGRATION), reference_states (N+1,state_size), nominal
and initial_target (12,), pitch_axis (3,). Joint order is MuJoCo's URDF order.
Commands include the recorded balance controller outputs; no policy is inferred.
"""

import hashlib
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from gym import GYM_ROOT_DIR
from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.base.urdf_limits import parse_urdf_limits
from gym.envs.go2.go2_config import Go2Cfg
from gym.utils.helpers import class_to_dict

CONTROL_DT = 0.02
PHYSICS_DT = 0.002
KP, KD, ACTION_SCALE = 60.0, 3.0, 2.0
TARGET_HEIGHT = 0.5600034657639977
TARGET_ROTATION = np.array(
    [
        -0.038478081417006725,
        0.023214148184316278,
        0.9989897599948376,
        0.004223156314122425,
        0.9997249652079541,
        -0.0230685693250003,
        -0.9992505202406519,
        0.003331255624004619,
        -0.038565535582367395,
    ]
).reshape(3, 3)
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def load_trajectory(path: Path, sha256: str, model) -> dict:
    """Check artifact identity, numeric schema, and MuJoCo compatibility."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256.lower():
        raise ValueError("Trajectory SHA256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if data.get("schema_version") != 1:
        raise ValueError("Expected trajectory schema_version 1")
    if str(data.get("mujoco_version")) != mujoco.__version__:
        raise ValueError("Trajectory requires a different MuJoCo version")
    size = mujoco.mj_stateSize(model, STATE_SPEC)
    actions = data.get("actions", np.empty(0))
    if actions.ndim != 2 or actions.shape[1] != 12 or len(actions) == 0:
        raise ValueError("Expected nonempty actions (N,12)")
    shapes = {
        "initial_state": (size,),
        "reference_states": (len(actions) + 1, size),
        "nominal": (12,),
        "initial_target": (12,),
        "pitch_axis": (3,),
    }
    for key, shape in shapes.items():
        if key not in data or data[key].shape != shape:
            raise ValueError(f"Invalid trajectory shape: {key}")
    for key in ["actions", *shapes]:
        if data[key].dtype.kind != "f" or not np.isfinite(data[key]).all():
            raise ValueError(f"Expected finite floating-point array: {key}")
    if (
        np.max(abs(actions)) > 1
        or not np.allclose(np.linalg.norm(data["pitch_axis"]), 1, atol=1e-12)
        or abs(data["pitch_axis"][2]) > 1e-12
    ):
        raise ValueError("Invalid normalized commands or horizontal pitch axis")
    if not np.array_equal(data["initial_state"], data["reference_states"][0]):
        raise ValueError("Initial state differs from reference start")
    return data


def build_model():
    """Build the unmodified nominal Go2 through the existing CPU backend."""

    def namespace(value):
        return (
            SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
            if isinstance(value, dict)
            else value
        )

    cfg = namespace(class_to_dict(Go2Cfg()))
    cfg.sim_dt = PHYSICS_DT
    backend = MuJocoCPUBackend()
    model = backend._load_model(cfg, discard_visual=True)
    backend._configure_model(model, cfg, "cpu")
    return model


def clearance(model, data, geom):
    """Lowest collision-primitive point above the z=0 ground plane, in metres."""
    kind = model.geom_type[geom]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    size = model.geom_size[geom]
    z = data.geom_xpos[geom, 2]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return z - size[0]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return z - np.abs(rotation[2]) @ size
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return (
            z
            - abs(rotation[2, 2]) * size[1]
            - size[0] * np.linalg.norm(rotation[2, :2])
        )
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        return z + np.min(model.mesh_vert[start : start + count] @ rotation[2])
    raise ValueError(f"Unsupported Go2 collision geometry: {kind}")


def strict_balance(
    *,
    safe,
    rear_clearance,
    rear_loads,
    front_support,
    orientation,
    base_height,
    com_height,
    com_speed,
    angular_speed,
    support_error,
    slip,
    calf_clearance,
    head_clearance,
    force_ratio,
):
    """Original strict handstand gate, evaluated each 2 ms; SI units/radians."""
    return bool(
        safe
        and rear_clearance > 0.10
        and (rear_loads < 0.02).all()
        and front_support
        and orientation < np.deg2rad(10)
        and abs(base_height - 0.57) < 0.06
        and abs(com_height - TARGET_HEIGHT) < 0.06
        and com_speed < 0.25
        and angular_speed < 0.5
        and support_error < 0.04
        and slip < 0.05
        and calf_clearance > 0
        and head_clearance > 0
        and force_ratio < 4.0
    )


class Replay:
    """Continuous native MuJoCo dynamics, bounded PD control, and strict scoring."""

    def __init__(self, model, trajectory):
        self.m, self.d, self.trajectory = model, mujoco.MjData(model), trajectory
        self.names = [model.joint(j).name for j in range(1, model.njnt)]
        expected = [
            f"{leg}_{part}_joint"
            for leg in ("FL", "FR", "RL", "RR")
            for part in ("hip", "thigh", "calf")
        ]
        if self.names != expected or model.nq != 19 or model.nv != 18:
            raise ValueError("Expected nominal Go2 joint ordering")
        joints = [model.joint(n).id for n in self.names]
        self.qids, self.vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.bounds = model.jnt_range[joints]
        limits = parse_urdf_limits(Go2Cfg.asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR))
        self.effort, self.speed = np.array([limits[n] for n in self.names]).T
        self.base = model.body("base").id
        self.feet = [model.body(f"{leg}_foot").id for leg in ("FL", "FR", "RL", "RR")]
        self.foot_geoms = [np.flatnonzero(model.geom_bodyid == b)[0] for b in self.feet]
        self.calf = self.geoms(("FL_calf", "FR_calf"))
        self.head = self.geoms(("Head",))
        self.weight = model.body_mass.sum() * 9.81
        mujoco.mj_setState(model, self.d, trajectory["initial_state"], STATE_SPEC)
        mujoco.mj_forward(model, self.d)
        if (
            model.opt.timestep != PHYSICS_DT
            or self.d.time != 0
            or self.d.qpos[2] < 0.3
            or np.linalg.norm(self.d.qpos[3:7] - [1, 0, 0, 0]) > 1e-3
            or np.max(abs(self.d.geom_xpos[self.foot_geoms, 2] - 0.022)) > 0.003
            or np.any(self.d.xfrc_applied)
            or np.any(self.d.qfrc_applied)
        ):
            raise ValueError(
                "Trajectory must start standing at t=0 without applied forces"
            )
        self.anchor = self.d.geom_xpos[self.foot_geoms[:2], :2].copy()
        self.target = trajectory["initial_target"].copy()
        self.safe = True
        self.hold = self.best_hold = self.peak_pitch = 0.0
        self.illegal_contacts = 0
        self.captured = self.takeoff = False
        self.loaded = np.zeros(2, dtype=bool)
        self.release = np.full(2, np.nan)
        self.previous_rear = None
        self.window = deque(maxlen=200)  # Original 0.4 s takeoff certificate window.

    def geoms(self, prefixes):
        """Select original collidable body geometries by name prefix."""
        return [
            g
            for g in range(self.m.ngeom)
            if self.m.body(int(self.m.geom_bodyid[g])).name.startswith(prefixes)
            and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])
        ]

    def measure(self, previous_velocity):
        """Evaluate physical contacts, safety, takeoff and balance after a substep."""
        m, d = self.m, self.d
        loads, front_force = np.zeros(4), 0.0
        wrench = np.empty(6)
        collisions = 0
        for i, contact in enumerate(d.contact):
            b1, b2 = m.geom_bodyid[[contact.geom1, contact.geom2]]
            if b1 == 0 and b2 == 0:
                continue
            if bool(b1) != bool(b2):
                body = int(b2 if b1 == 0 else b1)
                mujoco.mj_contactForce(m, d, i, wrench)
                if body in self.feet[:2]:
                    sign = 1.0 if b1 == 0 else -1.0
                    front_force += (sign * contact.frame.reshape(3, 3).T @ wrench[:3])[
                        2
                    ]
            if contact.dist > 0:
                continue
            if b1 and b2:
                collisions += 1
            elif body in self.feet:
                # Retain the original contact-frame load convention exactly.
                loads[self.feet.index(body)] += abs(
                    contact.frame.reshape(3, 3)[:, 2] @ wrench[:3]
                )
            else:
                collisions += 1
        self.illegal_contacts += collisions
        self.safe &= bool(
            not collisions
            and (abs(d.qvel[self.vids]) / self.speed <= 1.001).all()
            and (d.qpos[self.qids] >= self.bounds[:, 0] - 0.001).all()
            and (d.qpos[self.qids] <= self.bounds[:, 1] + 0.001).all()
        )
        mujoco.mj_subtreeVel(m, d)
        rotation = d.xmat[self.base].reshape(3, 3)
        axis = self.trajectory["pitch_axis"]
        pitch = np.arctan2(-rotation[2, 0], rotation[:, 0] @ [axis[1], -axis[0], 0])
        if self.safe:
            self.peak_pitch = max(self.peak_pitch, pitch)
        feet = d.geom_xpos[self.foot_geoms]
        front = bool((loads[:2] > 0.05 * self.weight).all())
        rear_clear = float(feet[2:, 2].min() - 0.022)
        self.captured |= bool(self.safe and front and rear_clear > 0.10)
        self.certify_takeoff(loads, feet, front, pitch, previous_velocity)
        com = d.subtree_com[self.base]
        segment = feet[1, :2] - feet[0, :2]
        u = np.clip(
            (com[:2] - feet[0, :2]) @ segment / max(segment @ segment, 1e-10), 0, 1
        )
        jac = np.zeros((3, m.nv))
        slip = 0.0
        for g in self.foot_geoms[:2]:
            mujoco.mj_jacGeom(m, d, jac, None, int(g))
            slip = max(slip, np.linalg.norm((jac @ d.qvel)[:2]))
        stable = strict_balance(
            safe=self.safe,
            rear_clearance=rear_clear,
            rear_loads=loads[2:] / self.weight,
            front_support=front,
            orientation=np.arccos(
                np.clip((np.trace(TARGET_ROTATION.T @ rotation) - 1) / 2, -1, 1)
            ),
            base_height=d.qpos[2],
            com_height=com[2],
            com_speed=np.linalg.norm(d.subtree_linvel[self.base]),
            angular_speed=np.linalg.norm(d.qvel[3:6]),
            support_error=np.linalg.norm(com[:2] - feet[0, :2] - u * segment),
            slip=slip,
            calf_clearance=min(clearance(m, d, g) for g in self.calf),
            head_clearance=min(clearance(m, d, g) for g in self.head),
            force_ratio=front_force / self.weight,
        )
        self.hold = self.hold + PHYSICS_DT if stable else 0.0
        self.best_hold = max(self.best_hold, self.hold)

    def certify_takeoff(self, loads, feet, front, pitch, previous_velocity):
        """Require a recent powered rear-leg push and synchronized rear release."""
        loads = loads[2:] / self.weight
        heights = feet[2:, 2]
        vertical = (
            (heights - self.previous_rear) / PHYSICS_DT
            if self.previous_rear is not None
            else np.zeros(2)
        )
        self.previous_rear = heights.copy()
        self.loaded |= loads > 0.08
        knee_velocity = self.d.qvel[self.vids[[8, 11]]]
        power = np.maximum(self.d.qfrc_applied[self.vids] * previous_velocity, 0)
        work = np.array([power[6:9].sum(), power[9:12].sum()])
        self.window.append(
            (
                np.clip(loads - 0.15, 0, 1.5)
                * np.clip(knee_velocity / 0.5, 0, 1)
                * PHYSICS_DT,
                np.maximum(knee_velocity, 0) * PHYSICS_DT,
                work * (loads > 0.08) * PHYSICS_DT,
            )
        )
        impulse, extension, work = np.sum(self.window, axis=0)
        for j in range(2):
            if loads[j] > 0.04:
                self.release[j] = np.nan
            elif (
                self.loaded[j]
                and loads[j] < 0.02
                and heights[j] - 0.022 > 0.008
                and vertical[j] > 0.02
                and np.isnan(self.release[j])
            ):
                self.release[j] = self.d.time
        recent = (
            np.isfinite(self.release).all() and self.d.time - min(self.release) < 0.4
        )
        pushed = (
            self.loaded.all()
            and (impulse >= 0.002).all()
            and (extension >= 0.03).all()
            and (work >= 0.05).all()
        )
        self.takeoff |= bool(
            self.safe
            and front
            and (loads < 0.02).all()
            and (heights - 0.022 > 0.015).all()
            and pitch > 0.12
            and np.linalg.norm(feet[:2, :2] - self.anchor, axis=1).max() < 0.10
            and pushed
            and recent
            and abs(self.release[0] - self.release[1]) <= 0.04
        )

    def run(self) -> dict:
        """Integrate every command continuously and return measured results."""
        error = 0.0
        for index, action in enumerate(self.trajectory["actions"]):
            desired = np.clip(
                self.trajectory["nominal"] + ACTION_SCALE * action, *self.bounds.T
            )
            self.target += np.clip(
                desired - self.target,
                -0.7 * self.speed * CONTROL_DT,
                0.7 * self.speed * CONTROL_DT,
            )
            for _ in range(round(CONTROL_DT / PHYSICS_DT)):
                velocity = self.d.qvel[self.vids].copy()
                torque = np.clip(
                    KP * (self.target - self.d.qpos[self.qids]) - KD * velocity,
                    -self.effort,
                    self.effort,
                )
                torque[(abs(velocity) >= self.speed) & (torque * velocity > 0)] = 0
                self.d.qfrc_applied[:] = 0
                self.d.qfrc_applied[self.vids] = torque
                mujoco.mj_step(self.m, self.d)
                mujoco.mj_forward(self.m, self.d)
                self.measure(velocity)
            state = np.empty(mujoco.mj_stateSize(self.m, STATE_SPEC))
            mujoco.mj_getState(self.m, self.d, state, STATE_SPEC)
            if not np.isfinite(state).all():
                raise ValueError("Non-finite simulation state")
            error = max(
                error,
                float(
                    np.max(abs(state - self.trajectory["reference_states"][index + 1]))
                ),
            )
        return dict(
            strict_hold_seconds=self.best_hold,
            five_second_success=self.best_hold >= 5,
            certified_takeoff=self.takeoff,
            front_capture=self.captured,
            legal_contacts=self.illegal_contacts == 0,
            prefix_safe=self.safe,
            max_pitch_degrees=float(np.rad2deg(self.peak_pitch)),
            max_state_error=error,
            replay_matches=error <= 1e-10,
        )
