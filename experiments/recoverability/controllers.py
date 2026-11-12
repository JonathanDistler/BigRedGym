"""Saved-source controllers and explicit adapters to a shared physical world."""

import argparse
import hashlib
import importlib
from pathlib import Path
import numpy as np
import torch
import gym.envs  # noqa: F401
from gym.utils.original_cfg import (
    checkpoint_configs,
    original_cfg_source_dir,
    _saved_env_imports,
)
from gym.utils.helpers import class_to_dict, set_seed
from gym.utils.robot_metrics import configure_contact_cpg
from gym.utils.task_registry import task_registry, select_backend
from gym.envs.go2.frontyield_cpu import install_batched_cpu_sync

CANDIDATES = [
    ("balance_early", "logs/go2balance/Oct28_22-26-30_/model_1500.pt", "go2balance"),
    (
        "balance_cpg_front",
        "logs/go2balance/Oct31_23-29-27_/model_1500.pt",
        "go2balance",
    ),
    (
        "balance_longhold",
        "logs/go2balance_longhold/Nov01_22-09-05_/model_1500.pt",
        "go2balance",
    ),
    (
        "reachability_early",
        "logs/go2backwards_reachability/Oct30_22-11-01_/model_1500.pt",
        "go2reachability",
    ),
    (
        "reachability_late",
        "logs/go2backwards_reachability/Nov02_21-05-21_/model_1500.pt",
        "go2reachability",
    ),
    (
        "reachability_fast",
        "logs/go2reachability_fast_probe/Nov03_01-09-45_/model_100.pt",
        "go2reachability",
    ),
]


class SavedController:
    def __init__(self, checkpoint, task):
        self.path = Path(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        actor = saved["actor_state_dict"]
        dim = actor["NN.0.weight"].shape[1]
        cfg, train = checkpoint_configs(checkpoint, task)
        self.saved_config = class_to_dict(cfg)
        if dim == 59:
            # Restore the archived command-line override.
            # Evidence: named archive and prior saved-source audit.
            configure_contact_cpg(
                argparse.Namespace(
                    cpg=True, contact_modes=["stance", "stance", "swing", "swing"]
                ),
                cfg,
                train,
            )
        elif dim != 47:
            raise ValueError(f"Unsupported saved observation interface: {dim}")
        cfg.env.num_envs = 1
        cfg.env.episode_length_s = 12
        cfg.seed = 7
        cfg.push_robots.toggle = False
        if hasattr(cfg, "balance_perturbations"):
            cfg.balance_perturbations.enabled = False
        train.runner.device = "cpu"
        train.runner.resume = False
        train.logging.enable_local_saving = False
        task_registry.convert_frequencies_to_params(cfg, train)
        task_registry.set_log_dir_name(train)
        set_seed(7)
        with _saved_env_imports(original_cfg_source_dir(self.path.parent)):
            module = importlib.import_module("gym.envs.go2." + task)
            cls = getattr(
                module,
                {"go2balance": "Go2Balance", "go2reachability": "Go2Reachability"}[
                    task
                ],
            )
        self.env = cls(
            cfg=cfg,
            device="cpu",
            headless=True,
            backend=select_backend(cfg, "cpu", "mujoco"),
        )
        install_batched_cpu_sync(self.env._backend)
        self.runner = task_registry.make_alg_runner(self.env, train)
        self.runner.alg.actor.load_state_dict(actor, strict=True)
        self.runner.switch_to_eval()
        self.actor = self.runner.alg.actor.eval().requires_grad_(False)
        self.dim = dim
        self.obs_names = list(self.runner.actor_cfg["obs"])
        self.level = (
            saved.get("environment_training_state", {})
            .get("handstand_curriculum", {})
            .get("level", 0)
        )
        self.meta = dict(
            checkpoint=str(self.path),
            sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            saved_environment=self.saved_config,
            effective_environment=class_to_dict(cfg),
            actor_fields=self.obs_names,
            actor_dim=dim,
            actor_config=self.runner.actor_cfg,
            training_state=saved.get("environment_training_state"),
            cpg_override=(
                "reconstructed stance/stance/swing/swing; "
                "runtime flag absent from source"
            )
            if dim == 59
            else "none",
        )
        self.digest = self.network_hash()

    def network_hash(self):
        h = hashlib.sha256()
        for k, v in self.actor.state_dict().items():
            h.update(k.encode())
            h.update(v.cpu().numpy().tobytes())
        return h.hexdigest()

    def close(self):
        assert self.network_hash() == self.digest, "Frozen controller mutated"
        self.env._backend.close()


class BalanceAdapter:
    """Separate encoder, unbounded residual action, native 100 Hz absolute target.

    No state projection, policy averaging, or extra actor normalization.
    Controller memory is explicit and is not part of the robot's physical state.
    """

    def __init__(self, controller, world):
        self.controller = controller
        self.world = world
        e = controller.env
        self.names = [e.dof_names[i] for i in e.actuated_dof_indices.tolist()]
        self.q = np.array(
            [world.m.jnt_qposadr[world.m.joint(n).id] for n in self.names]
        )
        self.v = np.array([world.m.jnt_dofadr[world.m.joint(n).id] for n in self.names])
        self.reference = e._standing_posture()[0].detach().cpu().clone()
        self.default = e.default_dof_pos[0].detach().cpu().clone()
        self.command = e.commands[0].detach().cpu().clone()
        self.previous = torch.tensor(world.target, dtype=torch.float32) - self.reference
        if controller.dim == 59:
            self.phase = e._cpg_offsets.detach().clone()
            self.load = torch.zeros(4)
        self.clock = 0.0

    def observation(self):
        c = self.controller
        w = self.world
        r = w.d.xmat[w.base].reshape(3, 3)
        values = dict(
            base_ang_vel=torch.tensor(w.d.qvel[3:6], dtype=torch.float32),
            projected_gravity=torch.tensor(
                r.T @ np.array([0.0, 0.0, -1.0]), dtype=torch.float32
            ),
            commands=self.command,
            dof_pos_obs=torch.tensor(w.d.qpos[self.q], dtype=torch.float32)
            - self.default,
            dof_vel=torch.tensor(w.d.qvel[self.v], dtype=torch.float32),
            dof_pos_target=self.previous,
            handstand_stage_obs=torch.tensor([1.0]),
        )
        if c.dim == 59:
            values["cpg_phase_obs"] = torch.stack(
                (self.phase.sin(), self.phase.cos()), dim=-1
            ).flatten()
            values["cpg_contact_obs"] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        return torch.cat(
            [
                (values[k] / torch.as_tensor(c.env.scales.get(k, 1.0))).flatten()
                for k in c.obs_names
            ]
        ).unsqueeze(0)

    @torch.inference_mode()
    def target(self, stochastic=False, generator=None):
        raw = self.controller.actor.act_inference(self.observation())[0]
        if stochastic:
            raw = raw + self.controller.actor.std * torch.randn(12, generator=generator)
        scale = torch.as_tensor(self.controller.env.scales["dof_pos_target"])
        self.previous = raw * scale
        return (self.reference + self.previous).numpy().astype(float)

    def advance_memory(self, loads, dt):
        self.clock += dt
        if self.controller.dim != 59:
            return
        c = self.controller.env.cfg.contact_cpg
        alpha = 1.0 if c.force_filter_s == 0 else -np.expm1(-dt / c.force_filter_s)
        self.load.lerp_(
            torch.tensor(loads / self.world.weight, dtype=torch.float32), float(alpha)
        )
        rate = 2 * torch.pi * c.frequency_hz - c.sigma * self.load * (
            self.phase.cos() + c.zeta
        )
        self.phase.add_(dt * rate).remainder_(2 * torch.pi)
