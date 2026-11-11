"""Audit trajectory independence, stage coverage and restored policy interfaces."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2kickup_recovery import Go2KickupRecovery
from gym.envs.go2.recovery_state import CONTROLLER, restore, sync
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from scripts.reproduce_transition_hold import BALANCE
from scripts.evaluate_kickup_handoff import Policy
from scripts.train_kickup_recovery import configuration

DEFAULTS = {
    "train": "datasets/go2kickup_recovery_train_v2",
    "validation": "datasets/go2kickup_recovery_validation_v2",
    "heldout": "datasets/go2kickup_recovery_heldout_v2",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(mask, seeds, labels):
    ids = np.flatnonzero(mask)
    counts = Counter(int(seeds[i]) for i in ids)
    total = len(ids)
    outcomes = Counter(labels[i]["outcome"] for i in ids)
    trajectory_outcomes = Counter(
        next(labels[i]["outcome"] for i in ids if int(seeds[i]) == seed)
        for seed in counts
    )
    return dict(
        snapshots=total,
        trajectories=len(counts),
        per_seed=dict(sorted(counts.items())),
        snapshot_outcomes=dict(outcomes),
        trajectory_outcomes=dict(trajectory_outcomes),
        largest_trajectory_share=max(counts.values(), default=0) / max(total, 1),
        inverse_simpson_trajectory_concentration=(
            total**2 / sum(v * v for v in counts.values()) if total else 0
        ),
        note="Concentration describes sampling weights, not independent-state ESS.",
    )


def audit_bank(directory):
    directory = Path(directory)
    with np.load(directory / "train.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    labels = json.loads((directory / "labels.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    seeds = arrays["seed"]
    n = len(seeds)
    if n != len(labels) or any(int(s) != row["seed"] for s, row in zip(seeds, labels)):
        raise ValueError("NPZ/label trajectory alignment mismatch")
    if sha256(directory / "train.npz") != provenance["dataset_sha256"]:
        raise ValueError("Dataset SHA256 differs from recorded provenance")
    for k, value in arrays.items():
        if value.ndim and len(value) != n:
            raise ValueError(f"Unaligned field: {k}")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite field: {k}")
    for k in CONTROLLER:
        if f"controller_{k}" not in arrays:
            raise ValueError(f"Missing controller memory: {k}")
    stages = {str(i): summarize(arrays[f"stage_{i}"], seeds, labels) for i in (3, 4, 5)}
    strides, spans = [], {}
    for seed in sorted(set(seeds.tolist())):
        times = sorted(r["time"] for r in labels if r["seed"] == seed)
        spans[str(seed)] = max(times) - min(times)
        strides.extend(np.diff(times).tolist())
    numeric = {}
    for key in (
        "pitch_rad",
        "pitch_rate",
        "com_support_x",
        "com_speed",
        "base_z",
        "angular_momentum",
        "clearance_2",
        "clearance_3",
    ):
        values = np.array([row[key] for row in labels])
        numeric[key] = dict(
            zip(
                ("min", "p10", "median", "p90", "max"),
                np.quantile(values, [0, 0.1, 0.5, 0.9, 1]).tolist(),
            )
        )
    weight = arrays["model_body_mass"].sum(1) * 9.81
    loads = np.array([[r["force_0"], r["force_1"]] for r in labels])
    supported = loads >= 0.06 * weight[:, None]
    support = Counter(
        "both" if s.all() else "one" if s.any() else "none" for s in supported
    )
    for stage in (3, 4, 5):
        stage_support = Counter(
            "both" if supported[i].all() else "one" if supported[i].any() else "none"
            for i in np.flatnonzero(arrays[f"stage_{stage}"])
        )
        stages[str(stage)]["front_support_snapshot_counts"] = dict(stage_support)
    phase = arrays["controller_leg_phase"]
    if phase.shape != (n, 4) or not ((phase >= 0) & (phase < 2 * np.pi)).all():
        raise ValueError("Invalid CPG phase shape/range")
    overlap = {
        f"{i}_{j}": int((arrays[f"stage_{i}"] & arrays[f"stage_{j}"]).sum())
        for i, j in ((3, 4), (3, 5), (4, 5))
    }
    expected = set(provenance["seeds"])
    return dict(
        directory=str(directory),
        sha256=sha256(directory / "train.npz"),
        **summarize(np.ones(n, dtype=bool), seeds, labels),
        stages=stages,
        generated_seeds=sorted(expected),
        missing_trajectory_seeds=sorted(expected - set(seeds)),
        exact_native_duplicates=n - len({row.tobytes() for row in arrays["native"]}),
        stage_overlap_snapshots=overlap,
        stage_0_2="Procedural resets, not NPZ rows.",
        median_within_trajectory_spacing_s=float(np.median(strides))
        if strides
        else None,
        captured_span_s=spans,
        feature_quantiles=numeric,
        front_support_snapshot_counts=dict(support),
        cpg_phase_histogram_8_bins=[
            np.histogram(phase[:, i], bins=8, range=(0, 2 * np.pi))[0].tolist()
            for i in range(4)
        ],
        cpg_config=provenance["balance"]["env"]["contact_cpg"],
        total_mass_kg_range=[
            float(arrays["model_body_mass"].sum(1).min()),
            float(arrays["model_body_mass"].sum(1).max()),
        ],
        ground_friction_range=[
            float(arrays["model_geom_friction"][:, 0, 0].min()),
            float(arrays["model_geom_friction"][:, 0, 0].max()),
        ],
        controller_families=1,
        source_parameters=provenance["parameters"],
    )


def observation_mapping(env, runner, role):
    obs_names = getattr(runner, f"{role}_cfg")["obs"]
    mapping, parts, offset = [], [], 0
    for name in obs_names:
        raw = getattr(env, name)
        scale = torch.as_tensor(env.scales.get(name, 1.0), dtype=raw.dtype)
        part = raw / scale
        actual = runner.get_obs([name])
        torch.testing.assert_close(actual, part)
        width = part.shape[1]
        mapping.append(
            dict(
                name=name,
                start=offset,
                stop=offset + width,
                width=width,
                divisor=scale.tolist(),
            )
        )
        parts.append(part)
        offset += width
    torch.testing.assert_close(runner.get_obs(obs_names), torch.cat(parts, -1))
    return dict(dimension=offset, slices=mapping)


@torch.inference_mode()
def audit_interface(dataset):
    cfg, train = configuration(Path(dataset) / "train.npz")
    cfg.env.num_envs = 2
    cfg.seed = 7
    train.runner.device = "cpu"
    train.logging.enable_local_saving = False
    task_registry.convert_frequencies_to_params(cfg, train)
    task_registry.set_log_dir_name(train)
    backend = MuJocoCPUBackend()
    env = Go2KickupRecovery(cfg, "cpu", True, backend)
    frozen = None
    try:
        frozen = Policy(BALANCE, "go2balance", 2, True)
        runner = task_registry.make_alg_runner(env, train)
        saved = torch.load(BALANCE, weights_only=True, map_location="cpu")
        normalization = {}
        for role in ("actor", "critic"):
            network = getattr(runner.alg, role)
            state = saved[f"{role}_state_dict"]
            network.load_state_dict(state, strict=True)
            network.eval()
            rms = {k: v for k, v in state.items() if k.startswith("obs_rms.")}
            enabled = network._normalize_obs
            if enabled != bool(rms):
                raise ValueError(f"{role} normalization/config mismatch")
            for k, v in rms.items():
                torch.testing.assert_close(network.state_dict()[k], v, rtol=0, atol=0)
                if not torch.isfinite(v).all():
                    raise ValueError("Nonfinite normalization statistics")
            if rms and (rms["obs_rms.running_var"] < 0).any():
                raise ValueError("Negative normalization variance")
            normalization[role] = dict(
                enabled=enabled,
                buffers={
                    k: dict(
                        shape=list(v.shape),
                        sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest(),
                        min=float(v.min()),
                        max=float(v.max()),
                    )
                    for k, v in rms.items()
                },
            )
        runner.switch_to_eval()
        frozen.runner.alg.critic.load_state_dict(
            saved["critic_state_dict"], strict=True
        )
        frozen.runner.switch_to_eval()
        mapping = {
            role: observation_mapping(env, runner, role) for role in ("actor", "critic")
        }
        if [mapping[r]["dimension"] for r in ("actor", "critic")] != [59, 63]:
            raise ValueError("Unexpected observation dimensions")
        with np.load(Path(dataset) / "train.npz", allow_pickle=False) as source:
            arrays = {k: source[k] for k in source.files}
        cpg_errors, observation_errors, action_errors = [], [], []
        max_normalized_critic_input = 0.0
        max_critic_value = 0.0
        for i in range(len(arrays["seed"])):
            sample = {
                k: v[i]
                for k, v in arrays.items()
                if k == "native" or k.startswith(("controller_", "model_"))
            }
            restore(env, 0, sample)
            sync(env)
            restore(frozen.env, 0, sample)
            sync(frozen.env)
            phase = torch.as_tensor(sample["controller_leg_phase"])
            expected = torch.stack((phase.sin(), phase.cos()), -1).flatten()
            torch.testing.assert_close(env.cpg_phase_obs[0], expected)
            torch.testing.assert_close(
                env._cpg_load[0], torch.as_tensor(sample["controller__cpg_load"])
            )
            torch.testing.assert_close(
                env.cpg_contact_obs[0], torch.tensor([1.0, 1.0, 0.0, 0.0])
            )
            for role in ("actor", "critic"):
                obs = runner.get_obs(getattr(runner, f"{role}_cfg")["obs"])
                original_obs = frozen.runner.get_obs(
                    getattr(frozen.runner, f"{role}_cfg")["obs"]
                )
                torch.testing.assert_close(obs[0], original_obs[0], rtol=0, atol=1e-6)
                observation_errors.append(float((obs[0] - original_obs[0]).abs().max()))
                if not torch.isfinite(obs).all():
                    raise ValueError("Nonfinite restored observations")
            actor_obs = runner.get_obs(runner.actor_cfg["obs"])
            actions = runner.alg.actor.act_inference(actor_obs)
            if actions.shape != (2, 12) or not torch.isfinite(actions).all():
                raise ValueError("Invalid frozen-policy actions")
            original_action = frozen.runner.get_inference_actions()[0]
            torch.testing.assert_close(actions[0], original_action, rtol=0, atol=1e-6)
            action_errors.append(float((actions[0] - original_action).abs().max()))
            # Exercise the critic's checkpoint normalization in eval mode.
            critic_obs = runner.get_obs(runner.critic_cfg["obs"])
            value = runner.alg.critic.evaluate(critic_obs)
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite checkpoint-normalized critic value")
            normalized = runner.alg.critic.obs_rms(critic_obs)
            max_normalized_critic_input = max(
                max_normalized_critic_input, float(normalized[0].abs().max())
            )
            max_critic_value = max(max_critic_value, float(value[0].abs()))
            cpg_errors.append(float((env.cpg_phase_obs[0] - expected).abs().max()))
        for role in ("actor", "critic"):
            for k, v in saved[f"{role}_state_dict"].items():
                if k.startswith("obs_rms."):
                    torch.testing.assert_close(
                        getattr(runner.alg, role).state_dict()[k], v, rtol=0, atol=0
                    )
        procedural = {}
        env.cfg.handstand_curriculum.replay_fraction = 0
        torch.manual_seed(7)
        for stage in range(3):
            env.curriculum_level = stage
            rows = []
            for _ in range(16):
                env._reset_idx(torch.ones(2, dtype=torch.bool))
                if not (env._reset_level == stage).all():
                    raise ValueError("Procedural stage reset mismatch")
                torch.testing.assert_close(
                    env.leg_phase, env._cpg_offsets.expand(2, -1)
                )
                torch.testing.assert_close(
                    env._cpg_load, torch.zeros_like(env._cpg_load)
                )
                b = env._backend
                feet = b._canonical_to_native_body_np[env.feet_indices[:2].numpy()]
                for i, d in enumerate(b._datas):
                    mujoco.mj_subtreeVel(b._model_for_env(i), d)
                    rows.append(
                        dict(
                            pitch=float(
                                torch.atan2(
                                    env.projected_gravity[i, 0],
                                    -env.projected_gravity[i, 2],
                                )
                            ),
                            com_speed=float(np.linalg.norm(d.subtree_linvel[1])),
                            com_support_offset=float(
                                np.linalg.norm(
                                    d.subtree_com[1, :2] - d.xpos[feet, :2].mean(0)
                                )
                            ),
                            max_horizontal_base_speed_component=float(
                                abs(d.qvel[:2]).max()
                            ),
                            base_height=float(d.qpos[2]),
                        )
                    )
            procedural[str(stage)] = dict(
                resets=len(rows),
                observed_ranges={
                    k: [min(r[k] for r in rows), max(r[k] for r in rows)]
                    for k in rows[0]
                },
            )
        env.curriculum_level = 0
        env._reset_idx(torch.ones(2, dtype=torch.bool))
        return dict(
            checkpoint_sha256=sha256(BALANCE),
            procedural_reset_coverage=procedural,
            observations=mapping,
            normalization=normalization,
            actions=12,
            all_replay_states_checked=len(cpg_errors),
            maximum_original_observation_difference=max(observation_errors),
            maximum_original_action_difference=max(action_errors),
            maximum_abs_normalized_critic_input=max_normalized_critic_input,
            maximum_abs_critic_value=max_critic_value,
            maximum_phase_observation_error=max(cpg_errors),
            normalization_unchanged_after_eval=True,
            cpg_reset_offsets_rad=env._cpg_offsets.tolist(),
            cpg_config=class_to_dict(cfg.contact_cpg),
            normalization_semantics=(
                "Environment divisors first; actor has no RMS; critic uses saved "
                "RMS in eval, updated during training."
            ),
        )
    finally:
        if frozen is not None:
            frozen.env._backend.close()
        backend.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    for role, default in DEFAULTS.items():
        parser.add_argument(f"--{role}", default=default, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    banks = {role: audit_bank(getattr(a, role)) for role in DEFAULTS}
    for left, right in (
        ("train", "validation"),
        ("train", "heldout"),
        ("validation", "heldout"),
    ):
        if set(banks[left]["generated_seeds"]) & set(banks[right]["generated_seeds"]):
            raise ValueError("Cross-cohort trajectory leakage")
    native_hashes = {}
    for role in DEFAULTS:
        with np.load(getattr(a, role) / "train.npz", allow_pickle=False) as source:
            native_hashes[role] = {
                hashlib.sha256(row.tobytes()).hexdigest() for row in source["native"]
            }
    exact_cross_cohort_overlap = {
        f"{left}_{right}": len(native_hashes[left] & native_hashes[right])
        for left, right in (
            ("train", "validation"),
            ("train", "heldout"),
            ("validation", "heldout"),
        )
    }
    if any(exact_cross_cohort_overlap.values()):
        raise ValueError("Repeated exact integration states across cohorts")
    interface = audit_interface(a.train)
    other_interfaces = {
        role: audit_interface(getattr(a, role)) for role in ("validation", "heldout")
    }
    for bank in banks.values():
        if bank["cpg_config"] != interface["cpg_config"]:
            raise ValueError("Dataset/controller CPG configurations differ")
    report = dict(
        banks=banks,
        exact_cross_cohort_native_overlap=exact_cross_cohort_overlap,
        interface=interface,
        other_cohort_interface_checks={
            role: dict(
                restored_states=value["all_replay_states_checked"],
                maximum_phase_observation_error=value[
                    "maximum_phase_observation_error"
                ],
                normalization_unchanged_after_eval=value[
                    "normalization_unchanged_after_eval"
                ],
            )
            for role, value in other_interfaces.items()
        },
        warnings=[
            "Adjacent snapshots are correlated, not independent recovery trials.",
            "One kick/transition controller family; no successful training trajectory.",
            "Stage 3 and 4 overlap and do not form nested difficulty sets.",
            "Stages 0-2 are procedural; coverage is not represented in the NPZ.",
            "Historical validation/heldout cohorts are previously inspected seeds.",
        ],
    )
    with a.output.open("x") as f:
        json.dump(report, f, indent=2)
    print(
        json.dumps(
            {
                r: {k: banks[r][k] for k in ("snapshots", "trajectories", "stages")}
                for r in DEFAULTS
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
