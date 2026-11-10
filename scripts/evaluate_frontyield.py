"""Native-reference parity and matched-seed evaluation of bounded PPO policies."""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
import gym.envs  # noqa:F401
from gym.envs.go2.go2frontyield_config import Go2FrontYieldCfg, Go2FrontYieldRunnerCfg
from gym.utils.helpers import set_seed, class_to_dict
from gym.utils.task_registry import task_registry
from gym.utils.original_cfg import checkpoint_configs
from scripts.search_kickup_reference import Experiment


def matched_reset(env, ex, seeds):
    b = env._backend
    models = []
    datas = []
    anchors = []
    gains = []
    damping = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        m = copy.copy(ex.model)
        scale = rng.uniform(0.95, 1.05)
        m.body_mass[:] *= scale
        m.body_inertia[:] *= scale
        m.geom_friction[:, 0] = rng.uniform(0.6, 1.0)
        mujoco.mj_setConst(m, mujoco.MjData(m))
        d = mujoco.MjData(m)
        d.qpos[:] = ex.initial
        d.qpos[7 + ex.order] += rng.normal(0, 0.008, 12)
        angles = rng.uniform(-0.012, 0.012, 3)
        angles[2] = rng.uniform(-0.15, 0.15)
        mujoco.mju_euler2Quat(d.qpos[3:7], angles, "xyz")
        mujoco.mj_forward(m, d)
        d.qpos[2] += (
            float(
                __import__(
                    "gym.envs.go2.go2_config", fromlist=["FOOT_RADIUS"]
                ).FOOT_RADIUS
            )
            + 0.001
            - d.xpos[ex.feet, 2].min()
        )
        mujoco.mj_forward(m, d)
        gains.append(60 * rng.uniform(0.95, 1.05, 12))
        damping.append(3 * rng.uniform(0.95, 1.05, 12))
        models.append(m)
        datas.append(d)
        anchors.append(d.xpos[ex.feet[:2], :2].copy())
    b._models = models
    b._mjm = models[0]
    b._datas = datas
    for i, m in enumerate(models):
        b._contact_friction_t[i] = float(m.geom_friction[0, 0])
        b._link_mass_t[i] = torch.tensor(
            m.body_mass[b._canonical_to_native_body_np], dtype=torch.float32
        )
    b._sync_state_from_mujoco()
    env.p_gains[:] = torch.tensor(np.array(gains), dtype=torch.float32)
    env.d_gains[:] = torch.tensor(np.array(damping), dtype=torch.float32)
    env._refresh_base_observations(torch.ones(env.num_envs, dtype=torch.bool))
    env._front_anchor[:] = torch.tensor(np.array(anchors), dtype=torch.float32)
    env.dof_pos_obs[:] = env.dof_pos - env.default_dof_pos


def evaluate(checkpoint=None, output="output/frontyield_parity", level=None):
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    ex = Experiment()
    cfg, train = Go2FrontYieldCfg(), Go2FrontYieldRunnerCfg()
    config_source = "current opt-in task"
    if checkpoint:
        effective = Path(checkpoint).parent / "effective_config.json"
        if (
            effective.is_file()
            and json.loads(effective.read_text())["task"] == "go2frontyield"
        ):
            cfg, train = checkpoint_configs(checkpoint, "go2frontyield")
            config_source = str(effective)
        else:
            _, oldtrain = checkpoint_configs(checkpoint, "go2kickup")
            for key in (
                "obs",
                "actions",
                "normalize_obs",
                "hidden_dims",
                "activation",
                "smooth_exploration",
            ):
                if class_to_dict(oldtrain.actor).get(key) != class_to_dict(
                    train.actor
                ).get(key):
                    raise ValueError(f"Actor interface mismatch: {key}")
    cfg.env.num_envs = 16
    cfg.seed = 7
    train.runner.device = "cpu"
    train.runner.resume = False
    train.logging.enable_local_saving = False
    task_registry.convert_frequencies_to_params(cfg, train)
    task_registry.set_log_dir_name(train)
    set_seed(7)
    env = task_registry.make_env(
        "go2frontyield", cfg, device="cpu", headless=True, backend="mujoco"
    )
    runner = task_registry.make_alg_runner(env, train)
    if checkpoint:
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        runner.alg.actor.load_state_dict(saved["actor_state_dict"], strict=True)
        state = saved.get("environment_training_state", {}).get(
            "handstand_curriculum", {}
        )
        level = state.get("level", 0) if level is None else level
    level = 0 if level is None else level
    runner.switch_to_eval()
    env.curriculum_level = level
    env._episode_active.zero_()
    env._reset_idx(torch.ones(16, dtype=torch.bool))
    env._reset_level[:] = level
    seeds = list(range(3000, 3016))
    matched_reset(env, ex, seeds)
    original = env._post_physics_step
    rows = [[] for _ in seeds]
    native_contact_seen = [False] * 16
    from gym.envs.go2.go2_config import FOOT_RADIUS

    def capture():
        original()
        t = float(env._kick_clock[0] + env._physics_offset) - cfg.sim_dt
        for i in range(16):
            d = env._backend._datas[i]
            m = env._backend._model_for_env(i)
            if t > 0.15:
                for c in d.contact:
                    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                    if (
                        c.dist < 0
                        and (b1 == 0 or b2 == 0)
                        and (b2 if b1 == 0 else b1) not in ex.feet
                    ):
                        native_contact_seen[i] = True
            feet = d.xpos[ex.feet]
            clear = feet[:, 2] - FOOT_RADIUS
            row = dict(
                time=t,
                clearance_min=float(clear[2:].min()),
                front_drift=float(env._front_displacement()[i].max()),
                pitch_rad=float(
                    np.arctan2(
                        -d.xmat[ex.base].reshape(3, 3)[2, 0],
                        d.xmat[ex.base].reshape(3, 3)[0, 0],
                    )
                ),
                pitch_rate=float(d.cvel[ex.base, 1]),
                body_contact=native_contact_seen[i],
                terminated=bool(env.terminated[i]),
            )
            for j in range(12):
                row[f"actual_{j}"] = float(d.qpos[7 + ex.order[j]])
                row[f"torque_{j}"] = float(env.torques[i, j])
                row[f"residual_{j}"] = float(env._applied_residual[i, j])
            for j in range(4):
                row[f"force_{j}"] = float(
                    env.contact_forces[i, env.feet_indices[j], 2].clamp(min=0)
                )
                row[f"clearance_{j}"] = float(clear[j])
            rows[i].append(row)

    env._post_physics_step = capture
    with torch.inference_mode():
        for _ in range(160):
            actions = (
                runner.get_inference_actions() if checkpoint else torch.zeros(16, 12)
            )
            runner.set_actions(runner.actor_cfg["actions"], actions, False)
            env.step()
    results = []
    parity = []
    for i, seed in enumerate(seeds):
        r = rows[i]
        weight = float(env._body_weight()[i])
        best = hold = 0.0
        loaded = False
        for row in r:
            loaded |= min(row["force_2"], row["force_3"]) >= 0.03 * weight
            ok = (
                loaded
                and row["clearance_min"] >= 0.03
                and max(row["force_2"], row["force_3"]) < 0.015 * weight
                and min(row["force_0"], row["force_1"]) >= 0.06 * weight
                and row["front_drift"] <= 0.04
                and not row["body_contact"]
            )
            hold = hold + cfg.sim_dt if ok else 0
            best = max(best, hold)
        push = [x for x in r if 0.776 <= x["time"] <= 0.979]
        post = [x for x in r if 1.05 <= x["time"] <= 1.35]
        handoff_hold = best_handoff = 0.0
        for row in r:
            candidate = (
                row["time"] > 1.05
                and np.deg2rad(75) <= row["pitch_rad"] <= np.deg2rad(105)
                and abs(row["pitch_rate"]) <= 1.5
                and row["clearance_min"] >= 0.03
                and min(row["force_0"], row["force_1"]) >= 0.06 * weight
                and not row["body_contact"]
            )
            handoff_hold = handoff_hold + cfg.sim_dt if candidate else 0.0
            best_handoff = max(best_handoff, handoff_hold)
        result = dict(
            seed=seed,
            handoff_candidate=bool(best_handoff >= 0.06),
            valid_takeoff=bool(
                best >= 0.02
                and not any(x["body_contact"] for x in r)
                and max(x["front_drift"] for x in r) <= 0.04
            ),
            hold_30mm_s=best,
            peak_min_clearance_m=max(x["clearance_min"] for x in r if x["time"] >= 0.7),
            max_front_drift_m=max(x["front_drift"] for x in r),
            peak_pitch_rad=max(x["pitch_rad"] for x in r),
            peak_abs_pitch_velocity=max(abs(x["pitch_rate"]) for x in r),
            terminal_pitch_rad=r[-1]["pitch_rad"],
            terminal_pitch_velocity=r[-1]["pitch_rate"],
            flight_mean_pitch_rad=float(np.mean([x["pitch_rad"] for x in post])),
            front_contact_fraction=sum(
                min(x["force_0"], x["force_1"]) >= 0.06 * weight
                for x in r
                if x["time"] > 0.15
            )
            / sum(x["time"] > 0.15 for x in r),
            push_mean_abs_residual_rad=float(
                np.mean([[abs(x[f"residual_{j}"]) for j in range(6, 12)] for x in push])
            ),
            max_abs_residual_rad=max(
                abs(x[f"residual_{j}"]) for x in r for j in range(12)
            ),
            curriculum_success=bool(env._best_hold[i] >= 0.06),
            nonfoot_contact=any(x["body_contact"] for x in r),
            overrotation=any(
                x["pitch_rad"] > 1.75 or abs(x["pitch_rate"]) > 4.0 for x in r
            ),
        )
        results.append(result)
        if not checkpoint:
            native, nrows = ex.trial(env.reference_parameters, seed=seed, record=True)
            diffs = {
                key: max(abs(a[key] - b[key]) for a, b in zip(r, nrows))
                for key in ["clearance_min", "front_drift", "pitch_rad", "pitch_rate"]
                + [f"actual_{j}" for j in range(12)]
                + [f"torque_{j}" for j in range(12)]
            }

            def timing(samples, front):
                if front:
                    return next(
                        (
                            x["time"]
                            for x in samples
                            if min(x["force_0"], x["force_1"]) >= 0.06 * weight
                        ),
                        None,
                    )
                return next(
                    (
                        x["time"]
                        for x in samples
                        if min(x["clearance_2"], x["clearance_3"]) >= 0.01
                        and max(x["force_2"], x["force_3"]) < 0.015 * weight
                    ),
                    None,
                )

            timing_diffs = {
                name: abs(timing(r, front) - timing(nrows, front))
                for name, front in [
                    ("front_loading", True),
                    ("rear_geometric_liftoff", False),
                ]
            }
            parity.append(
                dict(
                    seed=seed,
                    native=native,
                    max_differences=diffs,
                    contact_timing_differences_s=timing_diffs,
                )
            )
        with (dest / f"trial_{seed}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=r[0])
            writer.writeheader()
            writer.writerows(r)
    report = dict(
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if checkpoint
        else None,
        level=level,
        config_source=config_source,
        results=results,
        parity=parity,
        config=class_to_dict(cfg),
    )
    (dest / "evaluation.json").write_text(json.dumps(report, indent=2))
    env._backend.close()
    print(
        "takeoff",
        sum(x["valid_takeoff"] for x in results),
        "/16",
        "clearance",
        np.mean([x["peak_min_clearance_m"] for x in results]),
        "drift",
        np.mean([x["max_front_drift_m"] for x in results]),
        flush=True,
    )
    if parity:
        print(
            "parity maxima",
            {
                k: max(x["max_differences"][k] for x in parity)
                for k in parity[0]["max_differences"]
            },
            flush=True,
        )
        assert sum(x["valid_takeoff"] for x in results) == 16, (
            "Zero-residual takeoff parity failed"
        )
        maxima = {
            k: max(x["max_differences"][k] for x in parity)
            for k in parity[0]["max_differences"]
        }
        assert maxima["clearance_min"] < 1e-5 and maxima["front_drift"] < 1e-5
        assert maxima["pitch_rad"] < 5e-5 and maxima["pitch_rate"] < 0.005
        assert max(v for k, v in maxima.items() if k.startswith("actual_")) < 5e-5
        assert max(v for k, v in maxima.items() if k.startswith("torque_")) < 0.01
        assert (
            max(v for x in parity for v in x["contact_timing_differences_s"].values())
            <= 0.0041
        )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--level", type=int)
    p.add_argument("--output-dir", default="output/frontyield_parity")
    args = p.parse_args()
    evaluate(args.checkpoint, args.output_dir, args.level)
