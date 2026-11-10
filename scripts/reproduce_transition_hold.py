"""Reproduce and instrument the historical seed-3009, 3.88-second bridge."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.evaluate_kickup_handoff import (
    Policy,
    Experiment,
    kick,
    recovery,
    measurements,
    dump,
)

KICK = (
    "logs/go2frontyield_fresh_200_20261110_034101_1da5fd/Nov10_03-42-51_/model_200.pt"
)
BALANCE = "logs/go2balance/Oct31_23-29-27_/model_1500.pt"
REFERENCE = "output/handoff_reference_search/candidate_reference.json"


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--viewer", action="store_true")
    args = p.parse_args()
    dest = args.output_dir
    dest.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    with torch.inference_mode():
        b = Policy(BALANCE, "go2balance", 16, True)
        k = Policy(KICK, "go2frontyield", 16)
        run = kick(
            k,
            Experiment(),
            dest,
            "candidate",
            parameters=json.loads(Path(REFERENCE).read_text()),
        )
        saved = np.load("output/handoff_bridge_probe/candidate_states.npz")
        error = max(
            np.max(np.abs(value - saved[f"1.2_{i}_{kind}"]))
            for i, sample in enumerate(run["snapshots"][1.2])
            for kind, value in zip(("qpos", "qvel"), sample)
        )
        records = []
        e = b.env
        post = e._post_physics_step
        action = b.action
        policy_targets = np.zeros((16, 12))
        counter = 0
        handle = None

        def capture_action(zero=False):
            action(zero)
            policy_targets[:] = (e._standing_posture() + e.dof_pos_target).numpy()

        def capture():
            nonlocal counter, handle
            post()
            counter += 1
            row = measurements(e)[9]
            d = e._backend._datas[9]
            m = e._backend._model_for_env(9)
            for axis, value in zip("xyz", d.qpos[:3]):
                row[f"base_{axis}"] = float(value)
            for j, value in enumerate(e.root_states[9]):
                row[f"root_state_{j}"] = float(value)
            for axis, value in zip("xyz", d.subtree_com[1]):
                row[f"com_world_{axis}"] = float(value)
            feet = e._backend._canonical_to_native_body_np[e.feet_indices.numpy()]
            contact_bodies = []
            import mujoco

            for c in d.contact:
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if c.dist < -1e-5 and (b1 == 0 or b2 == 0):
                    body = b2 if b1 == 0 else b1
                    if body not in feet:
                        contact_bodies.append(
                            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body)
                        )
            row["nonfoot_contact_bodies"] = json.dumps(contact_bodies)
            row.update(
                time=counter * e.cfg.sim_dt,
                episode_time=float(e.episode_length_buf[9]) * e.dt,
                terminated=bool(e.terminated[9]),
                timed_out=bool(e.timed_out[9]),
            )
            for j in range(12):
                row[f"torque_{j}"] = float(e.torques[9, j])
                row[f"target_{j}"] = float(
                    (e._standing_posture() + e.dof_pos_target)[9, j]
                )
                row[f"policy_target_{j}"] = float(policy_targets[9, j])
                row[f"velocity_limit_{j}"] = float(
                    e.dof_vel_limits[e.actuated_dof_indices[j]]
                )
                row[f"torque_limit_{j}"] = float(e.actuated_torque_limits[j])
            records.append(row)
            if args.viewer:
                import mujoco.viewer

                if handle is None:
                    handle = mujoco.viewer.launch_passive(
                        e._backend._model_for_env(9), e._backend._datas[9]
                    )
                    handle.cam.distance = 1.5
                    handle.cam.azimuth = 90
                handle.sync()

        reports = []
        for t in (1.1, 1.2):
            for duration in (0.1, 0.4, 0.6):
                detailed = t == 1.2 and duration == 0.6
                if detailed:
                    e._post_physics_step = capture
                    b.action = capture_action
                reports.append(
                    recovery(
                        b,
                        run["snapshots"][t],
                        dest,
                        f"bridge_{t}_{duration}",
                        run["targets"][t],
                        run["models"],
                        duration,
                    )
                )
                if detailed:
                    e._post_physics_step = post
                    b.action = action
        with (dest / "seed_3009_physics.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=records[0])
            w.writeheader()
            w.writerows(records)
        assert error < 1e-10, "Historical kick state did not reproduce"
        assert abs(reports[-1]["trials"][9]["hold_s"] - 3.88) < 1e-8
        dump(
            dest / "reproduction.json",
            dict(
                seed=3009,
                trial=9,
                state_max_absolute_error=error,
                expected_hold_s=3.88,
                reproduced_hold_s=reports[-1]["trials"][9]["hold_s"],
                simulated_recovery_seconds=counter * e.cfg.sim_dt,
                configured_episode_seconds=e.max_episode_length_s,
                native_qpos_at_switch=run["snapshots"][1.2][9][0].tolist(),
                native_qvel_at_switch=run["snapshots"][1.2][9][1].tolist(),
                commanded_target_at_switch=run["targets"][1.2][9].tolist(),
                balance_p_gains=e.p_gains[9].tolist(),
                balance_d_gains=e.d_gains[9].tolist(),
                warmstart_zeroed_at_switch=True,
                balance=b.meta,
                kick=k.meta,
                reports=reports,
            ),
        )
        if handle is not None:
            handle.close()
        b.env._backend.close()
        k.env._backend.close()


if __name__ == "__main__":
    main()
