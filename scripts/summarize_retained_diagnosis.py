"""Phase/joint errors on identical observations plus synchronized native traces."""

import argparse, csv, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLCONFIGDIR="/tmp/retained_mpl"
)
import numpy as np, torch
from experiments.unified_go2.core import (
    Student as Previous,
    adc_network,
    load,
    save_json,
)
from experiments.retained_go2.actor import Student


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    prep = Path("output/unified_go2_preparation_v1")
    data = np.load(prep / "demonstrations.npz")
    x = torch.tensor(data["observations"])
    target = torch.tensor(data["actions"])
    adc = adc_network(load(prep / "adc.pt"))
    bad = Previous()
    bad.load_state_dict(load("output/unified_go2_initial_v1/student.pt")["network"])
    initial_bad = Previous.from_adc(load(prep / "adc.pt"), load(prep / "balance.pt"))
    good = Student()
    good.load_state_dict(
        load("output/retained_go2_distillation_v3/student.pt")["network"]
    )
    with torch.no_grad():
        values = {
            name: net.actor(x).tanh()
            for name, net in [
                ("adc", adc),
                ("bad_initialized", initial_bad),
                ("previous_trained", bad),
                ("corrected", good),
            ]
        }
    phases = {
        "ADC_loaded": (data["roles"] == 0)
        & (data["observations"][:, 89:91].sum(-1) > 0.05),
        "ADC_airborne": (data["roles"] == 0)
        & (data["observations"][:, 89:91].sum(-1) <= 0.05),
        "balance": data["roles"] == 1,
    }
    report = {}
    for name, prediction in values.items():
        report[name] = {
            phase: dict(
                n=int(mask.sum()),
                mse=float((prediction[mask] - target[mask]).square().mean()),
                max_error=float((prediction[mask] - target[mask]).abs().max()),
                per_joint_mse=(prediction[mask] - target[mask])
                .square()
                .mean(0)
                .tolist(),
            )
            for phase, mask in phases.items()
            if mask.any()
        }
    report["normalization"] = dict(
        expert_running_statistics=None,
        common_inputs_unchanged=True,
        router_center=good.actor.center.tolist(),
        router_scale=good.actor.scale.tolist(),
        router_inputs=good.actor.indices.tolist(),
        scope="Router statistics never transform the ADC/balance head inputs",
    )
    pairs = {
        "ADC": "output/retained_go2_validation_v1/videos/adc_standing_0_physics.csv",
        "previous": "output/unified_go2_evaluation_initial_v2/videos/student_standing_0_physics.csv",
        "corrected": "output/retained_go2_validation_v3/videos/student_standing_0_physics.csv",
    }
    rows = {name: list(csv.DictReader(open(path))) for name, path in pairs.items()}
    keys = [
        k
        for k in rows["ADC"][0]
        if k.endswith(("_target", "_tau", "_v"))
        or k
        in [
            "rear_force_z",
            "front_moment_com",
            "rear_moment_com",
            "rear_positive_power",
            "pitch",
            "pitch_velocity",
            "L_com",
        ]
    ]
    report["matched_ADC_corrected_max_difference"] = {
        k: float(
            np.max(
                abs(
                    np.array([float(r[k]) for r in rows["ADC"]])
                    - np.array([float(r[k]) for r in rows["corrected"]])
                )
            )
        )
        for k in keys
    }
    report["available_prefix_seconds"] = {
        name: float(rs[-1]["time"]) for name, rs in rows.items()
    }
    save_json(a.output / "phase_joint_diagnosis.json", report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    columns = [
        "RL_calf_joint_target",
        "FL_calf_joint_tau",
        "rear_force_z",
        "front_moment_com",
        "pitch_velocity",
        "L_com",
    ]
    units = [
        "Rear knee target (rad)",
        "Front knee torque (Nm)",
        "Rear force z (N)",
        "Front moment (Nm)",
        "Pitch rate (rad/s)",
        "L_COM (kg m²/s)",
    ]
    for name, color in [
        ("ADC", "tab:blue"),
        ("previous", "tab:red"),
        ("corrected", "tab:green"),
    ]:
        rs = [r for r in rows[name] if float(r["time"]) <= 0.8]
        t = [float(r["time"]) for r in rs]
        for ax, key in zip(axes, columns):
            ax.plot(
                t,
                [float(r[key]) for r in rs],
                label=name,
                color=color,
                linestyle="--" if name == "corrected" else "-",
            )
    for ax, unit in zip(axes, units):
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Actual simulation time (s); previous episode ends at0.22s")
    fig.tight_layout()
    fig.savefig(a.output / "takeoff_motion_torque.png", dpi=140)
    plt.close(fig)
    # Actual capture-state reference differences, not necessary energy/H lower bounds.
    bank = load("output/unified_go2_preparation_v1/bank.pt")
    capture = min(
        bank["train"]["balance"], key=lambda s: s["values"]["momentum_sample"]["pitch"]
    )
    s = capture["values"]["momentum_sample"]
    save_json(
        a.output / "capture_reference.json",
        {
            k: s[k]
            for k in [
                "pitch",
                "pitch_velocity",
                "com_height",
                "mechanical_energy",
                "potential",
                "L_com",
                "front_moment_com",
                "com_support_x",
            ]
        },
    )
    print(
        "Saved phase errors, exact ADC/retained differences, motion/torque plot and capture reference",
        flush=True,
    )


if __name__ == "__main__":
    main()
