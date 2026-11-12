"""Identification from training trajectories and genuinely held-out forecasts."""

import numpy as np
import pandas as pd

from experiments.morphology.learning.pendulum_reference import acceleration, quintic


MODELS = ("fixed", "variable", "configuration_inertia")


def smooth(x, width=11):
    return np.convolve(
        np.pad(np.asarray(x), (width // 2, width // 2), mode="edge"),
        np.ones(width) / width,
        mode="valid",
    )


def enrich(frame, mass):
    d = frame.copy()
    if "length_rate" in d:
        d = d.rename(columns={"length_rate": "cop_length_rate"})
    t = d.time.to_numpy()
    dt = float(np.median(np.diff(t)))
    px = (d.FL_position_x + d.FR_position_x).to_numpy() / 2
    py = (d.FL_position_y + d.FR_position_y).to_numpy() / 2
    vx = (d.FL_velocity_x + d.FR_velocity_x).to_numpy() / 2
    dx = d.com_x.to_numpy() - px
    dz = d.com_z.to_numpy()
    rvx = d.com_velocity_x.to_numpy() - vx
    rvz = d.com_velocity_z.to_numpy()
    L = np.hypot(dx, dz)
    omega = (dz * rvx - dx * rvz) / L**2
    theta = np.unwrap(np.arctan2(dx, dz))
    Ldot = (dx * rvx + dz * rvz) / L
    alpha = np.gradient(smooth(omega), dt)
    Idot = np.gradient(smooth(d.inertia_com_y), dt)
    ax = np.gradient(smooth(vx), dt)
    pivot_term = -mass * dz * ax
    front_slip = np.maximum(
        np.hypot(d.FL_velocity_x, d.FL_velocity_y),
        np.hypot(d.FR_velocity_x, d.FR_velocity_y),
    )
    both = (d.FL_load > 0.05 * mass * 9.81) & (d.FR_load > 0.05 * mass * 9.81)
    eligible = (
        both
        & (d.rear_force_z < 0.04 * mass * 9.81)
        & (front_slip < 0.05)
        & ~d.nonfoot_collision
        & ~d.self_collision
        & (d.front_drift < 0.04)
        & (d.max_speed_ratio <= 1.001)
        & (d.joint_violation < 0.001)
    ).to_numpy()
    eligible = np.convolve(eligible.astype(int), np.ones(11), mode="same") == 11
    first = np.flatnonzero(eligible)
    L0 = float(L[first[0]]) if len(first) else float(L[0])
    columns = dict(
        theta=theta,
        omega=omega,
        length=L,
        length_rate=Ldot,
        theta_acceleration=alpha,
        inertia_rate=Idot,
        pivot_term=pivot_term,
        pivot_x=px,
        pivot_y=py,
        pivot_velocity_x=vx,
        front_slip=front_slip,
        eligible=eligible,
        radial_acceleration=np.gradient(smooth(Ldot), dt),
        fixed_length=L0,
    )
    contacts_y = d.front_moment_y + d.rear_moment_y + d.nonfoot_moment_y
    Fx = d.front_force_x + d.rear_force_x + d.nonfoot_force_x
    Fz = d.front_force_z + d.rear_force_z + d.nonfoot_force_z
    columns["contact_moment_pivot"] = contacts_y + dz * Fx - dx * Fz
    shape = d.momentum_y - d.inertia_com_y * omega
    columns["shape_momentum"] = shape
    columns["physical_effective_torque"] = columns[
        "contact_moment_pivot"
    ] - np.gradient(smooth(shape), dt)
    for name in MODELS:
        length_used = np.full(len(d), L0) if name == "fixed" else L
        rate = np.zeros(len(d)) if name == "fixed" else Ldot
        ic = d.inertia_com_y.to_numpy() if name == "configuration_inertia" else 0.0
        ir = Idot if name == "configuration_inertia" else 0.0
        columns[f"torque_identified_{name}"] = (
            (mass * length_used**2 + ic) * alpha
            + (2 * mass * length_used * rate + ir) * omega
            - mass * 9.81 * length_used * np.sin(theta)
            - pivot_term
        )
    for j, indices in enumerate(((0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11))):
        columns[f"command_{j}"] = (
            d[f"action_{indices[0]}"] + d[f"action_{indices[1]}"]
        ) / 2
    return pd.concat([d, pd.DataFrame(columns, index=d.index)], axis=1)


FEATURES = (
    "theta",
    "omega",
    "length",
    "length_rate",
    "inertia_com_y",
    *[f"command_{i}" for i in range(6)],
)


def fit_regression(X, y):
    mean = X.mean(0)
    scale = np.maximum(X.std(0), 0.01)
    Z = np.c_[np.ones(len(X)), (X - mean) / scale]
    reg = np.eye(Z.shape[1]) * 0.1
    reg[0, 0] = 1e-8
    coef = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
    return dict(mean=mean.tolist(), scale=scale.tolist(), coef=coef.tolist())


def predict(regression, X):
    Z = (np.asarray(X) - np.array(regression["mean"])) / np.array(regression["scale"])
    return np.c_[np.ones(len(np.atleast_2d(Z))), np.atleast_2d(Z)] @ np.array(
        regression["coef"]
    )


def identify(training, mass):
    d = pd.concat(training, ignore_index=True)
    d = d[d.eligible]
    if len(d) < 30:
        raise ValueError("Insufficient valid identification samples")
    X = d[list(FEATURES)].to_numpy()
    model = dict(
        mass=mass,
        features=list(FEATURES),
        training_samples=len(d),
        training_state_min=X.min(0).tolist(),
        training_state_max=X.max(0).tolist(),
    )
    model["torques"] = {
        name: fit_regression(X, d[f"torque_identified_{name}"].to_numpy())
        for name in MODELS
    }
    model["radial"] = fit_regression(X, d.radial_acceleration.to_numpy())
    model["inertia_rate"] = fit_regression(X, d.inertia_rate.to_numpy())
    gain = float(
        np.linalg.lstsq(np.c_[np.ones(len(d)), d.pitch], d.theta, rcond=None)[0][1]
    )
    model["observed_com_body_gain"] = gain
    model["reference_com_body_gain"] = float(np.clip(gain, 0.1, 2.0))
    model["torque_envelope"] = {
        name: np.quantile(d[f"torque_identified_{name}"], [0.01, 0.99]).tolist()
        for name in MODELS
    }
    return model


def validate(rollouts, model):
    errors = []
    forecasts = []
    conditional = []
    mass = model["mass"]
    for tag, d in rollouts:
        X = d[list(FEATURES)].to_numpy()
        valid = d.eligible.to_numpy()
        for name in MODELS:
            tau = predict(model["torques"][name], X)
            prediction = []
            for k, r in d.iterrows():
                fixed = name == "fixed"
                full = name == "configuration_inertia"
                prediction.append(
                    acceleration(
                        r.theta,
                        r.omega,
                        r.fixed_length if fixed else r.length,
                        0.0 if fixed else r.length_rate,
                        mass,
                        tau[k],
                        r.inertia_com_y if full else 0.0,
                        float(predict(model["inertia_rate"], X[k : k + 1])[0])
                        if full
                        else 0.0,
                        r.pivot_term,
                    )
                )
            actual = d.theta_acceleration.to_numpy()
            prediction = np.array(prediction)
            errors.append(
                dict(
                    rollout=tag,
                    model=name,
                    kind="one_step_current_state_and_support_acceleration",
                    samples=int(valid.sum()),
                    acceleration_rmse=float(
                        np.sqrt(np.mean((prediction[valid] - actual[valid]) ** 2))
                    )
                    if valid.any()
                    else None,
                )
            )
            for k in np.flatnonzero(valid):
                conditional.append(
                    dict(
                        rollout=tag,
                        model=name,
                        time=float(d.time.iloc[k]),
                        measured_acceleration=float(actual[k]),
                        predicted_acceleration=float(prediction[k]),
                    )
                )
            # 100ms forecasts: initial measured state + known command sequence.
            # No future measured COM, L, Ldot, inertia, torque or pivot acceleration.
            for start in range(0, len(d) - 50, 25):
                if not valid[start : start + 51].all():
                    continue
                initial = d.iloc[start]
                theta = initial.theta
                omega = initial.omega
                L = initial.length
                Ldot = initial.length_rate
                inertia = initial.inertia_com_y
                theta_errors = []
                omega_errors = []
                clamps = 0
                for k in range(start, start + 50):
                    r = d.iloc[k]
                    dt = float(d.time.iloc[k + 1] - r.time)
                    features = np.r_[
                        theta,
                        omega,
                        L,
                        Ldot,
                        inertia,
                        r[[f"command_{i}" for i in range(6)]].to_numpy(dtype=float),
                    ][None, :]
                    tau = float(predict(model["torques"][name], features)[0])
                    radial = (
                        float(predict(model["radial"], features)[0])
                        if name != "fixed"
                        else 0.0
                    )
                    Idot = (
                        float(predict(model["inertia_rate"], features)[0])
                        if name == "configuration_inertia"
                        else 0.0
                    )
                    a = acceleration(
                        theta,
                        omega,
                        L,
                        0.0 if name == "fixed" else Ldot,
                        mass,
                        tau,
                        inertia if name == "configuration_inertia" else 0.0,
                        Idot,
                    )
                    theta += omega * dt
                    omega += a * dt
                    if name != "fixed":
                        L += Ldot * dt
                        Ldot += radial * dt
                    if name == "configuration_inertia":
                        inertia += Idot * dt
                    if not (
                        0.15 < L < 0.65
                        and abs(omega) < 20
                        and abs(Ldot) < 3
                        and 0.05 < inertia < 3
                    ):
                        clamps += 1
                    L = float(np.clip(L, 0.15, 0.65))
                    Ldot = float(np.clip(Ldot, -3, 3))
                    inertia = float(np.clip(inertia, 0.05, 3))
                    omega = float(np.clip(omega, -20, 20))
                    truth = d.iloc[k + 1]
                    theta_errors.append(theta - truth.theta)
                    omega_errors.append(omega - truth.omega)
                    forecasts.append(
                        dict(
                            rollout=tag,
                            model=name,
                            start_time=float(initial.time),
                            time=float(truth.time),
                            theta_measured=float(truth.theta),
                            theta_predicted=float(theta),
                            omega_measured=float(truth.omega),
                            omega_predicted=float(omega),
                            length_measured=float(truth.length),
                            length_predicted=L,
                        )
                    )
                errors.append(
                    dict(
                        rollout=tag,
                        model=name,
                        kind="100ms_initial_state_plus_commands_fixed_pivot",
                        samples=50,
                        theta_rmse=float(np.sqrt(np.mean(np.array(theta_errors) ** 2))),
                        omega_rmse=float(np.sqrt(np.mean(np.array(omega_errors) ** 2))),
                        clamped_steps=clamps,
                    )
                )
    return errors, forecasts, conditional


def reference(initial, model, horizon=0.7, transition=0.3, shorten=0.01):
    t = np.arange(0, horizon + 0.001, 0.02)
    body, body_v, body_a = quintic(
        initial["pitch"], initial["pitch_velocity"], np.deg2rad(45), transition, t
    )
    gain = model["reference_com_body_gain"]
    theta_target = initial["theta"] + gain * (np.pi / 4 - initial["pitch"])
    theta, omega, alpha = quintic(
        initial["theta"], initial["omega"], theta_target, transition, t
    )
    L, Ldot, Lddot = quintic(
        initial["length"],
        initial["length_rate"],
        initial["length"] - shorten,
        transition,
        t,
    )
    mass = model["mass"]
    inertia = initial["inertia_com_y"]
    torque_fixed = mass * initial["length"] ** 2 * alpha - mass * 9.81 * initial[
        "length"
    ] * np.sin(theta)
    torque_variable = (
        mass * L**2 * alpha
        + 2 * mass * L * Ldot * omega
        - mass * 9.81 * L * np.sin(theta)
    )
    torque_full = (
        (inertia + mass * L**2) * alpha
        + 2 * mass * L * Ldot * omega
        - mass * 9.81 * L * np.sin(theta)
    )
    return pd.DataFrame(
        dict(
            time=t,
            pitch=body,
            pitch_velocity=body_v,
            theta=theta,
            omega=omega,
            length=L,
            length_rate=Ldot,
            com_support_x=L * np.sin(theta),
            com_support_z=L * np.cos(theta),
            torque_fixed=torque_fixed,
            torque_variable=torque_variable,
            torque_configuration_inertia=torque_full,
        )
    )
