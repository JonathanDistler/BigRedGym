"""Measured sweep/foot-motion timing; standing and intermediate starts stay separate."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(sorted(args.directory.glob('checkpoint_*.pt'))[-1],
                       weights_only=False, map_location='cpu')
    rows = []
    plots = []
    for name in saved['history']['trace_files']:
        path = Path(name)
        if not path.exists():
            path = args.directory / Path(*path.parts[path.parts.index('sessions'):])
        frame = pd.read_csv(path)
        for (kind, seed), f in frame.groupby(['start_kind', 'evaluation_seed']):
            f = f.sort_values('time')
            safe = f[f.prefix_safe.astype(bool)]
            if safe.empty:
                continue
            apex = safe.loc[safe.pitch.idxmax()]
            airborne = safe[safe.historical_takeoff.astype(bool)]
            onset = airborne[(airborne[['FL_backward', 'FR_backward']].max(axis=1) > .005)]
            sweep = airborne[(airborne.rear_coordination > .1) &
                             (airborne.rear_relative_momentum < -.02)]
            motion_time = float(onset.time.iloc[0]) if len(onset) else np.nan
            sweep_time = float(sweep.time.iloc[0]) if len(sweep) else np.nan
            before = airborne[airborne.time <= apex.time]
            rows.append(dict(mode=path.stem.rsplit('_', 1)[0],
                             iteration=int(path.stem.rsplit('_', 1)[1]),
                             start_kind=kind, evaluation_seed=seed,
                             peak_pitch_degrees=np.rad2deg(apex.pitch), peak_time=apex.time,
                             front_motion_time=motion_time, rear_sweep_time=sweep_time,
                             front_motion_lead_seconds=float(apex.time)-motion_time,
                             rear_sweep_lead_seconds=float(apex.time)-sweep_time,
                             front_motion_before_peak=bool(motion_time < apex.time),
                             rear_sweep_before_peak=bool(sweep_time < apex.time),
                             peak_L_com=apex.L_com, peak_L_support=apex.L_support,
                             peak_com_x=apex.com_x, peak_com_z=apex.com_z,
                             front_counter_impulse_before_peak=float(np.trapezoid(
                                 np.minimum(before.front_moment_com, 0), before.time)) if len(before)>1 else 0.,
                             early_reward=float(f.reward_early_placement.sum()),
                             sweep_reward=float(f.reward_rear_sweep.sum()),
                             head_collision=bool(f.head_collision.any()),
                             calf_collision=bool(f.calf_collision.any())))
            if kind == 'standing' and seed == 96000 and int(path.stem.rsplit('_',1)[1]) == saved['iteration']:
                plots.append((path.stem, f, float(apex.time)))
    pd.DataFrame(rows).to_csv(args.output/'timing.csv', index=False)
    for mode, f, apex in plots:
        fig, axes = plt.subplots(6, 1, figsize=(9, 13), sharex=True)
        axes[0].plot(f.time, np.rad2deg(f.pitch), label='pitch (deg)')
        axes[0].plot(f.time, f.pitch_velocity, label='pitch rate (rad/s)')
        axes[1].plot(f.time, f.FL_backward, label='FL backward (m)')
        axes[1].plot(f.time, f.FR_backward, label='FR backward (m)')
        for col in ['rear_hip_L_velocity', 'rear_hip_R_velocity', 'rear_knee_L_velocity', 'rear_knee_R_velocity']:
            axes[2].plot(f.time, f[col], label=col+' (rad/s)')
        for col in ['L_com', 'L_support', 'rear_relative_momentum']:
            axes[3].plot(f.time, f[col], label=col+' (kg m²/s)')
        axes[4].plot(f.time, f.front_moment_com, label='front moment (Nm)')
        axes[4].plot(f.time, f.rear_moment_com, label='rear moment (Nm)')
        axes[5].plot(f.time, f.com_x, label='COM x (m)')
        axes[5].plot(f.time, f.com_z, label='COM z (m)')
        axes[5].plot(f.time, f.phase, label='phase')
        for ax in axes:
            ax.axvline(apex, color='k', linestyle=':', label='safe peak')
            collision = f[~f.prefix_safe.astype(bool)]
            if len(collision):
                ax.axvspan(collision.time.iloc[0], f.time.max(), alpha=.12, color='red')
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(alpha=.2)
        axes[-1].set_xlabel('Simulation time (s); red shading marks invalid motion')
        fig.suptitle(mode+'; raw displacement alone is not catch success')
        fig.tight_layout()
        for suffix in ['png', 'pdf']:
            fig.savefig(args.output/f'{mode}_timing.{suffix}', dpi=140)
        plt.close(fig)


if __name__ == '__main__':
    main()
