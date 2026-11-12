#!/usr/bin/env python3
"""Go2 front-supported handstand: bounded, checkpoint-initialized PPO curriculum.

Run from the ROOT of JonathanDistler/BigRedGym, branch
experiment/pendulum_guided_rotation, with the repository's Python 3.11 environment.

This is a NEW opt-in experiment; it does not modify existing checkpoints or repo files.
The source checkpoint must be from the 44-obs / 74-critic / 12-action physical
morphology RotationRun lineage, not a different BigRedGym RSL-RL policy.

Requires an explicitly supplied source checkpoint from a locally restored,
user-trusted archive. Does NOT imitate the invalid 51-degree flight trajectory.

Examples:
  python molab_go2_handstand_curriculum.py --preflight --checkpoint PATH/model.pt
  python molab_go2_handstand_curriculum.py --checkpoint PATH/model.pt --max-updates 2
  python molab_go2_handstand_curriculum.py --checkpoint PATH/model.pt --max-updates 40
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

# When launched from a Molab-uploaded script, import the cloned repository from cwd.
sys.path.insert(0, str(Path.cwd().resolve()))

os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('MUJOCO_GL', 'egl' if os.environ.get('DISPLAY') is None else 'glfw')

import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic, advantages, update
from experiments.morphology.learning.rotation import RotationConfig, RotationEnvironment

STAGES = [
    dict(name='certified_takeoff', degrees=30, pass_rate=.90, hold=0.0),
    dict(name='supported_45', degrees=45, pass_rate=.80, hold=0.0),
    dict(name='supported_60_recovery', degrees=60, pass_rate=.70, hold=.15),
    dict(name='supported_75', degrees=75, pass_rate=.70, hold=.15),
    dict(name='supported_85', degrees=85, pass_rate=.70, hold=.20),
    dict(name='upright_catch', degrees=90, pass_rate=.60, hold=.30),
    dict(name='five_second_handstand', degrees=90, pass_rate=.50, hold=5.0),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x)))


def append_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction='ignore')
        if new:
            writer.writeheader()
        writer.writerows(rows)


def event_seed(training_seed, world, episode):
    return int(np.random.SeedSequence([training_seed, world, episode]).generate_state(1)[0])


def class_cfg(cls, d):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


class HandstandCurriculumEnvironment(RotationEnvironment):
    """Keep native joint motors, limits, collisions and 44/74 observations.

    Add reward shaping and physics-rate support/catch tracking only. No pose
    injection, invisible forces, contact suppression or automatic demonstrations.
    """
    def __init__(self, assets, cfg):
        self.curriculum_stage = 0
        self.stage_phi = 0.0
        self._substep_time = None
        self._last_feet = None
        self._last_loaded = False
        self.peak_touchdown_speed_mps = 0.0
        self.catch_snapshot = None
        self._front_gap = 0.0
        self.max_front_gap = 0.0
        self.contact_slip_distance = 0.0
        self.best_catch = 0.0
        self.catch_timer = 0.0
        self.certified_supported_pitch = 0.0
        self.supported_crossing = False
        self.crossing_snapshot = None
        self.stage_success = False
        super().__init__(assets, cfg)

    def reset(self, *args, **kwargs):
        self.stage_phi = 0.0
        self._substep_time = None
        self._last_feet = None
        self._last_loaded = False
        self.peak_touchdown_speed_mps = 0.0
        self.catch_snapshot = None
        self._front_gap = 0.0
        self.max_front_gap = 0.0
        self.contact_slip_distance = 0.0
        self.best_catch = 0.0
        self.catch_timer = 0.0
        self.certified_supported_pitch = 0.0
        self.supported_crossing = False
        self.crossing_snapshot = None
        self.stage_success = False
        return super().reset(*args, **kwargs)

    def set_stage(self, stage):
        self.curriculum_stage = stage
        self.stage_phi = 0.0

    def measure(self):
        row = super().measure()
        if getattr(self, 'tracker', None) is None:
            return row
        now = float(self.d.time)
        if self._substep_time is None:
            self._substep_time = now
            self._last_feet = row['feet'][:2].copy()
            self._last_loaded = bool(row['front_support'])
            return row
        delta_t = now - self._substep_time
        if delta_t <= 1e-10:
            return row
        self._substep_time = now
        loaded = bool(row['front_support'])
        feet = row['feet'][:2].copy()
        if loaded and self._last_loaded:
            self.contact_slip_distance += float(np.linalg.norm(feet[:, :2] - self._last_feet[:, :2], axis=1).max())
        if loaded and not self._last_loaded and self.tracker.confirmed:
            self.peak_touchdown_speed_mps = max(
                self.peak_touchdown_speed_mps,
                float(np.max(np.abs((feet[:, 2] - self._last_feet[:, 2]) / delta_t))))
        self._last_feet, self._last_loaded = feet, loaded
        if self.tracker.confirmed:
            self._front_gap = self._front_gap + delta_t if not loaded else 0.0
            self.max_front_gap = max(self.max_front_gap, self._front_gap)
        valid_native = (not row['illegal_contacts'] and not row['self_contacts']
                        and self.max_speed_ratio <= 1.001
                        and self.max_joint_violation < .001)
        allowed_gap = .02 if self.curriculum_stage <= 1 else (.40 if self.curriculum_stage == 2 else .08)
        rear_min = .03 if self.curriculum_stage <= 1 else .10
        secured = (self.tracker.confirmed and loaded
                   and row['front_drift'] <= .04
                   and self.max_front_gap <= allowed_gap + 1e-9
                   and self.contact_slip_distance <= .02 + 1e-9
                   and self.peak_touchdown_speed_mps <= 1.0
                   and row['rear_clearance'] > rear_min
                   and row['loads'][2:].sum() < .04 * self.weight
                   and valid_native)
        if secured:
            self.certified_supported_pitch = max(self.certified_supported_pitch, float(row['pitch']))
            angle = STAGES[self.curriculum_stage]['degrees']
            if row['pitch'] >= math.radians(angle) and row['pitch_velocity'] >= -.5:
                self.supported_crossing = True
                if self.crossing_snapshot is None:
                    self.crossing_snapshot = dict(pitch_degrees=math.degrees(row['pitch']),
                                                  front_drift_m=float(row['front_drift']),
                                                  contact_gap_s=float(self.max_front_gap),
                                                  slip_m=float(self.contact_slip_distance))
        angle = STAGES[self.curriculum_stage]['degrees']
        capture = (secured and abs(math.degrees(row['pitch']) - angle) <= 7.5
                   and abs(row['pitch_velocity']) <= (.5 if angle >= 85 else .8)
                   and np.linalg.norm(row['com_velocity']) <= (.25 if angle >= 85 else .75))
        self.catch_timer = self.catch_timer + delta_t if capture else 0.0
        self.best_catch = max(self.best_catch, self.catch_timer)
        if capture and (self.catch_snapshot is None or self.catch_timer > self.catch_snapshot['duration']):
            self.catch_snapshot = dict(duration=float(self.catch_timer),
                                       front_drift_m=float(row['front_drift']),
                                       contact_gap_s=float(self.max_front_gap),
                                       slip_m=float(self.contact_slip_distance),
                                       touchdown_speed_mps=float(self.peak_touchdown_speed_mps))
        return row

    def step(self, action):
        observations, reward, done, info = super().step(action)
        stage = self.curriculum_stage
        if stage >= 1:
            s = self.row
            goal = STAGES[stage]['degrees']
            allowed_gap = .02 if stage <= 1 else (.40 if stage == 2 else .08)
            valid = (self.tracker.confirmed and s['front_support']
                     and s['rear_clearance'] >= .015
                     and s['loads'][2:].sum() < .04 * self.weight
                     and not self.nonfoot_collision and not self.self_collision
                     and s['front_drift'] <= .04 and self.max_front_gap <= allowed_gap
                     and self.contact_slip_distance <= .02
                     and self.peak_touchdown_speed_mps <= 1.0
                     and self.max_speed_ratio <= 1.001
                     and self.max_joint_violation < .001)
            # Potential-based reward, credited only for physically supported motion.
            progress = float(np.clip((math.degrees(s['pitch']) - 20) / max(goal - 20, 1), 0, 1))
            phi = progress if valid else 0.0
            shaped = 2.0 * (self.cfg.gamma * (0.0 if done else phi) - self.stage_phi)
            self.stage_phi = phi
            supported_income = .5 * self.cfg.dt * float(valid)
            gap_penalty = -.8 * self.cfg.dt * float(self.tracker.confirmed and not s['front_support'])
            drift_penalty = -1.0 * self.cfg.dt * float(self.tracker.confirmed) * min(1.0, max(0.0, (s['front_drift'] - .025) / .025))
            near_goal = abs(math.degrees(s['pitch']) - goal) < 7.5
            speed_penalty = -.25 * self.cfg.dt * float(valid and near_goal) * min(4.0, abs(s['pitch_velocity']) ** 2)
            catch_reward = 2.0 * self.cfg.dt * float(valid and near_goal and self.catch_timer > 0)
            extra = shaped + supported_income + gap_penalty + drift_penalty + speed_penalty + catch_reward
            reward += extra
            self.reward_sum += extra
            self.terms_sum['curriculum_extra'] = self.terms_sum.get('curriculum_extra', 0.0) + extra
            info['reward'] = reward
        info.update(self.metrics())
        return observations, reward, done, info

    def metrics(self):
        info = super().metrics()
        if not hasattr(self, 'curriculum_stage'):
            return info
        stage = self.curriculum_stage
        spec = STAGES[stage]
        valid_native = (not self.nonfoot_collision and not self.self_collision
                        and self.max_speed_ratio <= 1.001
                        and self.max_joint_violation < .001
                        and self.max_torque_ratio <= 1.00001)
        if stage == 0:
            # Certified takeoff is an event; a later fall is recorded separately.
            passed = (info['takeoff'] and info['genuine_airborne_seconds'] >= .02
                      and info['genuine_flight_pitch'] >= math.radians(20))
        else:
            # Stage 1 is a genuine dynamic passage: qualify the state AT crossing.
            # Later failure is recorded but does not retroactively invalidate a
            # supported crossing; recovery is independently required at stage 2+.
            passed = bool(info['takeoff'] and self.crossing_snapshot is not None)
            if stage >= 2:
                # Recovery is an EVENT: validate support/velocity at the catch,
                # not a final frame after the robot may subsequently fall.
                caught = self.catch_snapshot
                max_gap = .40 if stage == 2 else .08
                passed = (passed and caught is not None
                          and caught['duration'] >= spec['hold']
                          and caught['contact_gap_s'] <= max_gap
                          and caught['slip_m'] <= .02
                          and caught['front_drift_m'] <= .04
                          and caught['touchdown_speed_mps'] <= 1.0)
            if stage == 6:
                passed = passed and info['handstand_seconds'] >= 5.0
        info.update(
            curriculum_stage=stage,
            curriculum_name=spec['name'],
            curriculum_goal_degrees=spec['degrees'],
            curriculum_success=bool(passed),
            supported_pitch_degrees=float(math.degrees(self.certified_supported_pitch)),
            front_contact_gap_seconds=float(self.max_front_gap),
            front_contact_slip_m=float(self.contact_slip_distance),
            catch_seconds=float(self.best_catch),
            touchdown_speed_mps=float(self.peak_touchdown_speed_mps),
        )
        return info


def load_source(checkpoint):
    # IMPORTANT: torch.load(weights_only=False) only on the user's own trusted file.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or not {'network', 'curriculum'} <= saved.keys():
        raise ValueError('Expected a morphology RotationRun checkpoint containing network and curriculum')
    source = saved['curriculum']['protocol']
    if not {'ppo', 'environment'} <= source.keys():
        raise ValueError('Checkpoint lacks saved morphology PPO/environment protocol')
    hp = dict(source['ppo'])
    cfg = class_cfg(RotationConfig, source['environment'])
    if not (abs(cfg.action_scale - 2.0) < 1e-6 and abs(cfg.target_filter - 1.0) < 1e-6):
        raise ValueError('Expected original joint target scale/filter 2.0/1.0; inspect checkpoint')
    cfg.rotation_enabled = False  # Original takeoff reward + new staged support reward.
    cfg.terminate_on_goal = False
    net = ActorCritic(44, 74, hp['hidden_dims'], hp['initial_log_std'],
                      log_std_min=hp.get('log_std_min', -3.0))
    net.load_state_dict(saved['network'], strict=True)
    if net.actor[0].in_features != 44 or net.critic[0].in_features != 74 or net.actor[-1].out_features != 12:
        raise ValueError('Policy interface differs from 44/74/12')
    return net, hp, cfg, source


def evaluate(net, assets, cfg, stage, eval_seeds, stochastic=False):
    env = HandstandCurriculumEnvironment(assets, cfg)
    env.set_stage(stage)
    rows = []
    for seed in eval_seeds:
        actor, _ = env.reset(1.0, seed)
        rng = torch.Generator().manual_seed(seed + 1_000_000)
        while True:
            with torch.no_grad():
                distribution = net.distribution(torch.tensor(actor, dtype=torch.float32))
                raw = distribution.loc
                if stochastic:
                    raw = raw + distribution.scale * torch.randn(12, generator=rng)
            (actor, _), _, done, info = env.step(raw.tanh().numpy())
            if done:
                break
        rows.append(dict(
            seed=seed, stochastic=stochastic, stage=stage,
            success=bool(info['curriculum_success']),
            takeoff=bool(info['takeoff']),
            peak_pitch_degrees=float(math.degrees(info['max_pitch'])),
            supported_pitch_degrees=float(info['supported_pitch_degrees']),
            max_front_gap_seconds=float(info['front_contact_gap_seconds']),
            loaded_slip_m=float(info['front_contact_slip_m']),
            front_drift_m=float(info['front_drift']),
            catch_seconds=float(info['catch_seconds']),
            touchdown_speed_mps=float(info['touchdown_speed_mps']),
            handstand_seconds=float(info['handstand_seconds']),
            collision=bool(info['nonfoot_collision'] or info['self_collision']),
            max_speed_ratio=float(info['max_speed_ratio']),
            termination=info['termination'],
        ))
    return rows


def summarize(rows):
    return dict(
        passed=float(np.mean([r['success'] for r in rows])),
        takeoff=float(np.mean([r['takeoff'] for r in rows])),
        supported_pitch_deg=float(np.mean([r['supported_pitch_degrees'] for r in rows])),
        peak_pitch_deg=float(np.mean([r['peak_pitch_degrees'] for r in rows])),
        collisions=float(np.mean([r['collision'] for r in rows])),
        max_front_gap_s=float(max(r['max_front_gap_seconds'] for r in rows)),
        best_handstand_s=float(max(r['handstand_seconds'] for r in rows)),
    )


def ppo_iteration(net, optimizer, hp, envs, actors, critics, training_seed, episode_counts,
                  steps_per_iteration, stage):
    n = len(envs)
    if steps_per_iteration % n:
        raise ValueError('steps_per_iteration must be divisible by num_envs')
    for env in envs:
        env.set_stage(stage)
    chunks = {k: [] for k in ('obs', 'priv', 'raw', 'log', 'value', 'reward', 'done', 'nextval')}
    completed = []
    for _ in range(steps_per_iteration // n):
        with torch.no_grad():
            raw, log, value = net.sample(torch.tensor(actors), torch.tensor(critics))
        for key, val in [('obs', actors.copy()), ('priv', critics.copy()),
                         ('raw', raw.numpy()), ('log', log.numpy()), ('value', value.numpy())]:
            chunks[key].append(val)
        next_actor, next_critic, terminal_critic = [], [], []
        reward_step, done_step = [], []
        for i, env in enumerate(envs):
            (a, c), reward, done, info = env.step(raw[i].tanh().numpy())
            reward_step.append(reward)
            done_step.append(done)
            terminal_critic.append(c)
            if done:
                completed.append(dict(stage=stage, success=bool(info['curriculum_success']),
                                      takeoff=bool(info['takeoff']),
                                      supported_pitch_degrees=info['supported_pitch_degrees']))
                episode_counts[i] += 1
                a, c = env.reset(1.0, event_seed(training_seed, i, episode_counts[i]))
            next_actor.append(a)
            next_critic.append(c)
        with torch.no_grad():
            nextval = net.critic(torch.tensor(np.stack(terminal_critic))).squeeze(-1).numpy()
        chunks['reward'].append(reward_step)
        chunks['done'].append(done_step)
        chunks['nextval'].append(nextval)
        actors, critics = np.stack(next_actor), np.stack(next_critic)
    adv, returns = advantages(np.asarray(chunks['reward'], dtype=np.float32),
                              np.asarray(chunks['value'], dtype=np.float32),
                              np.asarray(chunks['nextval'], dtype=np.float32),
                              np.asarray(chunks['done'], dtype=np.float32),
                              hp['gamma'], hp['gae_lambda'])
    def flat(x):
        arr = np.asarray(x, dtype=np.float32)
        return torch.tensor(arr.reshape((-1,) + arr.shape[2:]))
    batch = (flat(chunks['obs']), flat(chunks['priv']), flat(chunks['raw']),
             flat(chunks['log']), flat(returns), flat(adv))
    stats = update(net, optimizer, batch, hp, demonstrations=None)
    return actors, critics, completed, stats


def run(args):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Use Python 3.11, as required by the verified Molab workflow')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'Missing source checkpoint: {args.checkpoint}')
    if args.max_updates < 0 or args.max_updates > 300:
        raise ValueError('--max-updates must be between 0 and 300')
    if args.num_envs < 1:
        raise ValueError('--num-envs must be positive')
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    net, hp, cfg, source = load_source(args.checkpoint)
    source_hash = sha256(args.checkpoint)
    # New objective: fresh optimizer; original network weights exactly retained.
    lr = min(float(hp['learning_rate']), args.learning_rate_cap)
    hp['learning_rate'] = lr
    hp['bc_weight'] = 0.0  # No unsafe 51-degree BC demonstrations.
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'checkpoints').mkdir()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    metadata = dict(source_checkpoint=str(args.checkpoint.resolve()),
                    source_sha256=source_hash,
                    source_iteration=int(source.get('checkpoint_iteration', -1)),
                    git_commit=commit.stdout.strip() if commit.returncode == 0 else 'unknown',
                    python=sys.version, torch=torch.__version__,
                    cpu_mujoco=True, wandb=False,
                    actor_observations=44, critic_observations=74, actions=12,
                    max_updates=args.max_updates,
                    stage_specs=STAGES,
                    hyperparameters=hp,
                    config=asdict(cfg),
                    notes='Source policies remain unchanged; no imitation of unsupported flight')
    dump_json(out / 'manifest.json', metadata)
    assets = Assets(out / 'assets')
    # Hard observation/action compatibility preflight before any optimizer step.
    probe = HandstandCurriculumEnvironment(assets, cfg)
    a, c = probe.reset(1.0, 70401)
    assert a.shape == (44,) and c.shape == (74,), (a.shape, c.shape)
    assert len(probe.names) == 12
    assert probe.cfg.kp == 60.0 and probe.cfg.kd == 3.0, 'Unexpected gains; validate physics before training'
    with torch.no_grad():
        policy_action = net.actor(torch.tensor(a)).tanh().numpy()
    if not np.isfinite(policy_action).all():
        raise ValueError('Nonfinite source actor output')
    stage = 0
    iteration = 0
    steps = 0
    passing_windows = 0
    last_promotion = 0
    episodes_per_update = args.steps_per_update
    seeds = list(range(70400, 70400 + args.eval_seeds))
    # Preflight evaluation of the *loaded* source policy, before new training.
    before = {}
    for mode in (False, True):
        rows = evaluate(net, assets, cfg, 0, seeds, stochastic=mode)
        append_csv(out / 'evaluation.csv', [dict(iteration=0, **r) for r in rows])
        before['stochastic' if mode else 'deterministic'] = summarize(rows)
    dump_json(out / 'baseline.json', before)
    print('Preflight:', json.dumps(before, indent=2), flush=True)
    if args.preflight or args.max_updates == 0:
        dump_json(out / 'completion.json', dict(reason='preflight_only', ppo_interactions=0,
                                                bc_updates=0, final_stage=0))
        print('No optimizer updates executed. Output:', out)
        return
    if min(before[k]['takeoff'] for k in before) < .75:
        dump_json(out / 'completion.json', dict(reason='source_takeoff_below_75_percent',
                                                ppo_interactions=0, bc_updates=0, final_stage=0))
        print('NO-GO: source takeoff retention below 75%; no PPO performed.')
        return
    envs = [HandstandCurriculumEnvironment(assets, cfg) for _ in range(args.num_envs)]
    episode_counts = [0] * len(envs)
    initial = [env.reset(1.0, event_seed(args.seed, i, 0)) for i, env in enumerate(envs)]
    actors = np.stack([v[0] for v in initial])
    critics = np.stack([v[1] for v in initial])
    prev_gate_rate = 0.0
    reason = 'budget_reached'
    start = time.monotonic()
    for iteration in range(1, args.max_updates + 1):
        actors, critics, completed, stats = ppo_iteration(
            net, optimizer, hp, envs, actors, critics, args.seed,
            episode_counts, episodes_per_update, stage)
        steps += episodes_per_update
        train_summary = dict(iteration=iteration, stage=stage,
                             actual_ppo_interactions=steps, episodes=len(completed),
                             episodes_success=float(np.mean([r['success'] for r in completed])) if completed else None,
                             episodes_takeoff=float(np.mean([r['takeoff'] for r in completed])) if completed else None,
                             post_update_kl=stats['post_update_kl'], clip_fraction=stats['clip_fraction'],
                             learning_rate=lr, wall_seconds=round(time.monotonic() - start, 1))
        append_csv(out / 'training.csv', [train_summary])
        print('Update:', json.dumps(train_summary), flush=True)
        if (not math.isfinite(stats['post_update_kl']) or stats['post_update_kl'] > .03
                or not math.isfinite(stats['clip_fraction']) or stats['clip_fraction'] > .3):
            reason = 'ppo_divergence_stop'
            break
        if iteration % args.eval_every == 0 or iteration == args.max_updates:
            evaluations = {}
            for stochastic in (False, True):
                rows = evaluate(net, assets, cfg, stage, seeds, stochastic=stochastic)
                append_csv(out / 'evaluation.csv', [dict(iteration=iteration, **r) for r in rows])
                evaluations['stochastic' if stochastic else 'deterministic'] = summarize(rows)
            rates = [s['passed'] for s in evaluations.values()]
            retention = min(s['takeoff'] for s in evaluations.values())
            eligible = min(rates) >= STAGES[stage]['pass_rate']
            passing_windows = passing_windows + 1 if eligible else 0
            snapshot = dict(iteration=iteration, stage=stage,
                            curriculum_name=STAGES[stage]['name'],
                            evaluations=evaluations, passing_windows=passing_windows,
                            eligible=eligible)
            dump_json(out / f'probe_{iteration:04d}.json', snapshot)
            ckpt = out / 'checkpoints' / f'policy_update_{iteration:04d}_stage{stage}.pt'
            torch.save(dict(network=net.state_dict(), optimizer=optimizer.state_dict(),
                            iteration=iteration, curriculum_stage=stage, steps=steps,
                            source_sha256=source_hash, hp=hp, seed=args.seed,
                            eval_summary=evaluations), ckpt)
            if stage < len(STAGES) - 1 and passing_windows >= 2:
                stage += 1
                passing_windows = 0
                last_promotion = iteration
                append_csv(out / 'promotions.csv', [dict(iteration=iteration, new_stage=stage,
                                                         curriculum=STAGES[stage]['name'],
                                                         total_ppo_interactions=steps)])
                print('PROMOTED to', stage, STAGES[stage]['name'], flush=True)
                # Begin new level with fresh episodes, preserving trained network.
                initial = [env.reset(1.0, event_seed(args.seed, i, episode_counts[i]))
                           for i, env in enumerate(envs)]
                actors = np.stack([v[0] for v in initial])
                critics = np.stack([v[1] for v in initial])
            if retention < .75:
                reason = 'takeoff_retention_regressed'
                break
            if stage == len(STAGES) - 1 and eligible:
                reason = 'five_second_handstand_gate_passed'
                break
            if iteration >= args.pilot_updates and stage == 0:
                reason = 'pilot_did_not_retain_certified_takeoff'
                break
            if iteration - last_promotion >= args.stage_patience and stage > 0:
                reason = 'stage_plateau_no_support_progression'
                break
    dump_json(out / 'completion.json', dict(reason=reason,
                                            ppo_interactions=steps, bc_updates=0,
                                            final_stage=stage, final_iteration=iteration,
                                            source_checkpoint_sha256=source_hash))
    print('COMPLETE:', reason, 'Stage:', stage, 'Interactions:', steps, 'Directory:', out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='User-trusted physical-morphology RotationRun checkpoint_000036.pt')
    p.add_argument('--output', type=Path,
                   default=Path('output') / ('molab_handstand_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--max-updates', type=int, default=10)
    p.add_argument('--pilot-updates', type=int, default=10)
    p.add_argument('--stage-patience', type=int, default=40)
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--steps-per-update', type=int, default=1024)
    p.add_argument('--eval-seeds', type=int, default=12)
    p.add_argument('--eval-every', type=int, default=2)
    p.add_argument('--learning-rate-cap', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.steps_per_update < a.num_envs or a.steps_per_update % a.num_envs:
        p.error('--steps-per-update must be divisible by --num-envs')
    if not (1 <= a.eval_seeds <= 100 and 1 <= a.eval_every <= 50):
        p.error('Invalid evaluation settings')
    run(a)


if __name__ == '__main__':
    main()
