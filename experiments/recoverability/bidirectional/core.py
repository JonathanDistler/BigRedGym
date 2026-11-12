"""Physical reset proposals, action-integrated perturbations and contact diagnostics."""
import copy
import hashlib
import math
import mujoco
import numpy as np
import torch
from experiments.recoverability.physics import World, Metrics, feasible_state
from experiments.recoverability.controllers import BalanceAdapter
from experiments.recoverability.transition import snapshot, restore

BANDS = (90, 85, 80, 75, 70)
CONDITIONS = ('nominal', 'pitch_minus', 'pitch_plus', 'roll_minus', 'roll_plus',
              'contact_left', 'contact_right', 'rear_extend', 'rear_tuck', 'com_minus', 'com_plus')


def physical_hash(w):
    """Time-independent state identity; 1e-9 quantization, same-model only."""
    h = hashlib.sha256()
    h.update(str(w.scale).encode())
    h.update(np.round(np.r_[w.d.qpos, w.d.qvel, w.d.act], 9).tobytes())
    return h.hexdigest()


def telemetry(w):
    normal = np.zeros(4); tangent = np.zeros(4); utilization = np.zeros(4)
    force = np.zeros(6)
    for i, contact in enumerate(w.d.contact):
        if contact.dist > 0:
            continue
        bodies = w.m.geom_bodyid[[contact.geom1, contact.geom2]]
        if bool(bodies[0]) == bool(bodies[1]):
            continue
        foot = w.foot_lookup.get(int(max(bodies)), 4)
        if foot >= 4:
            continue
        mujoco.mj_contactForce(w.m, w.d, i, force)
        fn = max(0., float(force[0])); ft = float(np.linalg.norm(force[1:3]))
        normal[foot] += fn; tangent[foot] += ft
        if fn > .05*w.weight:
            utilization[foot] = max(utilization[foot], ft/max(contact.friction[0]*fn, 1e-12))
    # All terms use world-frame velocities at each body's own COM.
    com = w.d.subtree_com[w.base]
    angular = np.zeros(3); rear = np.zeros(3)
    jp = np.zeros((3,w.m.nv)); jr = np.zeros_like(jp)
    for body in range(1,w.m.nbody):
        mujoco.mj_jacBodyCom(w.m,w.d,jp,jr,body)
        v = jp@w.d.qvel; omega = jr@w.d.qvel
        rot = w.d.ximat[body].reshape(3,3)
        h = rot@np.diag(w.m.body_inertia[body])@rot.T@omega
        h += np.cross(w.d.xipos[body]-com, w.m.body_mass[body]*v)
        angular += h
        if w.m.body(body).name.startswith(('RL_', 'RR_')):
            rear += h
    raw = w.cfg.kp*(w.target-w.d.qpos[w.qids])-w.cfg.kd*w.d.qvel[w.vids]
    out = dict(pitch_angular_momentum=float(angular[1]), rear_pitch_momentum=float(rear[1]),
               requested_torque_ratio=float(np.max(abs(raw)/w.effort)),
               actual_torque_ratio=float(np.max(abs(w.d.qfrc_applied[w.vids])/w.effort)))
    for j,leg in enumerate(('FL','FR','RL','RR')):
        out.update({f'{leg}_normal':float(normal[j]),f'{leg}_tangent':float(tangent[j]),
                    f'{leg}_friction_utilization':float(utilization[j])})
    return out


class ContactMetrics(Metrics):
    """Original safety and five-second success conditions, with extra telemetry."""
    def __init__(self,w):
        super().__init__(w)
        self.trace=[]; self.physics_steps=0
        self.supported_steps=0; self.contact_loss_time=0.
        self.min_calf=1.; self.max_friction=0.; self.max_requested_torque=0.

    def observe(self,row,dt):
        sample=super().observe(row,dt)
        self.physics_steps+=int(dt>0)
        self.supported_steps+=int(row['front_support'] and dt>0)
        self.contact_loss_time+=dt*int(not row['front_support'])
        self.min_calf=min(self.min_calf,sample['FL_calf_clearance'],sample['FR_calf_clearance'])
        # Safety and loaded-slip integration are 500 Hz; force/momentum logs 100 Hz.
        if self.physics_steps%5==0 or dt==0 or self.failed:
            sample.update(telemetry(self.w)); self.trace.append(sample.copy())
            self.max_friction=max(self.max_friction,sample['FL_friction_utilization'],sample['FR_friction_utilization'])
            self.max_requested_torque=max(self.max_requested_torque,sample['requested_torque_ratio'])
        return sample

    def result(self):
        return dict(**super().result(),physics_steps=self.physics_steps,
                    support_fraction=self.supported_steps/max(1,self.physics_steps),
                    contact_loss_seconds=self.contact_loss_time,min_calf_clearance=self.min_calf,
                    max_friction_utilization=self.max_friction,max_requested_torque_ratio=self.max_requested_torque)


def motor_perturbation(condition,amplitude):
    delta=np.zeros(12)
    sign=-1 if condition.endswith('minus') else 1
    if condition.startswith('pitch_'):
        delta[[1,4]]=sign*amplitude;delta[[7,10]]=-sign*amplitude
    elif condition.startswith('roll_'):
        delta[[0,3]]=sign*amplitude
    elif condition.startswith('contact_'):
        index=1 if condition=='contact_left' else 4
        delta[index]=amplitude;delta[index+1]=-amplitude
    elif condition.startswith('rear_'):
        sign=1 if condition=='rear_tuck' else -1
        delta[[7,10]]=-sign*amplitude;delta[[8,11]]=sign*2*amplitude
    elif condition.startswith('com_'):
        delta[[1,4]]=sign*amplitude;delta[[2,5]]=-sign*amplitude
    return delta


def compact_snapshot(w,m,a=None):
    """Preserve safety history without copying the entire telemetry trajectory."""
    trace=getattr(m,'trace',None)
    if trace is not None:m.trace=[]
    try:return snapshot(w,m,a)
    finally:
        if trace is not None:m.trace=trace


def prepare(c,assets,cfg,pitch,condition,seed,split='train',scale=1.):
    """IK defines a valid reset proposal, NOT evidence of standing reachability.

    Pitch, COM velocity, roll and contacts after preparation come from motors and
    full dynamics. No velocity, external force, friction or contact edits.
    """
    w=World(assets,cfg)
    accepted,screen=feasible_state(w,c.env._standing_posture()[0].numpy(),seed,pitch=pitch,scale=scale)
    m=ContactMetrics(w); a=BalanceAdapter(c,w)
    if not accepted:return w,m,a,dict(valid=False,ik=screen,origin='IK proposal'),None
    rng=np.random.default_rng(seed)
    amplitude=(.025 if split=='train' else .04)*rng.uniform(.9,1.1)
    duration=(.04 if split=='train' else .06)
    delta=motor_perturbation(condition,amplitude)
    for tick in range(round(duration/w.m.opt.timestep)):
        if tick%5==0:
            target=a.target()+delta
            a.previous=torch.tensor(target,dtype=torch.float32)-a.reference
        row=w.tick(target);a.advance_memory(row['loads'],w.m.opt.timestep)
        sample=m.observe(row,w.m.opt.timestep)
        if m.failed:break
    valid=bool(w.screen()[0] and not m.failed)
    metadata=dict(valid=valid,ik=screen,origin='IK proposal followed by integrated motor perturbation',
                  standing_reachable=False,amplitude_rad=amplitude,duration_seconds=duration,
                  actual_pitch=sample['pitch'],actual_omega=sample['omega'],actual_roll_rate=sample['roll_rate'],
                  actual_com_support_x=sample['com_support_x'],actual_com_vx=sample['com_vx'],
                  front_support=bool(row['front_support']),state_id=physical_hash(w))
    return w,m,a,metadata,compact_snapshot(w,m,a) if valid else None


def continuation(c,w,m,a,stochastic,seed,seconds=6.5,actor=None):
    generator=torch.Generator().manual_seed(seed)
    start=float(w.d.time);start_steps=m.physics_steps;m.hold=m.best_hold=0.
    m.observe(w.physical_row(),0.)
    for tick in range(round(seconds/w.m.opt.timestep)):
        if tick%5==0:
            if actor is None:target=a.target(stochastic=stochastic,generator=generator)
            else:
                with torch.no_grad():
                    raw=actor(a.observation())[0]
                    if stochastic:raw=raw+c.actor.std*torch.randn(12,generator=generator)
                    a.previous=raw*torch.as_tensor(c.env.scales['dof_pos_target'])
                    target=(a.reference+a.previous).numpy().astype(float)
        row=w.tick(target);a.advance_memory(row['loads'],w.m.opt.timestep)
        m.observe(row,w.m.opt.timestep)
        if m.failed or m.best_hold>=5:break
    return dict(**m.result(),continuation_seconds=float(w.d.time)-start,
                continuation_physics_steps=m.physics_steps-start_steps)


def restore_contact(c,assets,cfg,saved):
    w,old,a=restore(assets,cfg,saved,c)
    m=ContactMetrics(w)
    for key,value in vars(old).items():
        if key not in ('w','trace'):setattr(m,key,copy.deepcopy(value))
    return w,m,a
