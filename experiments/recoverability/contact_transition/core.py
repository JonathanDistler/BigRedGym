"""Full-dynamics action splines and independently gated transition measurements."""
from dataclasses import dataclass,asdict
import numpy as np
import mujoco
import torch
from experiments.recoverability.physics import World,Metrics
from experiments.recoverability.bidirectional.core import telemetry,compact_snapshot

TIMES=np.array([.15,.35,.55,.75,.95,1.15,1.4])
# Front thigh/calf, rear thigh/calf, front yielding, soft anchor, hip spread.
LOW=np.array([-.6,-.6,-.7,-.7,.2,0.,-.12])
HIGH=np.array([.6,.6,.7,.7,1.3,1.,.12])
NEUTRAL=np.array([0.,0.,0.,0.,1.,0.,0.])


@dataclass(frozen=True)
class Limits:
    horizon:float=1.6
    target_low:float=45.
    target_high:float=55.
    goal_hold:float=.10
    slip:float=.02
    drift:float=.04
    calf_clearance:float=.005
    transient_gap:float=.08
    transient_total_gap:float=.12
    touchdown_speed:float=.75
    requested_torque:float=1.5
    com_target:float=-.16
    com_progress:float=.02


def neutral():return np.tile(NEUTRAL,(len(TIMES),1))


def commands_at(parameters,time):
    parameters=np.asarray(parameters)
    if parameters.shape!=(len(TIMES),len(NEUTRAL)) or not np.isfinite(parameters).all():
        raise ValueError('Expected a finite 7x7 motor-spline parameter array')
    p=np.clip(parameters,LOW,HIGH)
    return np.array([np.interp(time,np.r_[0.,TIMES],np.r_[NEUTRAL[j],p[:,j]]) for j in range(len(NEUTRAL))])


def modify(w,action,parameters,time,strategy):
    """Only motor targets change; native torque/rate limits remain in World.step."""
    u=commands_at(parameters,time)
    desired=np.clip(w.nominal+w.cfg.action_scale*action,*w.bounds.T)
    desired[[1,4]]+=u[0];desired[[2,5]]+=u[1]
    desired[[7,10]]+=u[2];desired[[8,11]]+=u[3]
    desired[[0,6]]+=u[6];desired[[3,9]]-=u[6]
    q=w.d.qpos[w.qids]
    loads=w.row['loads'][:2]
    for side in range(2):
        ids=np.arange(side*3,side*3+3)
        gain=u[4]
        if strategy=='load_yield' and loads[side]<.05*w.weight:gain=1.
        if strategy=='no_yield':gain=1.
        desired[ids]=q[ids]+gain*(desired[ids]-q[ids])
    if strategy!='no_anchor' and u[5]>0:
        front=np.arange(6);jac=[]
        for g in w.foot_geoms[:2]:
            jp=np.zeros((3,w.m.nv));mujoco.mj_jacGeom(w.m,w.d,jp,None,int(g))
            jac.append(jp[:2,w.vids[front]])
        error=(w.anchor-w.d.geom_xpos[w.foot_geoms[:2],:2]).flatten()
        dq=np.linalg.lstsq(np.vstack(jac),error,rcond=1e-4)[0]
        anchor=q[:6]+np.clip(dq,-.12,.12)
        desired[:6]=(1-u[5])*desired[:6]+u[5]*anchor
    return np.clip((np.clip(desired,*w.bounds.T)-w.nominal)/w.cfg.action_scale,-1,1)


class Monitor(Metrics):
    def __init__(self,w,mode,limits,trace=False):
        super().__init__(w)
        self.require_takeoff=True
        self.mode=mode;self.limits=limits;self.record=trace;self.trace=[];self.count=0
        self.goal_timer=0.;self.goal_best=0.;self.best_state=None
        self.cross30=None;self.progress_peak=0.;self.progress_com=-1.
        self.support_gap=0.;self.total_gap=0.;self.worst_gap=0.
        self.max_touchdown=0.;self.previous_supported=False;self.max_requested=0.
        self.min_transition_calf=1.;self.final=None;self.cross45=None;self.saturated_steps=0
        self.launch_time=0.;self.actions=[];self.start_snapshot=None

    def observe(self,row,dt):
        s=super().observe(row,dt);self.count+=1
        w=self.w;c=self.limits
        requested=np.max(abs(w.cfg.kp*(w.target-w.d.qpos[w.qids])-w.cfg.kd*w.d.qvel[w.vids])/w.effort)
        self.max_requested=max(self.max_requested,float(requested));self.saturated_steps+=int(requested>1.)
        supported=bool(row['front_support'])
        took_off=bool(w.tracker.confirmed)
        vertical_speed=0.
        if supported and not self.previous_supported:
            for g in w.foot_geoms[:2]:
                jp=np.zeros((3,w.m.nv));mujoco.mj_jacGeom(w.m,w.d,jp,None,int(g))
                vertical_speed=max(vertical_speed,float(abs((jp@w.d.qvel)[2])))
            if took_off:self.max_touchdown=max(self.max_touchdown,vertical_speed)
        self.previous_supported=supported
        if took_off:
            self.support_gap=0. if supported else self.support_gap+dt
            self.total_gap+=dt*int(not supported);self.worst_gap=max(self.worst_gap,self.support_gap)
        if self.cross30 is None and s['pitch']>=30 and took_off and not self.failed:
            self.cross30=s.copy()
        if self.cross30 is not None:
            calf=min(s['FL_calf_clearance'],s['FR_calf_clearance'])
            self.min_transition_calf=min(self.min_transition_calf,calf)
            if not self.failed and supported:
                self.progress_peak=max(self.progress_peak,s['pitch'])
                if s['pitch']>=35:self.progress_com=max(self.progress_com,s['com_support_x'])
            mode_ok=(self.worst_gap<=.004 if self.mode=='maintained' else
                     self.worst_gap<=c.transient_gap and self.total_gap<=c.transient_total_gap and self.max_touchdown<=c.touchdown_speed)
            goal=(not self.failed and supported and took_off and mode_ok and c.target_low<=s['pitch']<=c.target_high
                  and abs(s['omega'])<=2 and abs(s['roll_rate'])<=1 and np.linalg.norm(row['com_velocity'])<=.75
                  and row['rear_clearance']>.015 and row['loads'][2:].sum()<.03*w.weight
                  and s['com_support_x']>=c.com_target and s['com_support_x']>=self.cross30['com_support_x']+c.com_progress
                  and self.slip<=c.slip and self.drift<=c.drift and calf>=c.calf_clearance)
            self.goal_timer=self.goal_timer+dt if goal else 0.
            self.goal_best=max(self.goal_best,self.goal_timer)
            if self.cross45 is None and s['pitch']>=45 and not self.failed:self.cross45=s.copy()
            if goal and self.goal_timer>=self.goal_best:self.best_state=s.copy()
        s.update(time_from_launch=float(w.d.time)-self.launch_time,front_support=int(supported),takeoff=int(took_off),
                 rear_clearance=row['rear_clearance'],front_gap=self.support_gap,total_front_gap=self.total_gap,
                 goal_hold=self.goal_timer,requested_torque_ratio=float(requested),touchdown_speed=vertical_speed)
        if self.record and (self.count%5==0 or self.failed):
            s.update(telemetry(w))
            for j in range(12):
                s[f'target{j}']=float(w.target[j]);s[f'torque{j}']=float(w.d.qfrc_applied[w.vids[j]])
            self.trace.append(s.copy())
        self.final=s.copy();return s

    def result_transition(self):
        c=self.limits
        mode_ok=(self.worst_gap<=.004 if self.mode=='maintained' else self.worst_gap<=c.transient_gap and self.total_gap<=c.transient_total_gap and self.max_touchdown<=c.touchdown_speed)
        physical=(not self.failed and self.max_speed<=1.001 and self.max_torque<=1.00001 and self.max_joint<.001)
        accept=(physical and self.cross30 is not None and self.goal_best>=c.goal_hold-1e-8
                and self.slip<=c.slip and self.drift<=c.drift and self.min_transition_calf>=c.calf_clearance
                and self.max_requested<=c.requested_torque and mode_ok and self.final['pitch']>=c.target_low
                and self.final['pitch']<=c.target_high and self.final['front_support'])
        return dict(**super().result(),accepted=bool(accept),mode=self.mode,physical_safe=bool(physical),
                    takeoff=bool(self.w.tracker.confirmed),goal_hold=self.goal_best,
                    progress_peak=self.progress_peak,progress_com=self.progress_com,
                    worst_front_gap=self.worst_gap,total_front_gap=self.total_gap,max_touchdown_speed=self.max_touchdown,
                    min_transition_calf=self.min_transition_calf,max_requested_torque_ratio=self.max_requested,
                    saturation_fraction=self.saturated_steps/max(self.count,1),cross30=self.cross30,cross45=self.cross45,
                    best_goal_state=self.best_state,final=self.final,physics_steps=self.count)


def initialize(assets,cfg,scale,seed,mode,limits,trace=False,perturbation=.012):
    w=World(assets,cfg);w.reset(scale,seed)
    m=Monitor(w,mode,limits,trace);w.monitor=m
    # Physically integrate standing reset perturbations; preserve prep slip/history.
    rng=np.random.default_rng(seed);target=w.nominal+rng.uniform(-perturbation,perturbation,12)
    for _ in range(20):
        row=w.tick(target);m.observe(row,w.m.opt.timestep)
        if m.failed:break
    w.row=w.physical_row();w.target=target.copy();m.launch_time=float(w.d.time)
    for sample in m.trace:
        sample['time_from_launch']=sample['time']-m.launch_time
    actor,_=w.observations()
    return w,m,actor


def rollout(net,assets,cfg,parameters,seed,mode='maintained',strategy='load_yield',scale=1.,stochastic=False,
            limits=None,trace=False,perturbation=.012):
    limits=limits or Limits()
    w,m,actor=initialize(assets,cfg,scale,seed,mode,limits,trace,perturbation)
    rng=torch.Generator().manual_seed(seed+9000000)
    for control in range(round(limits.horizon/cfg.dt)):
        if m.failed:break
        with torch.no_grad():
            distribution=net.distribution(torch.tensor(actor));raw=distribution.loc
            if stochastic:raw=raw+distribution.scale*torch.randn(12,generator=rng)
        command=modify(w,raw.tanh().numpy(),parameters,float(w.d.time)-m.launch_time,strategy)
        observation=actor.copy()
        (actor,_),_,done,_=w.step(command)
        if trace:m.actions.append(dict(control=control,time=float(w.d.time)-m.launch_time,
            observation=observation.tolist(),action=command.tolist(),target=w.target.tolist(),
            qpos=w.d.qpos.tolist(),qvel=w.d.qvel.tolist()))
        if m.failed:break
    result=m.result_transition()
    # Finite-horizon score has no fictitious balance handoff or reset interpolation.
    crossing=m.cross30 is not None
    supported=m.progress_peak if crossing else min(m.supported_peak*180/np.pi,30.)
    com=m.progress_com if m.progress_com>-1 else m.final['com_support_x']
    score=2*min(supported,55)+.8*np.clip(m.final['pitch'],0,55)+120*(com+.20)+150*m.goal_best
    score-=350*m.slip+150*m.drift+60*m.worst_gap+15*max(0.,m.max_requested-1.)
    score-=50*int(m.failed)+25*int(not w.tracker.confirmed)
    score-=30*max(0.,m.total_gap-.12)+10*max(0.,abs(m.final['omega'])-2)
    score+=20*int(result['physical_safe'])+200*int(result['accepted'])
    result['score']=float(score)
    return result,m.trace,m.actions
