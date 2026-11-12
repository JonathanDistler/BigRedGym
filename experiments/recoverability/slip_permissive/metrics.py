"""Two observers on ONE physical rollout. No physics or strict code is changed."""
import copy
import math
import numpy as np
import mujoco
from experiments.recoverability.configuration_planning.dynamics import ValidationMetrics
from experiments.recoverability.transition import snapshot,SPEC
from experiments.recoverability.physics import World
from experiments.recoverability.controllers import BalanceAdapter


class DualMetrics(ValidationMetrics):
    def __init__(self,w):
        super().__init__(w)
        self.exploratory_hold=self.exploratory_best_hold=0.
        self.crossings={str(x):None for x in (45,60,75)};self.arrival=None;self.takeoff_time=None
        self.rear_unloaded_time=0.;self.entry=None;self.balance_started=None
        self.strict_balance_best=0.;self.exploratory_balance_best=0.
        self.max_safe_pitch=0.;self.max_supported_safe_pitch=0.;self.last_sample=None
        self.strict_success_time=None;self.exploratory_success_time=None

    def observe(self,row,dt):
        s=super().observe(row,dt);w=self.w
        rear_unloaded=(row['rear_clearance']>.015 and row['loads'][2:].sum()<.03*w.weight and s['pitch']>7)
        self.rear_unloaded_time=self.rear_unloaded_time+dt if rear_unloaded else 0.
        if self.takeoff_time is None and self.rear_unloaded_time>=.02-1e-9:
            self.takeoff_time=float(w.d.time);self.active=True
        if not self.failed:
            self.max_safe_pitch=max(self.max_safe_pitch,s['pitch'])
            if row['front_support']:self.max_supported_safe_pitch=max(self.max_supported_safe_pitch,s['pitch'])
            for angle in self.crossings:
                if self.crossings[angle] is None and s['pitch']>=float(angle):
                    self.crossings[angle]=dict(time=float(w.d.time),supported=bool(row['front_support']),slip=self.slip,drift=self.drift)
            if self.arrival is None and 85<=s['pitch']<=95:
                self.arrival=dict(time=float(w.d.time),supported=bool(row['front_support']),omega=s['omega'],slip=self.slip,drift=self.drift)
        # Same original posture, COM-speed, angular-speed, rear-clearance/load,
        # bilateral front loading and hard safety conditions. Only slip/drift
        # restrictions are absent from the exploratory hold predicate.
        stable=(row['front_support'] and not self.failed and s['tilt']<=10
                and np.linalg.norm(w.d.qvel[3:6])<=1.5 and np.linalg.norm(row['com_velocity'])<=.35
                and row['rear_clearance']>.10 and row['loads'][2:].sum()<.03*w.weight)
        self.exploratory_hold=self.exploratory_hold+dt if stable else 0.
        self.exploratory_best_hold=max(self.exploratory_best_hold,self.exploratory_hold)
        if self.balance_started is not None:
            self.strict_balance_best=max(self.strict_balance_best,self.hold)
            self.exploratory_balance_best=max(self.exploratory_balance_best,self.exploratory_hold)
        g=self.gates()
        if g['strict_success'] and self.strict_success_time is None:self.strict_success_time=float(w.d.time)
        if g['exploratory_success'] and self.exploratory_success_time is None:self.exploratory_success_time=float(w.d.time)
        s.update(body_height=float(w.d.qpos[2]),front_FL_x=float(row['feet'][0,0]),front_FR_x=float(row['feet'][1,0]),exploratory_hold=self.exploratory_hold,strict_hold=self.hold,physical_takeoff=int(self.takeoff_time is not None))
        # Parent stored a copy before these observer-only fields were added.
        if self.trace and self.trace[-1]['time']==s['time']:self.trace[-1].update(s)
        self.last_sample=s.copy();return s

    def begin_balance(self):
        self.balance_started=float(self.w.d.time);self.phase='balance'
        self.entry=dict(slip=self.slip,drift=self.drift,longest_gap=self.longest_gap,total_gap=self.total_gap,
                        impact=self.impact,hard_safe=not self.failed,head=self.head_min,calf=self.calf_min)
        self.hold=self.best_hold=0.;self.exploratory_hold=self.exploratory_best_hold=0.

    def gates(self):
        hard_safe=not self.failed and self.max_speed<=1.001 and self.max_torque<=1.00001 and self.max_joint<.001
        entry=self.entry or dict(slip=self.slip,drift=self.drift,longest_gap=self.longest_gap,total_gap=self.total_gap,
                                impact=self.impact,hard_safe=not self.failed,head=self.head_min,calf=self.calf_min)
        shared_entry=(entry['hard_safe'] and entry['longest_gap']<=.08+1e-9 and entry['total_gap']<=.12+1e-9
                      and entry['impact']<=.75 and entry['head']>=.005 and entry['calf']>=.005)
        strict_entry=shared_entry and entry['slip']<=.02 and entry['drift']<=.04
        strict_success=hard_safe and strict_entry and self.strict_balance_best>=5.-1e-8
        exploratory_success=hard_safe and shared_entry and self.exploratory_balance_best>=5.-1e-8
        return dict(strict_success=bool(strict_success or self.strict_success_time is not None),
                    exploratory_success=bool(exploratory_success or self.exploratory_success_time is not None),
                    strict_entry=bool(strict_entry),exploratory_entry=bool(shared_entry),hard_safe=bool(hard_safe))

    def results(self):
        return dict(**self.outcome(),**self.gates(),strict_hold_seconds=self.strict_balance_best,
                    exploratory_hold_seconds=self.exploratory_balance_best,
                    safe_peak=self.max_safe_pitch,safe_supported_peak=self.max_supported_safe_pitch,
                    physical_takeoff_time=self.takeoff_time,legacy_takeoff=bool(self.w.tracker.confirmed),
                    crossings=self.crossings,near_upright_arrival=self.arrival,balance_handoff_time=self.balance_started,
                    entry_metrics=self.entry,strict_success_time=self.strict_success_time,exploratory_success_time=self.exploratory_success_time,final=self.last_sample)


def clone(w,m,c,a=None):
    # Clone numerical integration state and explicit controller memory for a
    # scratch counterfactual; never project/reset the live floating base.
    trace,states=m.trace,m.states;m.trace=[];m.states=[]
    try:saved=snapshot(w,m,a)
    finally:m.trace=trace;m.states=states
    scratch=World(w.assets,w.cfg)
    mujoco.mj_setState(scratch.m,scratch.d,saved['state'],SPEC);mujoco.mj_forward(scratch.m,scratch.d)
    scratch.target=saved['target'].copy();scratch.tracker=copy.deepcopy(w.tracker);new=DualMetrics(scratch)
    for key,value in saved['metrics'].items():
        if key!='w':setattr(new,key,copy.deepcopy(value))
    new.trace=[];new.states=[];scratch.monitor=new
    adapter=BalanceAdapter(c,scratch)
    for key,value in saved.get('adapter',{}).items():setattr(adapter,key,copy.deepcopy(value))
    return scratch,new,adapter


def balance(w,m,a,seed,stochastic=False,seconds=6.5):
    if m.balance_started is None:m.begin_balance()
    generator=__import__('torch').Generator().manual_seed(seed)
    for tick in range(round(seconds/.002)):
        if tick%5==0:target=a.target(stochastic=stochastic,generator=generator)
        row=w.tick(target);a.advance_memory(row['loads'],.002);w.measure()
        if m.failed or (m.exploratory_balance_best>=5.-1e-8 and m.strict_balance_best>=5.-1e-8):break
    result=m.results();result['balance_policy_stabilization']=bool(m.exploratory_balance_best>=.5)
    return result
