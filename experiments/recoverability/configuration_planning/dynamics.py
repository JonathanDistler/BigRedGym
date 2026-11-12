"""Exact native-motor simulation, state capture, and physical validation metrics."""
import numpy as np
import torch
import mujoco
from experiments.recoverability.physics import World,Metrics
from experiments.recoverability.bidirectional.core import telemetry

SPEC=mujoco.mjtState.mjSTATE_INTEGRATION


def integration(w):
    state=np.empty(mujoco.mj_stateSize(w.m,SPEC));mujoco.mj_getState(w.m,w.d,state,SPEC)
    return dict(integration=state,qpos=w.d.qpos.copy(),qvel=w.d.qvel.copy(),target=w.target.copy(),time=float(w.d.time))


def put(w,state):
    mujoco.mj_setState(w.m,w.d,state['integration'],SPEC);mujoco.mj_forward(w.m,w.d);w.target=state['target'].copy()


class ValidationMetrics(Metrics):
    def __init__(self,w):
        super().__init__(w);self.trace=[];self.states=[];self.steps=0;self.phase='kick';self.previous_vertical=np.zeros(2)
        self.last_loaded=np.zeros(2,dtype=bool);self.impact=0.;self.gap_run=0.;self.longest_gap=0.;self.total_gap=0.
        self.last_touching=np.zeros(2,dtype=bool);self.flight_run=0.;self.flight_max=0.;self.flight_total=0.
        self.head_min=1.;self.calf_min=1.;self.nonfoot_min=1.;self.contact_modes=[];self.last_mode=None
        self.active=False;self.stage45=False;self.max_requested=0.
        self.floor=np.flatnonzero(w.m.geom_bodyid==0)
        self.groups={name:[g for g in range(w.m.ngeom) if w.m.geom_bodyid[g] and w.m.geom_contype[g] and w.m.body(w.m.geom_bodyid[g]).name.startswith(prefix)]
                     for name,prefix in [('head',('Head',)),('calf',('FL_calf','FR_calf'))]}

    def observe(self,row,dt):
        sample=super().observe(row,dt);self.steps+=1;w=self.w
        load=row['loads'][:2]>.05*w.weight;vertical=np.zeros(2)
        touching=np.zeros(2,dtype=bool)
        for contact in w.d.contact:
            if contact.dist>0:continue
            b1,b2=w.m.geom_bodyid[[contact.geom1,contact.geom2]]
            if bool(b1)==bool(b2):continue
            foot=w.foot_lookup.get(int(b2 if b1==0 else b1),4)
            if foot<2:touching[foot]=True
        for i,g in enumerate(w.foot_geoms[:2]):
            jp=np.zeros((3,w.m.nv));mujoco.mj_jacGeom(w.m,w.d,jp,None,int(g));vertical[i]=(jp@w.d.qvel)[2]
            if self.active and ((load[i] and not self.last_loaded[i]) or (touching[i] and not self.last_touching[i])):
                self.impact=max(self.impact,abs(self.previous_vertical[i]),abs(vertical[i]))
        self.previous_vertical=vertical;self.last_loaded=load;self.last_touching=touching
        mode='both' if load.all() else 'FL' if load[0] else 'FR' if load[1] else 'unloaded'
        if mode!=self.last_mode:self.contact_modes.append(dict(time=float(w.d.time),mode=mode,phase=self.phase));self.last_mode=mode
        if self.active:
            self.gap_run=self.gap_run+dt if not load.all() else 0.;self.longest_gap=max(self.longest_gap,self.gap_run)
            self.total_gap+=dt*int(not load.all())
            self.flight_run=self.flight_run+dt if not touching.any() else 0.
            self.flight_max=max(self.flight_max,self.flight_run);self.flight_total+=dt*int(not touching.any())
        clear={}
        for name,geoms in self.groups.items():
            clear[name]=min((mujoco.mj_geomDistance(w.m,w.d,int(g),int(f),1.,None) for g in geoms for f in self.floor),default=1.)
        self.head_min=min(self.head_min,clear['head']);self.calf_min=min(self.calf_min,clear['calf'])
        requested=float(np.max(abs(w.cfg.kp*(w.target-w.d.qpos[w.qids])-w.cfg.kd*w.d.qvel[w.vids])/w.effort));self.max_requested=max(self.max_requested,requested)
        if self.active and not self.failed and sample['pitch']>=45 and load.all():self.stage45=True
        sample.update(head_clearance=clear['head'],all_front_calf_clearance=clear['calf'],phase=self.phase,
            contact_mode=mode,landing_vertical_velocity=abs(vertical).max(),max_prepost_impact_velocity=self.impact,
            geometric_contact_FL=int(touching[0]),geometric_contact_FR=int(touching[1]),geometric_flight_gap=self.flight_run,
            contact_gap=self.gap_run,front_total_gap=self.total_gap,requested_torque_ratio=requested,
            rear_clearance=row['rear_clearance'],load_FL=row['loads'][0],load_FR=row['loads'][1])
        if self.steps%5==0 or self.failed:
            sample.update(telemetry(w))
            for j in range(12):
                sample[f'torque{j}']=float(w.d.qfrc_applied[w.vids[j]])
                sample[f'target{j}']=float(w.target[j])
            self.trace.append(sample.copy())
            self.states.append(dict(time=float(w.d.time),qpos=w.d.qpos.copy(),qvel=w.d.qvel.copy(),target=w.target.copy()))
        return sample

    def outcome(self):
        return dict(**self.result(),stage45=self.stage45,head_clearance=self.head_min,all_calf_clearance=self.calf_min,
                    longest_contact_gap=self.longest_gap,total_contact_gap=self.total_gap,landing_impact_velocity=self.impact,
                    geometric_flight_total=self.flight_total,geometric_flight_max=self.flight_max,
                    contact_modes=self.contact_modes,max_requested_torque_ratio=self.max_requested,physics_steps=self.steps)


def capture(net,assets,cfg,seed,kind='30',perturb=.012,stochastic=False):
    w=World(assets,cfg);actor,_=w.reset(1.,seed);m=ValidationMetrics(w);w.monitor=m
    rng=np.random.default_rng(seed);target=w.nominal+rng.uniform(-perturb,perturb,12)
    # Real motor-integrated preparation; do not edit velocities or contact state.
    for _ in range(20):
        row=w.tick(target);m.observe(row,.002)
    w.row=w.physical_row();actor,_=w.observations();generator=torch.Generator().manual_seed(seed+300000)
    for _ in range(100):
        with torch.no_grad():
            dist=net.distribution(torch.tensor(actor));raw=dist.loc
            if stochastic:raw=raw+dist.scale*torch.randn(12,generator=generator)
        (actor,_),_,done,_=w.step(raw.tanh().numpy())
        pitch=np.rad2deg(w.row['pitch'])
        reached=(float(w.d.time)>=.34 if kind=='launch' else pitch>=30 and w.row['front_support'])
        if m.failed:return w,m,None
        if reached:return w,m,integration(w)
        if done:break
    return w,m,None


def segment(w,start,command,duration,monitor=None,quadrature=False):
    """PD-target ramp with the original 50 Hz target slew and 500 Hz physics."""
    put(w,start);initial_target=start['target'].copy();steps=round(duration/.002)
    path=dict(slip=0.,friction_exposure=0.,head=1.,calf=1.,bilateral_deficit=0.,unloaded_time=0.)
    if quadrature:
        floor=np.flatnonzero(w.m.geom_bodyid==0)
        groups={name:[g for g in range(w.m.ngeom) if w.m.geom_bodyid[g] and w.m.geom_contype[g] and w.m.body(w.m.geom_bodyid[g]).name.startswith(prefix)]
            for name,prefix in [('head',('Head',)),('calf',('FL_calf','FR_calf'))]}
    for tick in range(steps):
        if tick%10==0:
            desired=initial_target+min(1.,(tick+10)/steps)*(command-initial_target)
            target=w.target+np.clip(desired-w.target,-.7*w.speed*w.cfg.dt,.7*w.speed*w.cfg.dt)
        if monitor is None:
            w.target=target.copy()
            raw=w.cfg.kp*(target-w.d.qpos[w.qids])-w.cfg.kd*w.d.qvel[w.vids]
            torque=np.clip(raw,-w.effort,w.effort)
            torque[(abs(w.d.qvel[w.vids])>=w.speed)&(torque*w.d.qvel[w.vids]>0)]=0
            w.d.qfrc_applied[:]=0;w.d.qfrc_applied[w.vids]=torque
            mujoco.mj_step(w.m,w.d);mujoco.mj_forward(w.m,w.d)
        else:row=w.tick(target)
        if quadrature and tick%5==4:
            normal=np.zeros(2);slip=0.;util=0.;force=np.zeros(6)
            for ci,contact in enumerate(w.d.contact):
                if contact.dist>0:continue
                b1,b2=w.m.geom_bodyid[[contact.geom1,contact.geom2]]
                if bool(b1)==bool(b2):continue
                body=int(b2 if b1==0 else b1);foot=w.foot_lookup.get(body,4)
                if foot>=2:continue
                mujoco.mj_contactForce(w.m,w.d,ci,force);normal[foot]+=max(0.,force[0])
                if force[0]>.05*w.weight:
                    jp=np.zeros((3,w.m.nv));mujoco.mj_jac(w.m,w.d,jp,None,contact.pos,body)
                    velocity=jp@w.d.qvel;n=contact.frame[:3]
                    slip=max(slip,float(np.linalg.norm(velocity-n*(velocity@n))))
                    util=max(util,float(np.linalg.norm(force[1:3])/max(contact.friction[0]*force[0],1e-9)))
            path['slip']+=.01*slip;path['friction_exposure']+=.01*max(0.,util-.85)
            path['bilateral_deficit']+=.01*float(np.maximum(.05-normal/w.weight,0.).sum())
            path['unloaded_time']+=.01*int(normal.sum()<.05*w.weight)
            for name,geoms in groups.items():
                clear=min((mujoco.mj_geomDistance(w.m,w.d,int(g),int(f),1.,None) for g in geoms for f in floor),default=1.)
                path[name]=min(path[name],clear)
        if monitor is not None:
            # Preserve native takeoff tracker and actual contact observations.
            row=w.measure()
            if monitor.failed:break
    result=integration(w)
    if quadrature:result['quadrature']=path
    return result
