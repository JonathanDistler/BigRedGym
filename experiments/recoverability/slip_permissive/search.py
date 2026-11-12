"""Sliding-permissive multiple shooting; old strict optimizer is untouched."""
import numpy as np
import mujoco
from experiments.recoverability.configuration_planning.shooting import MultipleShooting
from experiments.recoverability.configuration_planning.dynamics import segment,put


class SlipPermissiveShooting(MultipleShooting):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.defect_weight=20.
        self.path_constraints=True
        self.command_low=np.maximum(self.w.bounds[:,0],self.w.nominal-self.w.cfg.action_scale)
        self.command_high=np.minimum(self.w.bounds[:,1],self.w.nominal+self.w.cfg.action_scale)

    def guess(self):
        x,lo,hi=super().guess()
        for k in range(self.n):
            offset=k*self.nstate
            lo[offset:offset+2]=-3.;hi[offset:offset+2]=3.
        lo[self.n*self.nstate:]=np.tile(self.command_low,self.n)
        hi[self.n*self.nstate:]=np.tile(self.command_high,self.n)
        x=np.clip(x,lo+1e-7,hi-1e-7)
        # Last node should arrive with balance-compatible, not increasing, spin.
        x[(self.n-1)*self.nstate+self.nv:self.n*self.nstate]=0.
        return x,lo,hi

    def residual(self,x,details=False):
        self.calls+=1;nodes,commands=self.decode(x);res=[];defects=[];physics=[]
        for k,(q,v) in enumerate(nodes):
            state=self.start if k==0 else self.node_start(*nodes[k-1],commands[k-1],k)
            end=segment(self.w,state,commands[k],self.durations[k],quadrature=True)
            path=end['quadrature']
            # NO cumulative slip, drift, friction-exposure or original x/y
            # anchor terms. Native MuJoCo still enforces its friction law.
            res.extend([500*min(0.,path['head']-.005),500*min(0.,path['calf']-.005),
                        max(0.,path['unloaded_time']-.08)/.02])
            dq=np.zeros(self.nv);mujoco.mj_differentiatePos(self.w.m,dq,1.,q,end['qpos'])
            res.extend(self.defect_weight*np.r_[dq/self.qscale,(end['qvel']-v)/self.vscale])
            res.extend(self.defect_weight*(end['target']-commands[k])/.1)
            defects.append(dict(root_position_m=float(np.linalg.norm(dq[:3])),rotation_rad=float(np.linalg.norm(dq[3:6])),
                joint_rad=float(abs(dq[6:]).max()),velocity_max=float(abs(end['qvel']-v).max()),
                pose_max_m_or_rad=float(abs(dq).max()),command_memory_defect=float(abs(end['target']-commands[k]).max())))
            self.problem.set(q,v);_,_,clear=self.problem.clearances()
            res.extend(500*np.minimum([clear['head']-.005,clear['front_calf']-.005,clear['all_nonfoot']-.005,-clear['self_penetration']],0.))
            feet=self.problem.d.geom_xpos[self.w.foot_geoms[:2]]
            res.extend((feet[:,2]-self.problem.radius)/.015)
            delta=np.zeros(self.nv);mujoco.mj_differentiatePos(self.w.m,delta,1.,self.refs[k],q)
            # Body x/y translate freely. Height/orientation and joint shape
            # remain useful geometric priors, not physical-state projections.
            res.extend(.15*delta[2:]/np.r_[.10,[.35]*3,[.6]*12])
            matrix=self.problem.d.xmat[self.w.base].reshape(3,3)
            pitch=np.degrees(np.arctan2(-matrix[2,0],matrix[0,0]))
            target_pitch=self.rows[k].get('desired_pitch',self.rows[k]['pitch'])
            res.append((pitch-target_pitch)/(4. if k==self.n-1 else 10.))
            com=self.problem.d.subtree_com[self.w.base]
            res.append((com[0]-feet[:,0].mean())/(.04 if k==self.n-1 else .10))
            res.append((q[2]-self.refs[k][2])/.06)
            put(self.w,end);row=self.w.physical_row();load=row['loads'][:2]/self.w.weight
            res.extend(np.minimum(load-.05,0.)/.08)
            joint=self.w.d.qpos[self.w.qids]
            res.extend(20*np.minimum(joint-self.w.bounds[:,0],0.));res.extend(20*np.minimum(self.w.bounds[:,1]-joint,0.))
            res.extend(20*np.maximum(abs(self.w.d.qvel[self.w.vids])/self.w.speed-1,0.))
            physics.append(dict(pitch=pitch,target_pitch=target_pitch,clearances=clear,front_loads=load.tolist(),
                                slip_recorded_only=path['slip'],friction_recorded_only=path['friction_exposure']))
        last=nodes[-1][1];res.extend(last[:3]/.35);res.extend(last[3:6]/1.5)
        res.extend(.2*(nodes[-1][0][self.w.qids]-self.refs[-1][self.w.qids])/.3)
        residual=np.nan_to_num(np.asarray(res),nan=1e6,posinf=1e6,neginf=-1e6)
        return (residual,defects,physics) if details else residual
