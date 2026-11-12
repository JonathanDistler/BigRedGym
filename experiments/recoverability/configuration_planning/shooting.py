"""Deterministic hybrid multiple shooting using native MuJoCo dynamics.

Intermediate shooting nodes are optimizer hypotheses, never accepted physical
resets. All candidates are independently replayed by one uninterrupted simulation.
"""
import time
import numpy as np
import mujoco
from scipy.optimize import least_squares
from experiments.recoverability.physics import World
from experiments.recoverability.configuration_planning.dynamics import segment,integration,put
from experiments.recoverability.configuration_planning.geometry import ConfigurationProblem


class MultipleShooting:
    def __init__(self,assets,cfg,start,waypoints,mode,durations):
        self.w=World(assets,cfg);self.problem=ConfigurationProblem(assets,cfg);self.start=start
        self.refs=[np.array(row['qpos']) for row in waypoints];self.rows=waypoints;self.mode=mode
        self.durations=durations;self.n=len(waypoints);self.nv=self.w.m.nv
        self.nstate=2*self.nv;self.history=[];self.calls=0
        self.defect_weight=10.
        self.path_constraints=False
        self.qscale=np.r_[[.03]*3,[.10]*3,[.15]*12]
        self.vscale=np.r_[[.3]*3,[.6]*3,[2.]*12]

    def decode(self,x):
        nodes=[]
        for k in range(self.n):
            state=x[k*self.nstate:(k+1)*self.nstate]
            q=self.refs[k].copy();mujoco.mj_integratePos(self.w.m,q,state[:self.nv],1.)
            nodes.append((q,state[self.nv:].copy()))
        return nodes,x[self.n*self.nstate:].reshape(self.n,12)

    def guess(self):
        values=[];lo=[];hi=[];commands=[];previous=self.start['qpos']
        for k,(ref,row) in enumerate(zip(self.refs,self.rows)):
            velocity=np.zeros(self.nv);mujoco.mj_differentiatePos(self.w.m,velocity,self.durations[k],previous,ref)
            speed=np.r_[[2.]*3,[6.]*3,self.w.speed]
            velocity=np.clip(velocity,-.8*speed,.8*speed)
            values.extend(np.r_[np.zeros(self.nv),velocity]);previous=ref
            lower=np.r_[[-.35,-.14,-.15],[-.6]*3,self.w.bounds[:,0]-ref[self.w.qids]]
            upper=np.r_[[.35,.14,.15],[.6]*3,self.w.bounds[:,1]-ref[self.w.qids]]
            lo.extend(np.r_[lower,-speed]);hi.extend(np.r_[upper,speed])
            tau=np.array(row.get('static',{}).get('joint_torques',[0.]*12))
            command=ref[self.w.qids]+(tau+self.w.cfg.kd*velocity[self.w.vids])/self.w.cfg.kp
            commands.append(np.clip(command,*self.w.bounds.T))
        values.extend(np.array(commands).flatten());lo.extend(np.tile(self.w.bounds[:,0],self.n));hi.extend(np.tile(self.w.bounds[:,1],self.n))
        return np.array(values),np.array(lo),np.array(hi)

    def node_start(self,q,v,target,k):
        mujoco.mj_resetData(self.w.m,self.w.d);self.w.d.qpos[:]=q;self.w.d.qvel[:]=v
        self.w.d.time=self.start['time']+sum(self.durations[:k]);mujoco.mj_forward(self.w.m,self.w.d)
        self.w.target=target.copy();return integration(self.w)

    def residual(self,x,details=False):
        self.calls+=1;nodes,commands=self.decode(x);res=[];defects=[];physics=[]
        for k,(q,v) in enumerate(nodes):
            if k==0:state=self.start
            else:state=self.node_start(*nodes[k-1],commands[k-1],k)
            end=segment(self.w,state,commands[k],self.durations[k],quadrature=self.path_constraints)
            if self.path_constraints:
                path=end['quadrature'];remaining=max(.002,.02-self.start.get('prefix_slip',.006))
                res.extend([max(0.,path['slip']-remaining/self.n)/.002,
                    path['friction_exposure']/.005,1000*min(0.,path['head']-.005),1000*min(0.,path['calf']-.005),
                    path['bilateral_deficit']/.005 if self.mode=='A' else max(0.,path['unloaded_time']-.08)/.01])
            dq=np.zeros(self.nv);mujoco.mj_differentiatePos(self.w.m,dq,1.,q,end['qpos'])
            defect=np.r_[dq/self.qscale,(end['qvel']-v)/self.vscale]
            res.extend(self.defect_weight*defect);res.extend(self.defect_weight*(end['target']-commands[k])/.1)
            defects.append(dict(pose_max_m_or_rad=float(abs(dq).max()),velocity_max=float(abs(end['qvel']-v).max()),
                                command_memory_defect=float(abs(end['target']-commands[k]).max())))
            self.problem.set(q,v);_,_,clear=self.problem.clearances()
            res.extend(1000*np.minimum([clear['head']-.005,clear['front_calf']-.005,clear['all_nonfoot']-.005,-clear['self_penetration']],0.))
            feet=self.problem.d.geom_xpos[self.w.foot_geoms[:2]]
            foot_height=feet[:,2]-self.problem.radius
            desired_height=.025 if self.mode=='C' and k==0 else 0.
            res.extend((foot_height-desired_height)/.005)
            if self.mode!='C' or k==self.n-1:
                res.extend(((feet[:,:2]-self.start_anchors)/.015).flatten())
            else:res.extend(np.zeros(4))
            # Initial guesses do not impose static equilibrium or zero velocity.
            delta=np.zeros(self.nv);mujoco.mj_differentiatePos(self.w.m,delta,1.,self.refs[k],q)
            res.extend(.3*delta/np.r_[[.1]*3,[.3]*3,[.4]*12])
            matrix=self.problem.d.xmat[self.w.base].reshape(3,3)
            pitch=np.degrees(np.arctan2(-matrix[2,0],matrix[0,0]))
            com=self.problem.d.subtree_com[self.w.base]
            target_pitch=45 if k==self.n-1 else self.rows[k]['pitch']
            res.append((pitch-target_pitch)/3.)
            res.append((com[0]-feet[:,0].mean()-max(-.10,self.rows[k]['com_target']))/.035)
            res.append((q[2]-self.refs[k][2])/.06)
            # Physical forces from simulated segment endpoints, not assigned forces.
            put(self.w,end);row=self.w.physical_row();load=row['loads'][:2]/self.w.weight
            if self.mode=='C' and k==0:res.extend(load/.1)
            elif self.mode=='B' and k==0:res.extend((load-np.array([.7,.3]))/.25)
            else:res.extend(np.minimum(load-.05,0.)/.05)
            joint=self.w.d.qpos[self.w.qids]
            res.extend(10*np.minimum(joint-self.w.bounds[:,0],0.));res.extend(10*np.minimum(self.w.bounds[:,1]-joint,0.))
            res.extend(10*np.maximum(abs(self.w.d.qvel[self.w.vids])/self.w.speed-1,0.))
            physics.append(dict(pitch=pitch,clearances=clear,front_loads=load.tolist()))
        last_v=nodes[-1][1]
        res.extend(last_v[:3]/.75);res.extend((last_v[3:6]-[0.,.8,0.])/2.)
        result=np.asarray(res,dtype=float).flatten()
        if not np.isfinite(result).all():result=np.nan_to_num(result,nan=1e6,posinf=1e6,neginf=-1e6)
        if details:return result,defects,physics
        return result

    def solve(self,max_nfev=8,warm_plan=None):
        put(self.w,self.start);self.start_anchors=self.w.d.geom_xpos[self.w.foot_geoms[:2],:2].copy()
        x,lo,hi=self.guess();initial=self.residual(x,True)
        if warm_plan is not None:
            chunks=[]
            for ref,node in zip(self.refs,warm_plan['nodes']):
                delta=np.zeros(self.nv);mujoco.mj_differentiatePos(self.w.m,delta,1.,ref,np.array(node['qpos']))
                factor=min(1.,sum(warm_plan.get('durations',self.durations))/sum(self.durations))
                chunks.extend(np.r_[delta,np.array(node['qvel'])*factor])
            chunks.extend(np.array(warm_plan['commands']).flatten());x=np.clip(chunks,lo+1e-8,hi-1e-8)
            initial=self.residual(x,True)
        began=time.monotonic()
        result=least_squares(self.residual,x,bounds=(lo,hi),max_nfev=max_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-4,x_scale='jac',diff_step=1e-4)
        residual,defects,physics=self.residual(result.x,True);nodes,commands=self.decode(result.x)
        return dict(mode=self.mode,solver_success=bool(result.success),message=result.message,nfev=result.nfev,
                    residual_calls=self.calls,seconds=time.monotonic()-began,initial_cost=float(initial[0]@initial[0]/2),
                    final_cost=float(residual@residual/2),defects=defects,node_physics=physics,
                    commands=commands.tolist(),durations=self.durations,nodes=[dict(qpos=q.tolist(),qvel=v.tolist()) for q,v in nodes],
                    initial_defects=initial[1],dynamic_feasibility_established=all(d['pose_max_m_or_rad']<1e-3 and d['velocity_max']<1e-2 and d['command_memory_defect']<1e-5 for d in defects),
                    validation_required=True)
