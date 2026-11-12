"""Exact full-model geometry, contact Jacobians and quasi-static feasibility."""
import numpy as np
import mujoco
from scipy.optimize import least_squares, minimize_scalar
from experiments.recoverability.physics import World,rotation


class ConfigurationProblem:
    def __init__(self,assets,cfg):
        self.w=World(assets,cfg);self.m=self.w.m;self.d=mujoco.MjData(self.m)
        self.vars=np.r_[0,1,self.w.vids];self.qvars=np.r_[0,1,self.w.qids]
        self.ground=np.flatnonzero(self.m.geom_bodyid==0)
        self.collision_geoms=[g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] and self.m.geom_contype[g]]
        self.nonfeet=[g for g in self.collision_geoms if g not in self.w.foot_geoms]
        self.group={}
        for name,prefix in [('head',('Head',)),('front_calf',('FL_calf','FR_calf'))]:
            self.group[name]=[g for g in self.nonfeet if self.m.body(self.m.geom_bodyid[g]).name.startswith(prefix)]
        self.radius=float(self.m.geom_size[self.w.foot_geoms[0],0])
        if not np.all(self.m.geom_type[self.w.foot_geoms]==mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError('Atlas contact geometry assumes the original spherical Go2 feet')
        if not np.all(self.m.geom_condim[self.w.foot_geoms]<=3):
            raise ValueError('Point-contact statics does not model rolling contact moments')
        self.last_x=None;self.cache=None

    def set(self,qpos,qvel=None):
        self.d.qpos[:]=qpos;self.d.qvel[:]=0 if qvel is None else qvel
        self.d.qfrc_applied[:]=0;mujoco.mj_forward(self.m,self.d)

    def clearances(self,jac=False):
        values=[];rows=[];points=np.empty(6)
        for g in self.nonfeet:
            best=1.;point=None
            for floor in self.ground:
                distance=mujoco.mj_geomDistance(self.m,self.d,int(g),int(floor),1.,points)
                if distance<best:best=distance;point=points[:3].copy()
            values.append(float(best))
            if jac:
                jp=np.zeros((3,self.m.nv))
                if point is not None:mujoco.mj_jac(self.m,self.d,jp,None,point,int(self.m.geom_bodyid[g]))
                rows.append(jp[2,self.vars])
        indices={g:i for i,g in enumerate(self.nonfeet)}
        groups={key:min((values[indices[g]] for g in geoms),default=1.) for key,geoms in self.group.items()}
        groups['all_nonfoot']=min(values,default=1.)
        self_depth=max((-float(c.dist) for c in self.d.contact if c.dist<0 and self.m.geom_bodyid[c.geom1] and self.m.geom_bodyid[c.geom2]),default=0.)
        return np.array(values),np.array(rows),dict(**groups,self_penetration=self_depth)

    def evaluate(self,x):
        if self.last_x is not None and np.array_equal(x,self.last_x):return self.cache
        self.d.qpos[:]=self.template;self.d.qpos[self.qvars]=x;self.d.qvel[:]=0
        mujoco.mj_forward(self.m,self.d)
        residual=[];jac=[]
        for i,g in enumerate(self.w.foot_geoms[:2]):
            jp=np.zeros((3,self.m.nv));mujoco.mj_jacGeom(self.m,self.d,jp,None,int(g))
            residual.extend(1000*(self.d.geom_xpos[g]-self.contacts[i]));jac.extend(1000*jp[:,self.vars])
        jc=np.zeros((3,self.m.nv));mujoco.mj_jacSubtreeCom(self.m,self.d,jc,self.w.base)
        com=self.d.subtree_com[self.w.base]
        residual.extend(500*(com[:2]-[self.com_target,self.com_y]));jac.extend(500*jc[:2,self.vars])
        # Weak branch regularization leaves all twelve joints free for IK.
        residual.extend(.015*(x[2:]-self.seed));jac.extend(np.c_[np.zeros((12,2)),.015*np.eye(12)])
        distances,dj,groups=self.clearances(True)
        margin=.005
        deficits=np.minimum(distances-margin,0.)
        residual.extend(1000*deficits);jac.extend(1000*dj*(distances<margin)[:,None])
        for geom in self.w.foot_geoms[2:]:
            clearance=self.d.geom_xpos[geom,2]-self.radius-.015
            jp=np.zeros((3,self.m.nv));mujoco.mj_jacGeom(self.m,self.d,jp,None,int(geom))
            residual.append(1000*min(clearance,0.));jac.append(1000*jp[2,self.vars]*int(clearance<0))
        # Fixed-size self-contact residual with a contact-normal Jacobian.
        self_res=0.;self_jac=np.zeros(14)
        for contact in self.d.contact:
            b1,b2=self.m.geom_bodyid[[contact.geom1,contact.geom2]]
            if b1 and b2 and contact.dist<self_res:
                self_res=float(contact.dist);j1=np.zeros((3,self.m.nv));j2=np.zeros_like(j1)
                mujoco.mj_jac(self.m,self.d,j1,None,contact.pos,int(b1));mujoco.mj_jac(self.m,self.d,j2,None,contact.pos,int(b2))
                self_jac=contact.frame[:3]@(j2-j1)[:,self.vars]
        residual.append(1000*self_res);jac.append(1000*self_jac)
        self.last_x=x.copy();self.cache=(np.array(residual),np.array(jac))
        return self.cache

    def solve(self,pitch,height,com_target,profile,seed,max_nfev=60):
        width,stagger,roll,yaw=profile[:4]
        self.com_y=profile[4] if len(profile)>4 else 0.
        self.contacts=np.array([[stagger/2,width/2,self.radius],[-stagger/2,-width/2,self.radius]])
        self.com_target=com_target;self.seed=np.clip(seed,*self.w.bounds.T)
        matrix=rotation(np.deg2rad(pitch),np.deg2rad(roll))
        cy,sy=np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw))
        matrix=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1.]])@matrix
        self.template=self.w.d.qpos.copy();self.template[:3]=[com_target,0,height]
        mujoco.mju_mat2Quat(self.template[3:7],matrix.flatten())
        self.template[self.w.qids]=self.seed
        lo=np.r_[-.65,-.12,self.w.bounds[:,0]+1e-5];hi=np.r_[.45,.12,self.w.bounds[:,1]-1e-5]
        x=np.clip(np.r_[com_target,0,self.seed],lo,hi);self.last_x=None
        result=least_squares(lambda x:self.evaluate(x)[0],x,jac=lambda x:self.evaluate(x)[1],bounds=(lo,hi),
                             max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-7)
        self.evaluate(result.x)
        q=self.d.qpos.copy();feet=self.d.geom_xpos[self.w.foot_geoms[:2]].copy();com=self.d.subtree_com[self.w.base].copy()
        _,_,clear=self.clearances()
        clear['rear_foot']=float(self.d.geom_xpos[self.w.foot_geoms[2:],2].min()-self.radius)
        foot_error=float(np.max(np.linalg.norm(feet-self.contacts,axis=1)))
        com_error=float(np.linalg.norm(com[:2]-[com_target,self.com_y]))
        violations=max(0.,float(np.max(self.w.bounds[:,0]-q[self.w.qids])),float(np.max(q[self.w.qids]-self.w.bounds[:,1])))
        geometric=(foot_error<=1e-4 and clear['head']>=.0049 and clear['front_calf']>=.0049 and
                   clear['all_nonfoot']>=.0049 and clear['rear_foot']>=.0149 and clear['self_penetration']<1e-6 and violations<1e-6)
        # COM-target tracking is a separate atlas cell constraint, not a collision.
        in_cell=com_error<=2e-4
        reasons=[]
        if foot_error>1e-4:reasons.append('front_contact_IK')
        if clear['head']<.0049:reasons.append('head_clearance')
        if clear['front_calf']<.0049:reasons.append('front_calf_clearance')
        if clear['all_nonfoot']<.0049:reasons.append('other_body_clearance')
        if clear['self_penetration']>=1e-6:reasons.append('self_collision')
        if clear['rear_foot']<.0149:reasons.append('rear_foot_clearance')
        if not in_cell:reasons.append('COM_target')
        return dict(pitch=pitch,height=height,com_target=com_target,com_y_target=self.com_y,profile=list(profile),geometric=bool(geometric),
                    in_cell=bool(in_cell),geometric_cell=bool(geometric and in_cell),solver_converged=bool(result.success),
                    nfev=result.nfev,foot_error=foot_error,com_target_error=com_error,com=com.tolist(),
                    com_relative=(com-np.r_[feet[:,:2].mean(0),0]).tolist(),qpos=q.tolist(),joints=q[self.w.qids].tolist(),
                    foot_positions=feet.tolist(),clearances=clear,joint_violation=violations,reasons=reasons)

    def static(self,qpos,mode='A',load_fraction=None):
        """Point-contact force equilibrium; static forces are never injected.

        Solve full free-base equilibrium and use its nullspace to minimize motor
        utilization. Coulomb cones are checked exactly, not a friction pyramid.
        """
        self.set(qpos);legs=(0,1) if mode!='single_FL' else (0,)
        jac=[];mu=[]
        for i in legs:
            g=int(self.w.foot_geoms[i]);point=self.d.geom_xpos[g].copy();point[2]-=self.radius
            jp=np.zeros((3,self.m.nv));mujoco.mj_jac(self.m,self.d,jp,None,point,int(self.w.feet[i]));jac.append(jp)
            contact=next((c for c in self.d.contact if g in (c.geom1,c.geom2) and 0 in self.m.geom_bodyid[[c.geom1,c.geom2]]),None)
            if contact is not None:
                if contact.dim>3:raise ValueError('Atlas point-contact solver requires condim <= 3')
                mu.append(float(contact.friction[0]))
            else:mu.append(float(max(self.m.geom_friction[g,0],self.m.geom_friction[self.ground[0],0])))
        J=np.vstack(jac);A=J[:,:6].T;g=self.d.qfrc_bias.copy()
        U,s,Vh=np.linalg.svd(A,full_matrices=True);rank=int(np.sum(s>1e-8))
        particular=np.linalg.lstsq(A,g[:6],rcond=1e-9)[0];null=Vh[rank:].T
        def metrics(force):
            f=force.reshape(-1,3);tau=g[self.w.vids]-J[:,self.w.vids].T@force
            fn=f[:,2];tangent=np.linalg.norm(f[:,:2],axis=1)
            cone=np.array(mu)*fn-tangent
            fraction=float(fn[0]/max(fn.sum(),1e-12))
            force_error=A@force-g[:6]
            return tau,fn,cone,fraction,force_error
        def cost(alpha):
            f=particular+(null[:,0]*alpha if null.shape[1] else 0)
            tau,fn,cone,fraction,error=metrics(f)
            penalty=1000*np.minimum(fn,0).dot(np.minimum(fn,0))+1000*np.minimum(cone,0).dot(np.minimum(cone,0))
            if load_fraction is not None:penalty+=1000*(fraction-load_fraction)**2
            return float(np.max(abs(tau)/self.w.effort)**2+penalty)
        alpha=minimize_scalar(cost,bounds=(-2*self.w.weight,2*self.w.weight),method='bounded').x if null.shape[1] else 0.
        force=particular+(null[:,0]*alpha if null.shape[1] else 0)
        tau,fn,cone,fraction,error=metrics(force)
        balanced=bool(np.linalg.norm(error[:3])<.05 and np.linalg.norm(error[3:])<.01)
        torque_ratio=float(np.max(abs(tau)/self.w.effort));friction_ok=bool((cone>=-1e-6).all())
        unilateral=bool((fn>=.01*self.w.weight).all())
        fraction_ok=load_fraction is None or abs(fraction-load_fraction)<.02
        return dict(feasible=bool(balanced and torque_ratio<=1 and friction_ok and unilateral and fraction_ok),
                    equilibrium=balanced,force_residual_N=float(np.linalg.norm(error[:3])),moment_residual_Nm=float(np.linalg.norm(error[3:])),
                    forces=force.reshape(-1,3).tolist(),joint_torques=tau.tolist(),torque_ratio=torque_ratio,
                    torque_margin_Nm=float(np.min(self.w.effort-abs(tau))),normal_forces=fn.tolist(),
                    friction_mu=mu,friction_margin_N=cone.tolist(),friction_utilization=(np.linalg.norm(force.reshape(-1,3)[:,:2],axis=1)/np.maximum(np.array(mu)*fn,1e-10)).tolist(),
                    load_fraction_FL=fraction,requested_load_fraction=load_fraction,unilateral=unilateral,
                    friction_ok=friction_ok,rank=rank,contact_mode=mode)
