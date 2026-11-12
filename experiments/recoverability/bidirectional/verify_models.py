"""Verify physical model scaling without changing contacts, motors, or geometry."""
import argparse
from pathlib import Path
import numpy as np
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.physics import World
from experiments.recoverability.run import dump


def audit_models(assets,cfg=None):
    native=assets.models[1.];base=World(assets,cfg);records=[]
    for scale in (.8,.85,.9,.95,1.):
        w=World(assets,cfg);w.reset(scale,0);m=w.m
        massive=m.body_mass>0
        massless=np.flatnonzero((m.body_mass==0) & (np.arange(m.nbody)>0))
        checks=dict(topology=(m.nbody,m.ngeom,m.njnt)==(native.nbody,native.ngeom,native.njnt),
            massive_body_inertia_positive=bool((m.body_inertia[massive]>0).all()),
            massless_bodies_fixed=bool((m.body_jntnum[massless]==0).all()),
            inertia_triangle_inequality=bool((2*m.body_inertia[1:].max(1)<=m.body_inertia[1:].sum(1)+1e-10).all()),
            finite_mass_com=bool(np.isfinite(m.body_mass).all() and np.isfinite(m.body_ipos).all()),
            collision_masks_equal=bool(np.array_equal(m.geom_contype,native.geom_contype) and np.array_equal(m.geom_conaffinity,native.geom_conaffinity)),
            friction_equal=bool(np.array_equal(m.geom_friction,native.geom_friction)),
            joint_limits_equal=bool(np.array_equal(w.bounds,base.bounds)),
            effort_equal=bool(np.array_equal(w.effort,base.effort)),speed_equal=bool(np.array_equal(w.speed,base.speed)),
            foot_geometry_equal=bool(np.array_equal(m.geom_size[w.foot_geoms],native.geom_size[base.foot_geoms])),
            torso_mass_equal=bool(m.body_mass[w.base]==native.body_mass[base.base]),
            torso_inertia_equal=bool(np.array_equal(m.body_inertia[w.base],native.body_inertia[base.base])))
        records.append(dict(scale=scale,verified=all(checks.values()),checks=checks,mass=float(m.body_mass.sum()),
                            body_com_offsets=m.body_ipos.tolist(),body_inertias=m.body_inertia.tolist(),
                            geometry_sizes=m.geom_size.tolist(),geometry_positions=m.geom_pos.tolist(),
                            standing_com=w.d.subtree_com[w.base].tolist(),joint_limits=w.bounds.tolist(),
                            actuator_effort=w.effort.tolist(),actuator_speed=w.speed.tolist()))
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    records=audit_models(Assets(a.output/'assets'));dump(a.output/'verified_models.json',records)
    assert all(r['verified'] for r in records)

if __name__=='__main__':main()
