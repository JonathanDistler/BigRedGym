"""Explicit invariants for requested unchanged physics and native controls."""
import argparse
from pathlib import Path
import numpy as np
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.physics import World
from experiments.recoverability.run import KICK,dump,model_audit
from experiments.recoverability.bidirectional.run import controller


def options(model):
    result={}
    for key in ('timestep','gravity','solver','integrator','cone','iterations','tolerance','ls_iterations','ls_tolerance','noslip_iterations','noslip_tolerance','impratio','enableflags','disableflags'):
        value=getattr(model.opt,key);result[key]=np.asarray(value).tolist()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    _,_,cfg,_=curriculum.load_source(KICK);assets=Assets(a.output/'assets');w=World(assets,cfg);c=controller()
    try:
        native=c.env._backend._model_for_env(0);audit=model_audit(c,w);audit['simulator_options']=options(w.m);audit['saved_balance_options']=options(native)
        assert audit['simulator_options']==audit['saved_balance_options']
        audit['kick_action_scale']=cfg.action_scale;audit['balance_residual_scale']=.25
        assert cfg.action_scale==2.
        audit['native_motor_interface']='PD kp=60 kd=3, original effort clipping and motor speed guard; no external forces'
        dump(a.output/'physics.json',audit)
    finally:c.close()

if __name__=='__main__':main()
