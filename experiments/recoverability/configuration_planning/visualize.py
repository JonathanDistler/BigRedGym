"""Offscreen MuJoCo replays; geometric initialization is labeled separately."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from PIL import Image,ImageDraw
from experiments.morphology.learning.environment import Assets


def animation(model,states,path,label):
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,360,480)
    camera=mujoco.MjvCamera();camera.lookat[:]=[.08,0,.35];camera.distance=1.35;camera.azimuth=90;camera.elevation=-15
    frames=[]
    try:
        stride=max(1,len(states)//70)
        for state in states[::stride]:
            data.qpos[:]=state['qpos'];data.qvel[:]=state.get('qvel',np.zeros(model.nv));mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera);frame=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(frame);draw.rectangle((0,0,480,39),fill=(20,20,20))
            draw.text((7,4),label,fill='white');draw.text((7,21),state.get('caption',f"Simulator time {state.get('time',0):.3f} s"),fill='white')
            frames.append(frame)
        frames[0].save(path,save_all=True,append_images=frames[1:],duration=70,loop=0)
    finally:renderer.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plans',type=Path,required=True);p.add_argument('--refined',type=Path,required=True)
    p.add_argument('--connectivity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);assets=Assets(a.output/'assets');model=assets.models[1.]
    for mode in 'ABCD':
        states=torch.load(a.plans/f'motion_{mode}.pt',weights_only=False)
        animation(model,states,a.output/f'actual_mode_{mode}.gif',f'MuJoCo actual single replay, mode {mode}; rejected')
    states=torch.load(a.refined/'motion_D.pt',weights_only=False)
    animation(model,states,a.output/'actual_refined_D.gif','MuJoCo actual refined launch replay; rejected')
    path=json.loads((a.connectivity/'dense_configuration_path.json').read_text())
    states=[dict(qpos=r['qpos'],caption=f"Pitch {r['pitch']} degrees; no dynamic reachability claim") for r in path['nodes']]
    animation(model,states,a.output/'geometry_only_path.gif','IK GEOMETRY ONLY - NOT an executed trajectory')

if __name__=='__main__':main()
