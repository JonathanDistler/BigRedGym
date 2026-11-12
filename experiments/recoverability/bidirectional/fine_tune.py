"""Opt-in upright balance fine-tuning only: two-update smoke / gated ten-update pilot.

The cloned saved actor keeps its observation encoder and unbounded residual action
interface. Its normalization buffers and the baseline policy remain frozen.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import curriculum
from experiments.recoverability.run import KICK,dump,sha
from experiments.recoverability.bidirectional.run import controller
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.bidirectional.core import restore_contact,continuation,prepare


class BalanceLearner(nn.Module):
    def __init__(self,c,exploration=.03):
        super().__init__()
        self.actor=copy.deepcopy(c.actor).eval()
        self.actor.requires_grad_(False)
        self.actor.NN.requires_grad_(True)
        self.log_std=nn.Parameter(torch.full((12,),float(np.log(exploration))))
        self.critic=nn.Sequential(nn.Linear(c.dim,128),nn.ELU(),nn.Linear(128,1))

    def distribution(self,obs):
        return Normal(self.actor.act_inference(obs),self.log_std.clamp(-5.,-1.).exp())

    def value(self,obs):return self.critic(obs).squeeze(-1)


class AdaptiveBands:
    def __init__(self,adaptive=True):
        self.bands=(90,85,80,75);self.stage=0;self.adaptive=adaptive;self.history=[];self.window=[]

    def observe(self,rows):
        self.window=(self.window+rows)[-20:]
        rows=self.window;n=len(rows);k=sum(r['success'] for r in rows)
        safe=all(not r['collision'] and r['max_speed']<=1.001 and r['max_torque']<=1.00001
                 and r['max_joint']<.001 and r['loaded_slip']<=.02 for r in rows)
        # Curriculum validation seeds differ from training and final held-out seeds.
        advance=self.adaptive and n>=20 and len({r['seed'] for r in rows})==n and k/n>=.8 and safe
        self.history.append(dict(stage=self.stage,n=n,k=k,safe=safe,advance=advance))
        if advance:self.stage=min(self.stage+1,len(self.bands)-1);self.window=[]
        return advance


def verified_gate(path):
    gate=json.loads(Path(path).read_text())
    if not (gate.get('fine_tuning_eligible') and gate.get('controller_compatible') and gate.get('valid_reset_proposals')):
        raise ValueError('Balance fine-tuning requires compatibility, reset and upright recovery validation')
    return gate


def reward(sample,row,dt,failed,ablation='none'):
    loaded=float(row['front_support']);omega=sample['omega'];x=sample['com_support_x']
    terms=dict(support=2*loaded,upright=4*np.exp(-(sample['tilt']/10)**2)*loaded,
               regulation=-.2*omega**2,com=2*np.exp(-(x/.05)**2),
               slip=-8*sample['slip_speed'],actuator=-.02*sample['torque_ratio']**2)
    if ablation=='no_slip':terms['slip']=0.
    if ablation=='no_com':terms['com']=0.
    # Temporary unloading loses the support reward but never terminates an episode.
    return dt*sum(terms.values())-10*bool(failed)


def step_control(w,m,a,raw,ablation):
    a.previous=raw.detach()*torch.as_tensor(a.controller.env.scales['dof_pos_target'])
    target=(a.reference+a.previous).numpy().astype(float);total=0.
    for _ in range(5):
        row=w.tick(target);a.advance_memory(row['loads'],w.m.opt.timestep)
        sample=m.observe(row,w.m.opt.timestep)
        total+=reward(sample,row,w.m.opt.timestep,m.failed,ablation)
        if m.failed:break
    done=m.failed or m.best_hold>=5
    if m.best_hold>=5:total+=5
    return total,done


def evaluate(learner,c,assets,cfg,pitch,seeds,split,stochastic=False):
    rows=[]
    for seed in seeds:
        w,m,a,reset,saved=prepare(c,assets,cfg,pitch,'nominal',seed,split)
        result=continuation(c,w,m,a,stochastic,seed+777,actor=learner.actor.act_inference) if saved is not None else m.result()
        rows.append(dict(seed=seed,pitch=pitch,stochastic=stochastic,reset=reset,**result))
    return rows


def run_updates(learner,c,assets,cfg,bank,metadata,count,seed,ablation,adaptive):
    generator=torch.Generator().manual_seed(seed);rng=np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimizer=torch.optim.Adam([dict(params=learner.actor.NN.parameters(),lr=1e-5),
                                dict(params=[learner.log_std],lr=1e-5),
                                dict(params=learner.critic.parameters(),lr=1e-3)])
    schedule=AdaptiveBands(adaptive);records=[];episode_steps=0
    def reset():
        # A scheduled frontier sample is admitted only from valid TRAIN proposals.
        target=schedule.bands[schedule.stage]
        if schedule.stage>0:
            draw=rng.random()
            target=90 if draw<.2 else schedule.bands[schedule.stage-1] if draw<.8 else target
        ids=[key for key,meta in metadata.items() if meta['band']==target and meta['split']=='train']
        if not ids:raise ValueError('No validated training reset states in active band')
        key=ids[int(rng.integers(len(ids)))];return restore_contact(c,assets,cfg,bank[key])
    w,m,a=reset()
    for update in range(1,count+1):
        observations=[];actions=[];old_log=[];values=[];rewards=[];dones=[];next_values=[]
        for _ in range(256):
            obs=a.observation()[0].detach()
            with torch.no_grad():
                dist=learner.distribution(obs);raw=dist.loc+dist.scale*torch.randn(12,generator=generator)
                log=dist.log_prob(raw).sum();value=learner.value(obs)
            r,done=step_control(w,m,a,raw,ablation);episode_steps+=1
            timeout=episode_steps>=650;done=done or timeout
            with torch.no_grad():nv=learner.value(a.observation()[0]).item() if not done else 0.
            observations.append(obs);actions.append(raw);old_log.append(log);values.append(value)
            rewards.append(r);dones.append(done);next_values.append(nv)
            if done:w,m,a=reset();episode_steps=0
        val=torch.stack(values);adv=torch.zeros(256);carry=0.
        for t in reversed(range(256)):
            delta=rewards[t]+.995*next_values[t]*(not dones[t])-val[t]
            carry=delta+.995*.95*(not dones[t])*carry;adv[t]=carry
        returns=adv+val;adv=(adv-adv.mean())/adv.std(unbiased=False).clamp(min=1e-6)
        obs=torch.stack(observations);raw=torch.stack(actions);old=torch.stack(old_log)
        stats=[]
        for epoch in range(2):
            for ids in torch.randperm(256,generator=generator).split(64):
                dist=learner.distribution(obs[ids]);log=dist.log_prob(raw[ids]).sum(-1)
                ratio=(log-old[ids]).exp();kl=((ratio-1)-(log-old[ids])).mean()
                if kl>.02:break
                policy=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
                vf=(learner.value(obs[ids])-returns[ids]).square().mean()
                loss=policy+.5*vf
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite balance PPO objective')
                optimizer.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(learner.parameters(),.5)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite balance PPO gradient')
                optimizer.step();stats.append(dict(loss=float(loss.detach()),kl=float(kl.detach())))
        validation=evaluate(learner,c,assets,cfg,schedule.bands[schedule.stage],range(202614120+100*update,202614124+100*update),'heldout')
        schedule.observe(validation)
        records.append(dict(update=update,optimizer_steps=len(stats),mean_reward=float(np.mean(rewards)),
                            losses=stats,validation=validation,curriculum_stage=schedule.stage))
        print('balance update',update,'validation',sum(r['success'] for r in validation),len(validation),flush=True)
    return records,schedule


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','smoke','pilot'))
    p.add_argument('--validation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke-gate',type=Path);p.add_argument('--ablation',choices=('none','no_slip','no_com','fixed_band'),default='none')
    a=p.parse_args();gate=verified_gate(a.validation/'validation_gate.json')
    audit=json.loads((a.validation/'compatibility.json').read_text());assert audit['compatible']
    if a.mode=='pilot':
        if a.smoke_gate is None:raise ValueError('Pilot requires a separate passing two-update smoke gate')
        smoke=json.loads(a.smoke_gate.read_text())
        if not smoke.get('pilot_eligible') or smoke.get('updates')!=2:raise ValueError('Smoke gate rejected balance pilot')
        if smoke.get('validation_sha256')!=sha(a.validation/'manifest.json'):raise ValueError('Smoke/validation provenance mismatch')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    config=dict(experiment='upright balance fine-tuning; not transition PPO',updates=2 if a.mode=='smoke' else 10,
                source_validation=str(a.validation),validation_sha256=sha(a.validation/'manifest.json'),
                steps_per_update=256,control_hz=100,actor_learning_rate=1e-5,epochs=2,initial_exploration_std=.03,
                adaptive_bands=[90,85,80,75],ablation=a.ablation,final_morphology=1.,transition_ppo_enabled=False,
                training_seed=202616120 if a.mode!='pilot' else 202616220,
                final_eval_seed=202615520 if a.mode!='pilot' else 202618520,
                frontier_mixture='20% 90 degrees, 60% previous validated band, 20% new band after advancement',
                pilot_requires='separate two-update smoke gate, no deterministic nominal regression or physical failure')
    dump(a.output/'configuration.json',config)
    if a.mode=='prepare':return
    net,_,cfg,_=curriculum.load_source(KICK);cfg.seconds=8
    c=controller();assets=Assets(a.output/'assets')
    try:
        learner=BalanceLearner(c)
        if a.mode=='pilot':
            ckpt=a.smoke_gate.parent/'balance_learner.pt'
            if sha(ckpt)!=smoke['checkpoint_sha256']:raise ValueError('Smoke checkpoint hash mismatch')
            learner.load_state_dict(torch.load(ckpt,weights_only=False)['learner'])
        bank=torch.load(a.validation/'start_states.pt',weights_only=False)
        trials=json.loads((a.validation/'basin_trials.json').read_text());metadata={}
        for r in trials:
            if (r['reset']['valid'] and r['split']=='train' and
                    (r['band']!=90 or (r['success'] and not r['stochastic']))):
                metadata.setdefault(r['reset']['state_id'],r)
        # Baseline and clone share the exact original mean before the smoke.
        initial=evaluate(learner,c,assets,cfg,90,range(202615120,202615124),'heldout')
        records,schedule=run_updates(learner,c,assets,cfg,bank,metadata,config['updates'],config['training_seed'],a.ablation,a.ablation!='fixed_band')
        final=[]
        for band in (90,85,80,75):
            for stochastic in (False,True):
                final.extend(evaluate(learner,c,assets,cfg,band,range(config['final_eval_seed'],config['final_eval_seed']+2),'heldout',stochastic))
        baseline=[]
        # Matched final seeds and native action noise for baseline comparisons.
        baselearner=BalanceLearner(c)
        for band in (90,85,80,75):
            for stochastic in (False,True):baseline.extend(evaluate(baselearner,c,assets,cfg,band,range(config['final_eval_seed'],config['final_eval_seed']+2),'heldout',stochastic))
        torch.save(dict(learner=learner.state_dict(),configuration=config,baseline_sha256=c.meta['sha256']),a.output/'balance_learner.pt')
        dump(a.output/'updates.json',records);dump(a.output/'final_evaluation.json',dict(initial=initial,baseline=baseline,learned=final,curriculum=schedule.history))
        native=[r for r in final if r['pitch']==90 and not r['stochastic']]
        passing=all(r['success'] and r['first_failure'] is None and r['loaded_slip']<=.02 for r in native)
        dump(a.output/'smoke_gate.json',dict(updates=config['updates'],pilot_eligible=bool(passing and all(r['success'] for r in initial)),
                  validation_sha256=config['validation_sha256'],checkpoint_sha256=sha(a.output/'balance_learner.pt'),
                  transition_ppo_eligible=False,baseline_unchanged=c.network_hash()==c.digest))
    finally:c.close()

if __name__=='__main__':main()
