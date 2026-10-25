"""Frozen one-training-seed G4 comparison and separate G4b transfer recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .cli import ROOT, rollout, source_hash
from .controller import Classical
from .bc import BehaviorCloning
from .ppo import PPOPolicy


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def policies(cfg):
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    return {'classical':Classical(),'bc':BehaviorCloning(ROOT/'artifacts/g2/bc_seed0/policy.pt',digest),'ppo':PPOPolicy(ROOT/'artifacts/g3/ppo_seed0/policy.pt',digest)}


def compare(cfg, agents, output):
    scenes=json.loads((ROOT/'scenarios/test/manifest.json').read_text())
    # Predeclared fallback: first ten scenarios in each family; no policy tuning.
    selected=[]
    for course in cfg['courses']:
        selected.extend([s for s in scenes if s['course']==course][:10])
    manifest={'scope':'preliminary: existing training seed 0 only; 40 paired held-out layouts','selection_rule':cfg['selection'],'source_sha256':source_hash(),'scenarios':selected,'checkpoints':{k:getattr(v,'checkpoint_sha256',None) for k,v in agents.items()},'bootstrap_seed':415,'bootstrap_samples':5000}
    save(output/'frozen_manifest.json',manifest)
    records=[]
    for method,policy in agents.items():
        for scene in selected:
            r=rollout(cfg,scene['seed'],scene['course'],output/'episodes'/method/str(scene['seed']),scene=scene,policy=policy)
            r['training_seed']=None if method=='classical' else 0
            records.append(r)
        print(method,'completed',flush=True)
    (output/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary={}
    for method,policy in agents.items():
        rows=[r for r in records if r['method']==policy.method] if hasattr(policy,'method') else [r for r in records if r['method']=='classical']
        def metrics(rs):
            successes=[r for r in rs if r['team_success']]
            return {'episodes':len(rs),'successes':len(successes),'collisions':sum(r['collision'] is not None for r in rs),'timeouts':sum(r['timeout'] for r in rs),'capped_completion_s':float(np.mean([r['capped_completion_s'] for r in rs])),'successful_makespan_s':float(np.mean([r['makespan_s'] for r in successes])) if successes else None,'per_robot_arrivals':sum(sum(r['per_robot_success']) for r in rs),'robot_attempts':sum(len(r['per_robot_success']) for r in rs),'mean_inference_ms':float(np.mean([r['controller_mean_ms'] for r in rs]))}
        summary[method]=metrics(rows)
        summary[method]['courses']={c:metrics([r for r in rows if r['course']==c]) for c in cfg['courses']}
    rng=np.random.default_rng(415)
    idx=rng.integers(0,len(selected),(5000,len(selected)))
    paired={}
    keys=list(agents)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            aa=np.array([r['team_success'] for r in records[i*40:(i+1)*40]],float)
            j=keys.index(b);bb=np.array([r['team_success'] for r in records[j*40:(j+1)*40]],float)
            paired[a+' minus '+b]=np.quantile((aa-bb)[idx].mean(1),[.025,.975]).tolist()
    save(output/'summary.json',{'methods':summary,'paired_success_difference_95_percentile_intervals':paired,'limitation':'One training seed; scenario bootstrap does not measure training variability. No additional training or tuning.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(keys,[summary[k]['successes']/40 for k in keys]);axes[0].set(title='Held-out team success (40 layouts)',ylim=(0,1))
    axes[1].bar(keys,[summary[k]['capped_completion_s'] for k in keys]);axes[1].set(title='Capped completion (lower is better)',ylabel='seconds')
    fig.tight_layout();fig.savefig(output/'comparison.png');plt.close(fig)
    print(json.dumps(summary),flush=True)


def demo(cfg,agents,output):
    from .bullet_world import BulletWorld
    import imageio.v2 as imageio
    from .scenarios import generate
    scene=generate(41000,'open')
    scene['course']='3d-demo-separated-lanes'
    scene['obstacles']=[{'kind':'circle','center':[4,3.5],'radius':.4},{'kind':'circle','center':[6,6.5],'radius':.4}]
    # Integration-only geometry, designated before running; never pooled with G4.
    save(output/'scene.json',scene)
    save(output/'backend.json',{'engine':'pybullet 3.2.7 DIRECT','gravity':0,'robot_mass_kg':1,'body':'radius .2m cylinder, height .24m','drive':'ideal body-frame velocity servo, decorative wheels, no wheel/friction model','sensors':'analytic planar geometry from measured Bullet poses; cylinder/box footprints match world','tie_rule':'finish times within 0.1s share rank; robot ID orders display','control_dt_s':.1,'physics_dt_s':.01,'source_sha256':source_hash(),'tolerance_m':1e-6})
    all_results=[]
    for method,policy in agents.items():
        out=output/method;out.mkdir(parents=True,exist_ok=True)
        world=BulletWorld(cfg)
        try:
            obs=world.reset(scene=scene);policy.reset(41000)
            logs=[];lat=[]
            with imageio.get_writer(out/'race.mp4',fps=10,codec='libx264',macro_block_size=1) as video:
                video.append_data(np.asarray(world.camera(method)))
                while not(world.terminated or world.truncated):
                    start=time.perf_counter();actions=policy.act(obs);lat.append(time.perf_counter()-start)
                    before=obs.copy();obs,*_=world.step(actions)
                    logs.append({'time_s':world.time,'state':world.state.tolist(),'observations':before.tolist(),'actions':actions.tolist(),'contact':world.contact})
                    video.append_data(np.asarray(world.camera(method)))
                world.camera(method).save(out/'final.png')
            result=world.summary();result.update(method=method,checkpoint_sha256=getattr(policy,'checkpoint_sha256',None),mean_inference_ms=float(np.mean(lat)*1000),backend='pybullet-3d',seed=41000)
            result['ranks']=[None if t is None else 1+sum(u is not None and u<t-.1 for u in world.finish_times) for t in world.finish_times]
            save(out/'episode.json',result)
            (out/'steps.jsonl').write_text(''.join(json.dumps(l)+'\n' for l in logs))
            all_results.append(result)
            print(json.dumps(result),flush=True)
        finally:world.close()
    save(output/'transfer_summary.json',all_results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['compare','demo']);args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/g0_contract.json').read_text());torch.set_num_threads(4)
    agents=policies(cfg)
    if args.command=='compare':compare(cfg,agents,ROOT/'artifacts/g4')
    else:demo(cfg,agents,ROOT/'artifacts/g4b')

if __name__=='__main__':main()
