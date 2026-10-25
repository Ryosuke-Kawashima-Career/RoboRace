"""Small release smoke: CPU checkpoint loading, real 3D stepping and frame capture."""
import json,hashlib,platform,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from roborace.bullet_world import BulletWorld
from roborace.ppo import PPOPolicy
ROOT=Path(__file__).resolve().parents[1]

def run():
    cfg=json.loads((ROOT/'configs/g0_contract.json').read_text())
    scene=json.loads((ROOT/'artifacts/g4b/scene.json').read_text())
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(4)
    policy=PPOPolicy(ROOT/'artifacts/g3/ppo_seed0/policy.pt',digest)
    world=BulletWorld(cfg)
    try:
        obs=world.reset(scene=scene);policy.reset(41000)
        while not(world.terminated or world.truncated):obs,*_=world.step(policy.act(obs))
        result=world.summary()
        assert result['team_success'] and result['collision'] is None
        assert abs(result['makespan_s']-15.9)<.11
        frame=world.camera('PPO / isolated release verification')
        out=ROOT/'artifacts/g5/clean_smoke';out.mkdir(parents=True,exist_ok=True)
        frame.save(out/'final.png')
        result.update(python=platform.python_version(),torch=torch.__version__,observation_dimension=obs.shape[1],camera_size=frame.size,isolated_environment=sys.prefix!=sys.base_prefix,checkpoint_sha256=policy.checkpoint_sha256)
        (out/'episode.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
        return result
    finally:world.close()
if __name__=='__main__':run()
