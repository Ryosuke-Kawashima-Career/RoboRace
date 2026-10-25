"""Check PPO training provenance, parameter updates and exact recorded inference."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roborace.ppo import PPOPolicy
from roborace.env import World


def main():
    torch.set_num_threads(4)
    base = ROOT / 'artifacts/g3/ppo_seed0'
    summary = json.loads((base / 'training_summary.json').read_text())
    selected = base / 'policy.pt'
    assert hashlib.sha256(selected.read_bytes()).hexdigest() == summary['checkpoint_sha256']
    policy, initial = PPOPolicy(selected), PPOPolicy(base / 'initial.pt')
    assert policy.metadata['updates'] > 0
    delta = sum(float((a-b).detach().square().sum()) for a,b in zip(policy.model.actor.parameters(), initial.model.actor.parameters())) ** .5
    assert delta > 0, 'Selected actor must actually have trained'
    allowed = {s['seed'] for s in json.loads((ROOT / 'scenarios/train/manifest.json').read_text())}
    assert set(summary['training_layout_seeds']) <= allowed
    frozen = json.loads((ROOT / 'artifacts/g1/scenario_validation.json').read_text())
    for split, evidence in frozen.items():
        assert hashlib.sha256((ROOT / 'scenarios' / split / 'manifest.json').read_bytes()).hexdigest() == evidence['sha256']
    history = json.loads((base / 'history.json').read_text())
    assert all(np.isfinite(r[k]) for r in history for k in ('policy_loss','value_loss','entropy_unit_space','approx_kl','clip_fraction','mean_rollout_reward'))
    source = base / 'source' / summary['source_sha256']
    digest = hashlib.sha256()
    for p in sorted(source.glob('*.py')):
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    assert digest.hexdigest() == summary['source_sha256']
    result = {'training': {'selected_actor_change_l2': delta, 'finite_training_metrics': True,
                           'training_seeds_valid': True, 'unchanged_manifests': True, 'source_snapshot_valid': True,
                           'world_steps': summary['world_steps']}}
    folders = sorted((ROOT / 'artifacts/g3').glob('demo_*'))
    if not folders:
        raise RuntimeError('Missing PPO recordings')
    from PIL import Image
    for folder in folders:
        trace = json.loads((folder / 'trajectory.json').read_text())
        episode = json.loads((folder / 'episode.json').read_text())
        cfg = json.loads((folder / 'config.json').read_text())
        policy = PPOPolicy(selected, hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest())
        assert episode['checkpoint_sha256'] == policy.checkpoint_sha256
        obs = World(cfg)
        observation = obs.reset(scene=trace['scene'])
        for action, next_state in zip(trace['actions'], trace['states'][1:]):
            np.testing.assert_array_equal(policy.act(observation), action)
            observation, _, _, _, _ = obs.step(action)
            np.testing.assert_array_equal(obs.state, next_state)
        assert obs.summary()['team_success'] == episode['team_success']
        with Image.open(folder / 'rollout.gif') as image:
            duration = 0
            for i in range(image.n_frames):
                image.seek(i)
                duration += image.info.get('duration', 0)
        result[folder.name] = {'checkpoint_action_match': True, 'exact_state_replay': True,
                               'team_success': episode['team_success'], 'gif_duration_ms': duration}
    (ROOT / 'artifacts/g3/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
