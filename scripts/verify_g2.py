"""Verify G2 split/filter integrity, trained checkpoint inference and replay."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roborace.bc import BehaviorCloning
from roborace.env import World


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    dataset_root = ROOT / 'artifacts/g2/dataset'
    manifest = json.loads((dataset_root / 'manifest.json').read_text())
    training = json.loads((ROOT / 'artifacts/g2/bc_seed0/training_summary.json').read_text())
    checkpoint = ROOT / 'artifacts/g2/bc_seed0/policy.pt'
    assert sha(dataset_root / 'demonstrations.npz') == manifest['dataset_sha256'] == training['dataset_sha256']
    assert sha(checkpoint) == training['checkpoint_sha256']
    assert manifest['split'] == 'train'
    allowed = {s['seed'] for s in json.loads((ROOT / 'scenarios/train/manifest.json').read_text())}
    assert all(record['layout_seed'] in allowed for record in manifest['records'])
    data = np.load(dataset_root / 'demonstrations.npz')
    ids, robots = data['episode_ids'], data['robot_ids']
    assert not set(training['train_episode_ids']) & set(training['holdout_episode_ids'])
    assert set(np.unique(ids)) == set(training['train_episode_ids']) | set(training['holdout_episode_ids'])
    for episode, robot in set(zip(ids.tolist(), robots.tolist())):
        record = manifest['records'][episode]
        assert record['summary']['collision'] is None
        assert record['summary']['per_robot_success'][robot]
        assert robot in record['successful_robot_ids_retained']
    fit = ids % 5 != 0
    policy = BehaviorCloning(checkpoint, manifest['environment_sha256'])
    np.testing.assert_array_equal(data['observations'][fit].mean(axis=0), policy.mean)
    np.testing.assert_array_equal(np.maximum(data['observations'][fit].std(axis=0), .05), policy.std)
    assert sha(ROOT / 'scenarios/train/manifest.json') == manifest['training_manifest_sha256']
    frozen = json.loads((ROOT / 'artifacts/g1/scenario_validation.json').read_text())
    for split, expected in frozen.items():
        assert sha(ROOT / 'scenarios' / split / 'manifest.json') == expected['sha256']
    results = {'dataset': {'checksum_valid': True, 'split_filter_valid': True, 'normalization_train_only': True,
                           'retained_samples': len(ids), 'scenario_manifests_unchanged': True}}
    for name in ('demo_open', 'demo_slalom_failure'):
        folder = ROOT / 'artifacts/g2' / name
        trace = json.loads((folder / 'trajectory.json').read_text())
        episode = json.loads((folder / 'episode.json').read_text())
        cfg = json.loads((folder / 'config.json').read_text())
        world = World(cfg)
        obs = world.reset(scene=trace['scene'])
        assert sha(checkpoint) == episode['checkpoint_sha256']
        for action, next_state in zip(trace['actions'], trace['states'][1:]):
            # This checks saved actions actually came from the learned checkpoint,
            # not merely that arbitrary actions replay through the simulator.
            np.testing.assert_array_equal(policy.act(obs), action)
            obs, _, _, _, _ = world.step(action)
            np.testing.assert_array_equal(world.state, next_state)
        np.testing.assert_array_equal(world.summary()['per_robot_success'], episode['per_robot_success'])
        results[name] = {'learned_action_match': True, 'exact_replay': True, 'team_success': episode['team_success']}
    from PIL import Image
    media = {}
    for name in ('demo_open', 'demo_slalom_failure'):
        with Image.open(ROOT / 'artifacts/g2' / name / 'rollout.gif') as im:
            duration = 0
            for i in range(im.n_frames):
                im.seek(i)
                duration += im.info.get('duration', 0)
            media[name] = {'frames': im.n_frames, 'duration_ms': duration}
    results['recordings'] = media
    (ROOT / 'artifacts/g2/verification.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
