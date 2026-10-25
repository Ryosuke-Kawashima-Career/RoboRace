"""Replay final recordings and verify source/scene/config/manifest provenance."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roborace.env import World


def main():
    checks = {}
    for name in ('demo_open_final', 'demo_crossing'):
        folder = ROOT / 'artifacts/g1' / name
        episode = json.loads((folder / 'episode.json').read_text())
        trace = json.loads((folder / 'trajectory.json').read_text())
        cfg = json.loads((folder / 'config.json').read_text())
        world = World(cfg)
        world.reset(scene=trace['scene'])
        states = [world.state.tolist()]
        for action in trace['actions']:
            world.step(action)
            states.append(world.state.tolist())
        np.testing.assert_array_equal(states, trace['states'])
        assert hashlib.sha256(np.asarray(states, dtype='<f8').tobytes()).hexdigest() == episode['trajectory_sha256']
        assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == episode['config_sha256']
        assert hashlib.sha256(json.dumps(trace['scene'], sort_keys=True).encode()).hexdigest() == episode['scene_sha256']
        source = ROOT / 'artifacts/g1/source_snapshots' / episode['source_sha256']
        digest = hashlib.sha256()
        for p in sorted(source.glob('*.py')):
            digest.update(p.name.encode())
            digest.update(p.read_bytes())
        assert digest.hexdigest() == episode['source_sha256']
        assert world.summary() == {k: episode[k] for k in world.summary()}
        checks[name] = {'exact_replay': True, 'provenance': True, 'team_success': episode['team_success']}
    manifest = json.loads((ROOT / 'artifacts/g1/scenario_validation.json').read_text())
    seeds = []
    for split, expected in manifest.items():
        data = (ROOT / 'scenarios' / split / 'manifest.json').read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected['sha256']
        records = json.loads(data)
        assert len(records) == expected['layouts']
        seeds.extend(r['seed'] for r in records)
    assert len(seeds) == len(set(seeds)) == 340
    checks['manifests'] = {'checksums_match': True, 'unique_seeds': 340}
    (ROOT / 'artifacts/g1/replay_verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
