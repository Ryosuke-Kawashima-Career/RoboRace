"""G1 commands: python -m roborace.cli demo / evaluate / scenarios."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from .controller import Classical
from .env import World
from .render import frame
from .scenarios import COURSES, build_manifests

ROOT = Path(__file__).resolve().parents[1]


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'roborace').glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rollout(cfg, seed, course, output, recording=False, scene=None, policy=None):
    output.mkdir(parents=True, exist_ok=True)
    digest = source_hash()
    snapshot = ROOT / 'artifacts/g1/source_snapshots' / digest
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / 'roborace').glob('*.py'):
        (snapshot / source.name).write_bytes(source.read_bytes())
    (output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    world = World(cfg)
    policy = policy if policy is not None else Classical()
    method = getattr(policy, 'method', 'classical')
    world.controller_name = method
    observation = world.reset(seed, course, scene)
    policy.reset(seed)
    frames = [frame(world)] if recording else []
    latency = []
    states, actions_log = [world.state.tolist()], []
    start = time.perf_counter()
    while not (world.terminated or world.truncated):
        t = time.perf_counter()
        actions = policy.act(observation)
        latency.append(time.perf_counter() - t)
        observation, _, _, _, _ = world.step(actions)
        states.append(world.state.tolist())
        actions_log.append(actions.tolist())
        if recording and (world.steps % 2 == 0 or world.terminated or world.truncated):
            frames.append(frame(world))
    result = world.summary()
    result.update({'method': method, 'seed': seed, 'course': course,
                   'wall_time_s': time.perf_counter() - start,
                   'controller_mean_ms': float(np.mean(latency) * 1000), 'controller_p95_ms': float(np.percentile(latency, 95) * 1000),
                   'source_sha256': digest, 'scene_sha256': hashlib.sha256(json.dumps(world.scene, sort_keys=True).encode()).hexdigest(),
                   'config_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                   'trajectory_sha256': hashlib.sha256(np.asarray(states, dtype='<f8').tobytes()).hexdigest(),
                   'python': platform.python_version(), 'numpy': np.__version__})
    if hasattr(policy, 'checkpoint_sha256'):
        result['checkpoint_sha256'] = policy.checkpoint_sha256
    (output / 'episode.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (output / 'trajectory.json').write_text(json.dumps({'scene': world.scene, 'states': states, 'actions': actions_log}) + '\n', encoding='utf-8')
    if recording:
        # The last frame's duration represents any final odd control step.
        durations = [200] * len(frames)
        if world.steps % 2:
            durations[-2] = 100
        durations[-1] = 200
        frames[0].save(output / 'rollout.gif', save_all=True, append_images=frames[1:], duration=durations, loop=0)
        frames[-1].save(output / 'final.png')
        (output / 'recording.json').write_text(json.dumps({'format': 'GIF', 'playback_speed': '1x',
            'simulated_duration_s': world.time, 'frame_count': len(frames),
            'requested_duration_ms': sum(durations), 'includes_final_hold_ms': durations[-1]}, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['demo', 'evaluate', 'scenarios', 'collect-bc', 'train-bc', 'train-ppo'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/g0_contract.json')
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--course', choices=COURSES, default='open')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/g1')
    parser.add_argument('--controller', choices=['classical', 'bc', 'ppo'], default='classical')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--bc-config', type=Path, default=ROOT / 'configs/bc.json')
    parser.add_argument('--ppo-config', type=Path, default=ROOT / 'configs/ppo.json')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/g2/dataset')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    policy = None
    if args.controller in ('bc', 'ppo'):
        if args.checkpoint is None:
            parser.error('--controller bc/ppo requires --checkpoint')
        import torch
        from .bc import BehaviorCloning
        torch.set_num_threads(cfg['compute_budget']['torch_threads'])
        contract = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        if args.controller == 'bc':
            policy = BehaviorCloning(args.checkpoint, contract)
        else:
            from .ppo import PPOPolicy
            policy = PPOPolicy(args.checkpoint, contract)
    if args.command == 'collect-bc':
        from .collect_bc import collect
        collect(args.dataset, json.loads(args.bc_config.read_text()), cfg)
    elif args.command == 'train-bc':
        from .train_bc import train
        train(args.dataset, args.output, json.loads(args.bc_config.read_text()), cfg)
    elif args.command == 'train-ppo':
        from .train_ppo import train
        train(args.output, json.loads(args.ppo_config.read_text()), cfg)
    elif args.command == 'scenarios':
        summary = build_manifests(ROOT / 'scenarios', cfg)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'scenario_validation.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary, indent=2))
    elif args.command == 'demo':
        print(json.dumps(rollout(cfg, args.seed, args.course, args.output, True, policy=policy), indent=2))
    else:
        # G1 evaluates only validation layouts, never the held-out test manifest.
        records = json.loads((ROOT / 'scenarios/validation/manifest.json').read_text())
        results = [rollout(cfg, s['seed'], s['course'], args.output / str(s['seed']), scene=s, policy=policy) for s in records[:8]]
        summary = {'split': 'validation', 'episodes': len(results),
                   'successes': sum(r['team_success'] for r in results),
                   'collision_episodes': sum(r['collision'] is not None for r in results),
                   'timeouts': sum(r['timeout'] for r in results),
                   'note': 'Validation integration sample, not frozen held-out method comparison', 'results': results}
        (args.output / 'validation_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
