"""Collect filtered training-only demonstrations, preserving failure statistics."""
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .controller import Classical
from .env import World

ROOT = Path(__file__).resolve().parents[1]


def collect(output, config, cfg):
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'scenarios/train/manifest.json'
    scenes = json.loads(manifest_path.read_text())[:config['dataset_layouts']]
    observations, targets, episode_ids, robot_ids = [], [], [], []
    records = []
    rng = np.random.default_rng(config['collection_seed'])
    total_world_steps = 0
    for episode_id, original in enumerate(scenes):
        scene = copy.deepcopy(original)
        offsets = rng.uniform(-config['heading_perturbation_rad'], config['heading_perturbation_rad'], 3)
        for robot, offset in zip(scene['starts'], offsets):
            robot[2] += float(offset)
        world, expert = World(cfg), Classical()
        obs = world.reset(scene=scene)
        expert.reset(scene['seed'])
        buffers = [[] for _ in range(world.n)]
        attempted = 0
        while not (world.terminated or world.truncated):
            actions = expert.act(obs)
            for i in range(world.n):
                if not world.finished[i]:
                    buffers[i].append((obs[i].copy(), actions[i].copy()))
                    attempted += 1
            obs, _, _, _, _ = world.step(actions)
        total_world_steps += world.steps
        retained = 0
        successful_robots = []
        if not world.contact:
            for i in range(world.n):
                if world.finished[i]:
                    successful_robots.append(i)
                    for observation, action in buffers[i]:
                        observations.append(observation)
                        targets.append(action)
                        episode_ids.append(episode_id)
                        robot_ids.append(i)
                        retained += 1
        records.append({'episode_id': episode_id, 'layout_seed': scene['seed'], 'course': scene['course'],
                        'heading_offsets_rad': offsets.tolist(), 'successful_robot_ids_retained': successful_robots,
                        'attempted_agent_samples': attempted, 'retained_agent_samples': retained, 'summary': world.summary()})
        print(f"collect {episode_id+1}/{len(scenes)} {scene['course']}: retain {retained}/{attempted}", flush=True)
    if not observations:
        raise RuntimeError('No successful expert trajectories')
    dataset = output / 'demonstrations.npz'
    np.savez_compressed(dataset, observations=np.asarray(observations, dtype=np.float32),
                        actions=np.asarray(targets, dtype=np.float32), episode_ids=np.asarray(episode_ids), robot_ids=np.asarray(robot_ids))
    from .cli import source_hash
    digest = source_hash()
    snapshot = output / 'source' / digest
    snapshot.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / 'roborace').glob('*.py'):
        (snapshot / p.name).write_bytes(p.read_bytes())
    report = {'split': 'train', 'filter': config['filter'], 'records': records, 'collection_config': config,
              'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(), 'source_sha256': digest,
              'training_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'environment_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
              'wall_time_s': time.perf_counter() - start, 'world_steps': total_world_steps,
              'retained_agent_samples': len(observations), 'attempted_agent_samples': sum(r['attempted_agent_samples'] for r in records),
              'course_retained_samples': {course: sum(r['retained_agent_samples'] for r in records if r['course'] == course) for course in cfg['courses']}}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'collection_config')}, indent=2), flush=True)
    return report
