"""On-policy PPO pilot. No expert/BC initialization or privileged observations."""
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import torch
from .env import World
from .ppo import ActorCritic, PPOPolicy, gae, clipped_surrogate, physical_actions

ROOT = Path(__file__).resolve().parents[1]


def save(path, model, cfg, config, steps, updates, digest, manifest_hash):
    torch.save({'state_dict': model.state_dict(), 'hidden_sizes': config['hidden_sizes'],
                'observation_schema': cfg['schema_version'], 'observation_dimension': 44,
                'environment_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                'source_sha256': digest, 'training_manifest_sha256': manifest_hash,
                'arrival_stop_distance_m': config['arrival_stop_distance_m'], 'training_config': config,
                'normalization': 'frozen physical scales from environment; no fitted observation statistics',
                'distribution': 'independent Beta over [0,1]^2; v=u0, omega=4*u1-2',
                'world_steps': steps, 'updates': updates, 'training_seed': config['training_seed'],
                'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__}, path)


def train(output, config, cfg):
    from .cli import rollout, source_hash
    torch.set_num_threads(config['torch_threads'])
    torch.manual_seed(config['training_seed'])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config['training_seed'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_path = ROOT / 'scenarios/train/manifest.json'
    scenes = json.loads(train_path.read_text())
    validation = json.loads((ROOT / 'scenarios/validation/manifest.json').read_text())[:4]
    manifest_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    digest = source_hash()
    snapshot = output / 'source' / digest
    snapshot.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / 'roborace').glob('*.py'):
        (snapshot / p.name).write_bytes(p.read_bytes())
    model = ActorCritic(config['hidden_sizes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    world = World(cfg)
    order = list(range(len(scenes)))
    rng.shuffle(order)
    cursor, used_seeds = 0, []
    def reset():
        nonlocal cursor
        if cursor and cursor % len(order) == 0:
            rng.shuffle(order)
        s = scenes[order[cursor % len(order)]]
        cursor += 1
        used_seeds.append(s['seed'])
        return world.reset(scene=s)
    # Single-robot interface preflight, excluded from policy training/sample count.
    one = copy.deepcopy(scenes[0])
    one['starts'], one['goals'] = one['starts'][:1], one['goals'][:1]
    debug = World(cfg)
    debug.reset(scene=one)
    for _ in range(8):
        debug.step([[.1, 0]])
    obs = reset()
    history, episodes = [], []
    steps, updates, agent_samples, actor_samples = 0, 0, 0, 0
    best_score, best_update = None, None
    save(output / 'initial.pt', model, cfg, config, 0, 0, digest, manifest_hash)
    weights_before = torch.cat([p.detach().flatten() for p in model.actor.parameters()]).clone()
    while steps < config['max_world_steps'] and time.perf_counter() - started < config['wall_clock_limit_s']:
        buffer = {k: [] for k in ('obs', 'unit', 'logp', 'values', 'next_values', 'rewards', 'terminals', 'boundaries', 'active', 'actor_active')}
        for _ in range(min(config['rollout_steps'], config['max_world_steps'] - steps)):
            active = ~world.finished.copy()
            actor_active = active & (obs[:, 0] * np.sqrt(200) > config['arrival_stop_distance_m'])
            with torch.no_grad():
                distribution, values = model.distribution_value(torch.from_numpy(obs))
                unit = distribution.sample()
                logp = distribution.log_prob(unit).sum(-1)
                actions = physical_actions(unit).numpy()
            actions[~actor_active] = 0
            before_dist = np.linalg.norm(world.goals - world.state[:, :2], axis=1)
            before_finished, previous = world.finished.copy(), world.velocity.copy()
            following, _, terminated, truncated, info = world.step(actions)
            after_dist = np.linalg.norm(world.goals - world.state[:, :2], axis=1)
            r = config['reward']
            reward = r['progress'] * (before_dist - after_dist) + r['time']
            reward += r['arrival'] * (world.finished & ~before_finished)
            reward += r['smoothness'] * ((actions - previous) ** 2).sum(axis=1)
            if world.contact:
                reward += r['collision']
            reward[~active] = 0
            with torch.no_grad():
                _, next_values = model.distribution_value(torch.from_numpy(following))
            terminals = world.finished.copy() | terminated
            boundaries = terminals | truncated
            for key, value in {'obs': obs.copy(), 'unit': unit.numpy(), 'logp': logp.numpy(), 'values': values.numpy(),
                               'next_values': next_values.numpy(), 'rewards': reward.astype(np.float32), 'terminals': terminals,
                               'boundaries': boundaries, 'active': active, 'actor_active': actor_active}.items():
                buffer[key].append(value)
            steps += 1
            agent_samples += int(active.sum())
            actor_samples += int(actor_active.sum())
            obs = following
            if terminated or truncated:
                episodes.append({'world_step': steps, 'layout_seed': world.scene['seed'], 'course': world.scene['course'], 'summary': info})
                obs = reset()
        arrays = {k: np.asarray(v) for k, v in buffer.items()}
        advantages, returns = gae(arrays['rewards'], arrays['values'], arrays['next_values'], arrays['terminals'], arrays['boundaries'], config['gamma'], config['gae_lambda'])
        flat = {k: torch.from_numpy(v.reshape((-1,) + v.shape[2:])) for k, v in arrays.items()}
        advantage = torch.from_numpy(advantages.reshape(-1))
        returns = torch.from_numpy(returns.reshape(-1))
        eligible = flat['actor_active']
        if eligible.any():
            advantage = (advantage - advantage[eligible].mean()) / (advantage[eligible].std(unbiased=False) + 1e-8)
        indices = np.flatnonzero(flat['active'].numpy())
        diagnostics = []
        for _ in range(config['update_epochs']):
            rng.shuffle(indices)
            for offset in range(0, len(indices), config['minibatch_size']):
                idx = indices[offset:offset+config['minibatch_size']]
                distribution, predicted = model.distribution_value(flat['obs'][idx])
                logp = distribution.log_prob(flat['unit'][idx]).sum(-1)
                mask = flat['actor_active'][idx]
                actor_loss = clipped_surrogate(logp[mask], flat['logp'][idx][mask], advantage[idx][mask], config['clip_ratio']) if mask.any() else predicted.sum() * 0
                entropy = distribution.entropy().sum(-1)[mask].mean() if mask.any() else predicted.sum() * 0
                value_loss = .5 * ((predicted - returns[idx]) ** 2).mean()
                loss = actor_loss + config['value_coefficient'] * value_loss - config['entropy_coefficient'] * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite PPO loss')
                optimizer.zero_grad()
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                if not torch.isfinite(gradient):
                    raise FloatingPointError('Non-finite PPO gradient')
                optimizer.step()
                with torch.no_grad():
                    ratio = torch.exp(logp[mask] - flat['logp'][idx][mask])
                    diagnostics.append([float(actor_loss.detach()), float(value_loss.detach()), float(entropy.detach()),
                                        float((flat['logp'][idx][mask] - logp[mask]).mean()) if mask.any() else 0,
                                        float((abs(ratio-1) > config['clip_ratio']).float().mean()) if mask.any() else 0])
        updates += 1
        stats = np.mean(diagnostics, axis=0)
        row = dict(zip(('policy_loss', 'value_loss', 'entropy_unit_space', 'approx_kl', 'clip_fraction'), stats.tolist()))
        row.update({'update': updates, 'world_steps': steps, 'active_agent_transitions': agent_samples,
                    'policy_agent_transitions': actor_samples, 'mean_rollout_reward': float(arrays['rewards'][arrays['active']].mean()),
                    'completed_training_episodes': len(episodes), 'wall_time_s': time.perf_counter() - started})
        final = steps >= config['max_world_steps'] or time.perf_counter() - started >= config['wall_clock_limit_s']
        if updates % config['validate_every_updates'] == 0 or final:
            checkpoint = output / f'update_{updates:04d}.pt'
            save(checkpoint, model, cfg, config, steps, updates, digest, manifest_hash)
            policy = PPOPolicy(checkpoint)
            results = [rollout(cfg, s['seed'], s['course'], output / f'validation_{updates:04d}' / str(s['seed']), scene=s, policy=policy) for s in validation]
            successes = sum(v['team_success'] for v in results)
            capped = sum(v['capped_completion_s'] for v in results) / len(results)
            collisions = sum(v['collision'] is not None for v in results)
            row.update({'validation_successes': successes, 'validation_episodes': len(results), 'validation_capped_s': capped, 'validation_collisions': collisions})
            score = (successes, -capped, -collisions)
            if best_score is None or score > best_score:
                best_score, best_update = score, updates
                shutil.copyfile(checkpoint, output / 'policy.pt')
        history.append(row)
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        if updates % 8 == 0 or final:
            print(json.dumps(row), flush=True)
    if best_update is None:
        raise RuntimeError('No trained checkpoint validated within budget')
    save(output / 'last.pt', model, cfg, config, steps, updates, digest, manifest_hash)
    torch.save(optimizer.state_dict(), output / 'optimizer_last.pt')
    weights_after = torch.cat([p.detach().flatten() for p in model.actor.parameters()])
    policy = PPOPolicy(output / 'policy.pt')
    summary = {'config': config, 'world_steps': steps, 'active_agent_transitions': agent_samples, 'policy_agent_transitions': actor_samples,
               'world_step_agent_slots': 3 * steps, 'wall_time_s': time.perf_counter() - started,
               'best_update': best_update, 'best_score': list(best_score), 'checkpoint_sha256': policy.checkpoint_sha256,
               'training_layout_seeds': sorted(set(used_seeds)), 'training_manifest_sha256': manifest_hash,
               'source_sha256': digest, 'actor_weight_change_l2': float(torch.linalg.norm(weights_after - weights_before)),
               'training_episode_successes': sum(v['summary']['team_success'] for v in episodes),
               'training_episode_collisions': sum(v['summary']['collision'] is not None for v in episodes),
               'completed_training_episodes': len(episodes), 'single_robot_preflight_steps': 8,
               'note': 'from-scratch, seed 0 bounded pilot; full multi-seed comparison remains G4; no test rollouts'}
    (output / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'training_episodes.json').write_text(json.dumps(episodes, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r['world_steps'] for r in history], [r['mean_rollout_reward'] for r in history])
    axes[0].set(xlabel='World steps', ylabel='Mean shaped reward', title='PPO rollout reward (not success)')
    validated = [r for r in history if 'validation_successes' in r]
    axes[1].plot([r['world_steps'] for r in validated], [r['validation_successes']/r['validation_episodes'] for r in validated], marker='o')
    axes[1].set(xlabel='World steps', ylabel='Team success fraction', ylim=(-.05, 1.05), title='Checkpoint selection validation')
    fig.tight_layout()
    fig.savefig(output / 'learning.png', dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
