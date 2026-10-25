"""Supervised learning with episode-separated holdout and closed-loop selection."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import torch
from .bc import BehaviorCloning, PolicyNet, normalize

ROOT = Path(__file__).resolve().parents[1]


def save_checkpoint(path, model, mean, std, config, cfg, manifest, epoch, digest):
    torch.save({'state_dict': model.state_dict(), 'hidden_sizes': config['hidden_sizes'],
                'normalization_mean': mean.tolist(), 'normalization_std': std.tolist(),
                'observation_schema': cfg['schema_version'], 'observation_dimension': 44,
                'action_scales': [1.0, 2.0], 'arrival_stop_distance_m': config['arrival_stop_distance_m'],
                'epoch': epoch, 'training_seed': config['training_seed'], 'training_config': config,
                'environment_sha256': manifest['environment_sha256'], 'dataset_sha256': manifest['dataset_sha256'],
                'training_manifest_sha256': manifest['training_manifest_sha256'], 'source_sha256': digest,
                'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__}, path)


def train(dataset_folder, output, config, cfg):
    from .cli import rollout, source_hash
    torch.set_num_threads(config['torch_threads'])
    torch.manual_seed(config['training_seed'])
    np.random.seed(config['training_seed'])
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset_folder = Path(dataset_folder)
    manifest = json.loads((dataset_folder / 'manifest.json').read_text())
    dataset_file = dataset_folder / 'demonstrations.npz'
    if hashlib.sha256(dataset_file.read_bytes()).hexdigest() != manifest['dataset_sha256']:
        raise ValueError('Dataset checksum mismatch')
    if manifest['split'] != 'train' or manifest['environment_sha256'] != hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest():
        raise ValueError('Training data split/environment mismatch')
    data = np.load(dataset_file)
    x, actions, ids = data['observations'], data['actions'], data['episode_ids']
    train_mask = ids % 5 != 0
    if not train_mask.any() or train_mask.all():
        raise ValueError('Need nonempty episode-separated fit and holdout partitions')
    mean = x[train_mask].mean(axis=0)
    std = np.maximum(x[train_mask].std(axis=0), 0.05)
    x_tensor = torch.from_numpy(normalize(x, mean, std))
    target = torch.from_numpy(actions / np.array([1, 2], dtype=np.float32))
    train_indices = np.flatnonzero(train_mask)
    holdout_indices = np.flatnonzero(~train_mask)
    weights = 1 + 2 * (target[:, 1].abs() > 0.1).float()
    model = PolicyNet(config['hidden_sizes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    validation_file = ROOT / 'scenarios/validation/manifest.json'
    validation_scenes = json.loads(validation_file.read_text())[:4]
    history, best_score, best_epoch = [], None, None
    digest = source_hash()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        permutation = np.random.permutation(train_indices)
        for i in range(0, len(permutation), config['batch_size']):
            batch = permutation[i:i+config['batch_size']]
            prediction = model(x_tensor[batch])
            per_sample = ((prediction - target[batch]) ** 2).mean(dim=1)
            loss = (per_sample * weights[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            errors = ((model(x_tensor) - target) ** 2).mean(dim=1)
            row = {'epoch': epoch, 'train_mse': float(errors[train_mask].mean()),
                   'holdout_mse': float(errors[~train_mask].mean()), 'elapsed_wall_s': time.perf_counter() - start}
        if epoch % config['validation_interval_epochs'] == 0 or epoch == config['epochs']:
            checkpoint = output / f'epoch_{epoch:03d}.pt'
            save_checkpoint(checkpoint, model, mean, std, config, cfg, manifest, epoch, digest)
            policy = BehaviorCloning(checkpoint, manifest['environment_sha256'])
            results = [rollout(cfg, s['seed'], s['course'], output / f'validation_epoch_{epoch:03d}' / str(s['seed']), scene=s, policy=policy) for s in validation_scenes]
            successes = sum(r['team_success'] for r in results)
            capped = sum(r['capped_completion_s'] for r in results) / len(results)
            row.update({'validation_successes': successes, 'validation_episodes': len(results), 'validation_capped_s': capped,
                        'validation_collisions': sum(r['collision'] is not None for r in results)})
            score = (successes, -capped, -row['holdout_mse'])
            if best_score is None or score > best_score:
                best_score, best_epoch = score, epoch
                shutil.copyfile(checkpoint, output / 'policy.pt')
        history.append(row)
        print(json.dumps(row), flush=True)
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n', encoding='utf-8')
        if time.perf_counter() - start >= config['wall_clock_limit_s']:
            break
    if best_epoch is None:
        raise RuntimeError('Budget expired before any checkpoint was validated')
    policy = BehaviorCloning(output / 'policy.pt', manifest['environment_sha256'])
    with torch.no_grad():
        before = policy.act(x[:64])
        after = BehaviorCloning(output / 'policy.pt').act(x[:64])
    np.testing.assert_array_equal(before, after)
    summary = {'method': policy.method, 'best_epoch': best_epoch, 'best_score': list(best_score),
               'train_samples': len(train_indices), 'holdout_samples': len(holdout_indices),
               'train_episode_ids': np.unique(ids[train_mask]).tolist(), 'holdout_episode_ids': np.unique(ids[~train_mask]).tolist(),
               'normalization_fit': 'training episodes only, excludes internal holdout and all validation/test data',
               'loss': 'MSE over [linear_speed, angular_speed/2], weight 3 for abs(angular_speed)>0.2 rad/s else 1',
               'validation_layout_seeds': [s['seed'] for s in validation_scenes],
               'validation_manifest_sha256': hashlib.sha256(validation_file.read_bytes()).hexdigest(),
               'checkpoint_sha256': policy.checkpoint_sha256, 'dataset_sha256': manifest['dataset_sha256'],
               'source_sha256': digest, 'config': config, 'completed_epochs': len(history),
               'wall_time_s': time.perf_counter() - start, 'exact_reload': True,
               'collection_wall_time_s': manifest['wall_time_s'],
               'note': 'G2 validation selection, not held-out test performance; seed 0 only; other training seeds remain for G4'}
    (output / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    # Standard plot artifact; plotting does not affect training or selection.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r['epoch'] for r in history], [r['train_mse'] for r in history], label='fit episodes')
    ax.plot([r['epoch'] for r in history], [r['holdout_mse'] for r in history], label='internal holdout episodes')
    ax.set(xlabel='Epoch', ylabel='Unweighted normalized-action MSE', title='Behavior cloning (training seed 0)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'loss.png', dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
