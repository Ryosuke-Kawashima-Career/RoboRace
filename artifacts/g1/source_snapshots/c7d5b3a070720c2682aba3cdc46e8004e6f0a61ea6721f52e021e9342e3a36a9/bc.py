"""Learned bounded-action MLP, with explicit shared arrival stopping rule."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(self, hidden_sizes=(128, 128, 64)):
        super().__init__()
        layers = []
        dim = 44
        for width in hidden_sizes:
            layers.extend((nn.Linear(dim, width), nn.Tanh()))
            dim = width
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.net(x)
        return torch.stack((torch.sigmoid(raw[:, 0]), torch.tanh(raw[:, 1])), dim=1)


def normalize(x, mean, std):
    result = (x - mean) / std
    # Preserve binary presence flags; missing peer features must remain zero.
    for start in (29, 34, 39):
        result[:, start:start+4] *= x[:, start+4:start+5]
        result[:, start+4] = x[:, start+4]
    return result


class BehaviorCloning:
    method = 'behavior_cloning_with_arrival_stop'

    def __init__(self, checkpoint, expected_contract=None):
        self.path = Path(checkpoint)
        data = torch.load(self.path, map_location='cpu', weights_only=True)
        if data['observation_schema'] != 'roborace-v1' or data['observation_dimension'] != 44:
            raise ValueError('Incompatible observation schema')
        if expected_contract and data['environment_sha256'] != expected_contract:
            raise ValueError('Checkpoint environment contract mismatch')
        self.model = PolicyNet(data['hidden_sizes'])
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()
        self.mean = np.asarray(data['normalization_mean'], dtype=np.float32)
        self.std = np.asarray(data['normalization_std'], dtype=np.float32)
        if self.mean.shape != (44,) or self.std.shape != (44,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError('Invalid normalization')
        self.stop_distance = data['arrival_stop_distance_m']
        self.checkpoint_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.metadata = {k: v for k, v in data.items() if k != 'state_dict'}

    def reset(self, seed=0):
        pass

    def act(self, observations):
        x = np.asarray(observations, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 44 or not np.isfinite(x).all():
            raise ValueError('Expected finite N x 44 observations')
        with torch.no_grad():
            actions = self.model(torch.from_numpy(normalize(x, self.mean, self.std))).numpy().copy()
        actions[:, 1] *= 2
        actions[x[:, 0] * np.sqrt(200) <= self.stop_distance] = 0
        return actions
