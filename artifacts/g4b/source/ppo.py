"""Parameter-shared independent PPO with bounded Beta actions and local critic."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Beta


class ActorCritic(nn.Module):
    def __init__(self, hidden_sizes=(64, 64)):
        super().__init__()
        def network(output):
            layers, size = [], 44
            for width in hidden_sizes:
                layers.extend((nn.Linear(size, width), nn.Tanh()))
                size = width
            layers.append(nn.Linear(size, output))
            return nn.Sequential(*layers)
        self.actor, self.critic = network(4), network(1)
        # Symmetric bounded exploration: initial mean speed .5, angular speed 0.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def distribution_value(self, observations):
        raw = self.actor(observations)
        alpha, beta = (torch.nn.functional.softplus(v) + 1 for v in raw.chunk(2, dim=-1))
        return Beta(alpha, beta), self.critic(observations).squeeze(-1)


def physical_actions(unit):
    return torch.stack((unit[..., 0], 4 * unit[..., 1] - 2), dim=-1)


def gae(rewards, values, next_values, terminals, boundaries, gamma=.99, lam=.95):
    """T x N GAE: true terminals disable bootstrap; resets disable recursion.

    Time limits bootstrap their final-state values but never carry advantage into
    the newly reset episode. Robot arrival is terminal for that agent.
    """
    advantage = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminals[t]) - values[t]
        carry = delta + gamma * lam * (~boundaries[t]) * carry
        advantage[t] = carry
    return advantage, advantage + values


def clipped_surrogate(new_log_prob, old_log_prob, advantage, clip=.2):
    ratio = torch.exp(new_log_prob - old_log_prob)
    return -torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage).mean()


class PPOPolicy:
    method = 'independent_ppo_with_arrival_stop'

    def __init__(self, checkpoint, expected_contract=None):
        path = Path(checkpoint)
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['observation_schema'] != 'roborace-v1' or data['observation_dimension'] != 44:
            raise ValueError('Incompatible PPO observation schema')
        if expected_contract and expected_contract != data['environment_sha256']:
            raise ValueError('PPO environment mismatch')
        self.model = ActorCritic(data['hidden_sizes'])
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()
        self.stop_distance = data['arrival_stop_distance_m']
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.metadata = {k: v for k, v in data.items() if k != 'state_dict'}

    def reset(self, seed=0):
        pass

    def act(self, observations):
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != 44 or not np.isfinite(obs).all():
            raise ValueError('Expected finite N x 44 observations')
        with torch.no_grad():
            distribution, _ = self.model.distribution_value(torch.from_numpy(obs))
            actions = physical_actions(distribution.mean).numpy()
        actions[obs[:, 0] * np.sqrt(200) <= self.stop_distance] = 0
        return actions
