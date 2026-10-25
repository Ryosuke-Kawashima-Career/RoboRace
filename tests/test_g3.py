import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from roborace.ppo import ActorCritic, PPOPolicy, gae, clipped_surrogate, physical_actions
from roborace.env import World

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'configs/g0_contract.json').read_text())


class PPOTests(unittest.TestCase):
    def test_terminal_versus_time_limit_bootstrap(self):
        rewards, values, next_values = [np.array([[x]], dtype=np.float32) for x in (1, 2, 10)]
        terminal = np.array([[True]])
        adv, ret = gae(rewards, values, next_values, terminal, terminal, gamma=.9)
        self.assertAlmostEqual(float(adv[0, 0]), -1)
        self.assertAlmostEqual(float(ret[0, 0]), 1)
        adv, ret = gae(rewards, values, next_values, ~terminal, terminal, gamma=.9)
        self.assertAlmostEqual(float(adv[0, 0]), 8)
        self.assertAlmostEqual(float(ret[0, 0]), 10)

    def test_no_advantage_leak_across_reset(self):
        rewards = np.array([[1], [100]], dtype=np.float32)
        zeros = np.zeros_like(rewards)
        boundary = np.array([[True], [False]])
        adv, _ = gae(rewards, zeros, zeros, np.zeros_like(boundary), boundary, gamma=1, lam=1)
        np.testing.assert_array_equal(adv, rewards)

    def test_clipped_objective_both_signs(self):
        loss = clipped_surrogate(torch.log(torch.tensor([2., .5])), torch.zeros(2), torch.tensor([1., -1.]), .2)
        # min(2,1.2) for positive A and min(-.5,-.8) for negative A.
        self.assertAlmostEqual(float(loss), -.2, places=6)

    def test_distribution_log_probs_and_gradient(self):
        torch.manual_seed(4)
        model = ActorCritic([16])
        obs = torch.from_numpy(World(CFG).reset(10000))
        distribution, values = model.distribution_value(obs)
        unit = distribution.sample()
        old = distribution.log_prob(unit).sum(-1).detach()
        recomputed = model.distribution_value(obs)[0].log_prob(unit).sum(-1)
        np.testing.assert_array_equal(old.numpy(), recomputed.detach().numpy())
        action = physical_actions(unit)
        self.assertTrue(((action[:, 0] >= 0) & (action[:, 0] <= 1)).all())
        self.assertTrue((action[:, 1].abs() <= 2).all())
        loss = clipped_surrogate(recomputed, old, torch.tensor([1., -.5, .2])) + values.square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.actor.parameters()), 0)

    def test_checkpoint_contract_bounds_and_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ppo.pt'
            torch.save({'state_dict': ActorCritic([16]).state_dict(), 'hidden_sizes': [16],
                        'observation_schema': 'roborace-v1', 'observation_dimension': 44,
                        'environment_sha256': 'expected', 'arrival_stop_distance_m': .27}, path)
            a, b = PPOPolicy(path, 'expected'), PPOPolicy(path)
            obs = World(CFG).reset(10000)
            np.testing.assert_array_equal(a.act(obs), b.act(obs))
            obs[:, 0] = 0
            np.testing.assert_array_equal(a.act(obs), np.zeros((3, 2)))
            with self.assertRaises(ValueError):
                PPOPolicy(path, 'bad-contract')
            with self.assertRaises(ValueError):
                a.act(np.zeros((3, 43)))


if __name__ == '__main__':
    unittest.main()
