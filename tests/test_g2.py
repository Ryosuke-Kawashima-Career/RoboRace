import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from roborace.bc import BehaviorCloning, PolicyNet, normalize
from roborace.env import World

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'configs/g0_contract.json').read_text())
CONTRACT = hashlib.sha256(json.dumps(CFG, sort_keys=True).encode()).hexdigest()


class BehaviorCloningTests(unittest.TestCase):
    def checkpoint(self, folder):
        torch.manual_seed(99)
        model = PolicyNet([16])
        data = {'state_dict': model.state_dict(), 'hidden_sizes': [16],
                'observation_schema': 'roborace-v1', 'observation_dimension': 44,
                'environment_sha256': CONTRACT, 'normalization_mean': [0.1] * 44,
                'normalization_std': [0.5] * 44, 'arrival_stop_distance_m': .27}
        path = Path(folder) / 'policy.pt'
        torch.save(data, path)
        return path

    def test_checkpoint_reload_and_action_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.checkpoint(folder)
            a, b = BehaviorCloning(path, CONTRACT), BehaviorCloning(path)
            obs = World(CFG).reset(20000)
            actions = a.act(obs)
            np.testing.assert_array_equal(actions, b.act(obs))
            self.assertTrue(((actions[:, 0] > 0) & (actions[:, 0] < 1)).all())
            self.assertTrue((abs(actions[:, 1]) < 2).all())
            with self.assertRaises(ValueError):
                BehaviorCloning(path, 'wrong-contract')
            with self.assertRaises(ValueError):
                a.act(np.full((3, 44), np.nan))
            with self.assertRaises(ValueError):
                a.act(np.zeros((3, 43)))

    def test_learned_weights_determine_non_arrival_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            policy = BehaviorCloning(self.checkpoint(folder))
            obs = World(CFG).reset(20000)
            before = policy.act(obs)
            with torch.no_grad():
                policy.model.net[-1].bias.add_(1)
            self.assertFalse(np.array_equal(before, policy.act(obs)))
            near = obs.copy()
            near[:, 0] = .2 / np.sqrt(200)
            np.testing.assert_array_equal(policy.act(near), np.zeros((3, 2)))

    def test_missing_peer_mask_is_preserved(self):
        x = np.ones((2, 44), dtype=np.float32)
        x[:, 39:] = 0
        transformed = normalize(x, np.full(44, .4), np.full(44, .5))
        np.testing.assert_array_equal(transformed[:, 39:], np.zeros((2, 5)))
        np.testing.assert_array_equal(transformed[:, 33], np.ones(2))


if __name__ == '__main__':
    unittest.main()
