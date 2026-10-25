import copy
import json
from pathlib import Path
import unittest
import hashlib
import tempfile
import numpy as np
from roborace.env import World
from roborace.geometry import collision, ray_box, ray_circle, swept_box
from roborace.controller import Classical
from roborace.scenarios import generate, feasible, build_manifests

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'configs/g0_contract.json').read_text())


def scene(starts=None, goals=None, obstacles=None):
    return {'seed': 0, 'course': 'test', 'arena': [10, 10],
            'starts': starts if starts is not None else [[1, 1, 0]],
            'goals': goals if goals is not None else [[9, 1]],
            'obstacles': obstacles or []}


class GeometryTests(unittest.TestCase):
    def test_rays(self):
        self.assertAlmostEqual(ray_circle(np.array([0., 0.]), np.array([1., 0.]), np.array([2., 0.]), .5), 1.5)
        self.assertAlmostEqual(ray_box(np.array([0., 0.]), np.array([1., 0.]), [2, -1, 3, 1]), 2)
        self.assertTrue(np.isinf(ray_box(np.array([0., 2.]), np.array([1., 0.]), [2, -1, 3, 1])))

    def test_thin_obstacle_tunneling(self):
        s = scene(obstacles=[{'kind': 'box', 'box': [2, .5, 2.001, 1.5]}])
        hit = collision(np.array([[1., 1., 0]]), np.array([[3., 1., 0]]), s, .2)
        self.assertEqual(hit['type'], 'obstacle')

    def test_peer_swap_and_wall(self):
        a = np.array([[1., 1., 0], [3., 1., 0]])
        self.assertEqual(collision(a, a[::-1], scene(), .2)['type'], 'peer')
        self.assertEqual(collision(a[:1], np.array([[.1, 1, 0]]), scene(), .2)['type'], 'wall')

    def test_box_corner_clearance(self):
        self.assertFalse(swept_box(np.array([.82, .82]), np.array([.82, .82]), [1, 1, 2, 2], .2))
        self.assertTrue(swept_box(np.array([.9, .9]), np.array([.9, .9]), [1, 1, 2, 2], .2))


class WorldTests(unittest.TestCase):
    def world(self, s=None, cfg=None):
        world = World(cfg or CFG)
        world.reset(scene=s or scene())
        return world

    def test_seed_observations_and_peers(self):
        a, b = World(CFG), World(CFG)
        oa, ob = a.reset(20000), b.reset(20000)
        np.testing.assert_array_equal(oa, ob)
        self.assertEqual(oa.shape, (3, 44))
        self.assertTrue(np.isfinite(oa).all())
        self.assertEqual(list(oa[0, 33::5]), [1, 1, 0])
        np.testing.assert_array_equal(oa[0, -5:], np.zeros(5))
        for _ in range(5):
            a.step(np.tile([.5, .2], (3, 1)))
            b.step(np.tile([.5, .2], (3, 1)))
        np.testing.assert_array_equal(a.state, b.state)

    def test_straight_turn_and_clipping(self):
        w = self.world()
        w.step([[2, 0]])
        np.testing.assert_allclose(w.state[0], [1.1, 1, 0], atol=1e-12)
        w.step([[0, 9]])
        self.assertAlmostEqual(w.state[0, 2], .2)
        for action in ([[np.nan, 0]], [[0, np.inf]], [[0]], [[0, 0], [0, 0]]):
            with self.assertRaises(ValueError):
                w.step(action)

    def test_sensor_range_and_peer_body_frame(self):
        w = self.world(scene(starts=[[1, 1, 0], [3, 1, 0]], goals=[[8, 1], [8, 2]]))
        obs = w.observe()
        self.assertAlmostEqual(float(obs[0, 5]), 1.8 / 5, places=6)
        self.assertAlmostEqual(float(obs[0, 5 + 12]), 1 / 5, places=6)
        np.testing.assert_allclose(obs[0, 29:34], [.4, 0, 0, 0, 1])
        w.velocity[0] = [1, 0]
        self.assertAlmostEqual(float(w.observe()[0, 31]), -.5)

    def test_finished_peer_remains_obstacle(self):
        w = self.world(scene(starts=[[2, 1, 0], [1, 1, 0]], goals=[[2, 1], [9, 1]]))
        for _ in range(5):
            w.step([[0, 0], [0, 0]])
        self.assertTrue(w.finished[0])
        for _ in range(20):
            result = w.step([[0, 0], [1, 0]])[-1]
            if w.terminated:
                break
        self.assertEqual(result['collision']['type'], 'peer')
        self.assertFalse(result['team_success'])

    def test_collision_before_goal_and_no_overlap(self):
        w = self.world(scene(starts=[[1, 1, 0]], goals=[[1.1, 1]], obstacles=[{'kind': 'circle', 'center': [1.5, 1], 'radius': .2}]))
        for _ in range(5):
            _, _, terminated, _, info = w.step([[1, 0]])
            if terminated:
                break
        self.assertIsNotNone(info['collision'])
        self.assertFalse(info['team_success'])
        self.assertLess(w.state[0, 0], 1.1)

    def test_goal_dwell_success_and_finished_stop(self):
        w = self.world(scene(goals=[[1, 1]]))
        for _ in range(4):
            self.assertFalse(w.step([[0, 0]])[2])
        result = w.step([[0, 0]])[-1]
        self.assertTrue(result['team_success'])
        self.assertAlmostEqual(result['capped_completion_s'], .5)
        with self.assertRaises(RuntimeError):
            w.step([[0, 0]])
        w = self.world(scene(starts=[[1, 1, 0], [3, 3, 0]], goals=[[1, 1], [8, 3]]))
        for _ in range(5):
            w.step([[0, 0], [0, 0]])
        w.step([[1, 1], [0, 0]])
        np.testing.assert_allclose(w.state[0], [1, 1, 0])

    def test_timeout_deadlock_and_capped_metric(self):
        cfg = copy.deepcopy(CFG)
        cfg['horizon_steps'] = 4
        cfg['deadlock']['window_steps'] = 2
        w = self.world(cfg=cfg)
        for _ in range(4):
            result = w.step([[0, 0]])[-1]
        self.assertTrue(result['timeout'])
        self.assertTrue(result['deadlock_observed'])
        self.assertFalse(w.terminated)
        self.assertEqual(result['capped_completion_s'], .4)

    def test_simultaneous_update_order(self):
        s = scene(starts=[[1, 1, 0], [3, 3, .4]], goals=[[9, 1], [9, 3]])
        w = self.world(s)
        rev = copy.deepcopy(s)
        rev['starts'].reverse()
        rev['goals'].reverse()
        v = self.world(rev)
        a = np.array([[.5, .2], [.6, -.1]])
        w.step(a)
        v.step(a[::-1])
        np.testing.assert_array_equal(w.state, v.state[::-1])

    def test_baseline_repeatable_and_observation_only(self):
        w = World(CFG)
        obs = w.reset(20000)
        policy = Classical()
        a = policy.act(obs)
        np.testing.assert_array_equal(a, policy.act(obs.copy()))
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(((a[:, 0] >= 0) & (a[:, 0] <= 1)).all())
        self.assertTrue((abs(a[:, 1]) <= 2).all())

    def test_scenario_feasibility(self):
        for course in ('open', 'slalom', 'bottleneck', 'crossing'):
            self.assertTrue(feasible(generate(20000, course)))
        s = scene(obstacles=[{'kind': 'box', 'box': [4, 0, 6, 10]}])
        self.assertFalse(feasible(s))

    def test_manifest_hash_matches_file_bytes(self):
        cfg = copy.deepcopy(CFG)
        cfg['splits'] = {'train': {'layouts': 1, 'seed_start': 10000}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = build_manifests(root, cfg)
            raw = (root / 'train/manifest.json').read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), summary['train']['sha256'])
            self.assertNotIn(b'\r\n', raw)


if __name__ == '__main__':
    unittest.main()
