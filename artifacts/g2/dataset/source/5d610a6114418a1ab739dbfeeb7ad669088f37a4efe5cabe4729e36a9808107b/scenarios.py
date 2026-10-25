"""Seeded course generation; held-out geometry is generated without policy tuning."""
from collections import deque
import hashlib
import json
import numpy as np
from .geometry import collision

COURSES = ('open', 'slalom', 'bottleneck', 'crossing')


def generate(seed, course='open'):
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.12, 0.12, (3, 2))
    starts = np.array([[1, 2], [1, 5], [1, 8]], dtype=float) + jitter
    goals = np.array([[9, 2], [9, 5], [9, 8]], dtype=float) + rng.uniform(-0.12, 0.12, (3, 2))
    obstacles = []
    if course == 'slalom':
        obstacles = [{'kind': 'circle', 'center': [float(x + rng.uniform(-0.2, 0.2)), float(y)], 'radius': 0.45}
                     for x, y in ((3, 2), (6, 3), (4, 5), (7, 6), (3, 8), (6, 8))]
    elif course == 'bottleneck':
        shift = float(rng.uniform(-0.25, 0.25))
        obstacles = [{'kind': 'box', 'box': [4.6, 0, 5.4, 4.1 + shift]},
                     {'kind': 'box', 'box': [4.6, 5.9 + shift, 5.4, 10]}]
    elif course == 'crossing':
        starts = np.array([[1, 5], [5, 1], [9, 5]], dtype=float) + jitter
        goals = np.array([[9, 5], [5, 9], [1, 5]], dtype=float) + rng.uniform(-0.12, 0.12, (3, 2))
    elif course != 'open':
        raise ValueError(f'Unknown course: {course}')
    headings = np.arctan2(goals[:, 1] - starts[:, 1], goals[:, 0] - starts[:, 0])
    return {'seed': int(seed), 'course': course, 'arena': [10, 10],
            'starts': np.column_stack((starts, headings)).tolist(), 'goals': goals.tolist(), 'obstacles': obstacles}


def feasible(scene, radius=0.2):
    """Conservative 4-neighbour grid route certificate; peers excluded during routing.

    Returns individual static feasibility, not joint multi-agent solvability.
    Every certified edge is swept checked; starts/goals connect to nearest node.
    """
    starts = np.array(scene['starts'])
    goals = np.array(scene['goals'])
    if collision(starts, starts, scene, radius) or collision(np.column_stack((goals, np.zeros(len(goals)))), np.column_stack((goals, np.zeros(len(goals)))), scene, radius):
        return False
    axis = np.arange(0.5, 10, 0.25)
    def point(node):
        return np.array([axis[node[0]], axis[node[1]], 0.0])
    def clear(a, b):
        return collision(a[None, :], b[None, :], scene, radius + 0.01) is None
    # A half-edge clearance margin certifies every grid edge between free nodes.
    xx, yy = np.meshgrid(axis, axis, indexing='ij')
    points = np.stack((xx, yy), axis=-1)
    margin = radius + 0.126
    free = ((points > margin) & (points < np.array(scene['arena']) - margin)).all(axis=-1)
    for obstacle in scene['obstacles']:
        if obstacle['kind'] == 'circle':
            free &= np.linalg.norm(points - obstacle['center'], axis=-1) > obstacle['radius'] + margin
        else:
            box = obstacle['box']
            free &= np.linalg.norm(points - np.clip(points, box[:2], box[2:]), axis=-1) > margin
    for start, goal in zip(starts, goals):
        s = tuple(np.argmin(abs(axis[:, None] - start[:2]), axis=0))
        g = tuple(np.argmin(abs(axis[:, None] - goal), axis=0))
        if not free[s] or not free[g] or not clear(start, point(s)) or not clear(point(g), np.r_[goal, 0.0]):
            return False
        queue, seen = deque([s]), {s}
        while queue and g not in seen:
            a = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                b = (a[0] + dx, a[1] + dy)
                if min(b) < 0 or max(b) >= len(axis) or b in seen:
                    continue
                if free[b]:
                    seen.add(b)
                    queue.append(b)
        if g not in seen:
            return False
    return True


def build_manifests(root, cfg):
    summary = {}
    for split, spec in cfg['splits'].items():
        path = root / split
        path.mkdir(parents=True, exist_ok=True)
        records = []
        for i in range(spec['layouts']):
            seed = spec['seed_start'] + i
            scene = generate(seed, COURSES[i % 4])
            if not feasible(scene, cfg['robot_radius_m']):
                raise ValueError(f'Infeasible layout {split}/{seed}; revise generator before tuning')
            records.append(scene)
        content = json.dumps(records, indent=2) + '\n'
        # Hash the exact canonical UTF-8 bytes on every OS, avoiding CRLF translation.
        (path / 'manifest.json').write_bytes(content.encode('utf-8'))
        summary[split] = {'layouts': len(records), 'sha256': hashlib.sha256(content.encode()).hexdigest(), 'individual_static_feasibility': True}
    return summary
