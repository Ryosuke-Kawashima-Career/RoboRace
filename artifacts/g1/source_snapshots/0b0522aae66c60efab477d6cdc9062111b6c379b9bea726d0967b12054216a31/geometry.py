"""Geometry shared by collision detection and sensors (metres)."""
import numpy as np


def segment_distance(point, a, b):
    delta = b - a
    t = np.clip(np.dot(point - a, delta) / max(np.dot(delta, delta), 1e-30), 0, 1)
    return float(np.linalg.norm(point - (a + t * delta)))


def ray_circle(origin, direction, center, radius):
    offset = origin - center
    b = np.dot(offset, direction)
    c = np.dot(offset, offset) - radius * radius
    if c <= 0:
        return 0.0
    disc = b * b - c
    if disc < 0:
        return float('inf')
    t = -b - np.sqrt(disc)
    return float(t) if t >= 0 else float('inf')


def ray_box(origin, direction, box):
    lower, upper = np.array(box[:2]), np.array(box[2:])
    near, far = -float('inf'), float('inf')
    for k in range(2):
        if abs(direction[k]) < 1e-12:
            if not lower[k] <= origin[k] <= upper[k]:
                return float('inf')
        else:
            ts = sorted(((lower[k] - origin[k]) / direction[k], (upper[k] - origin[k]) / direction[k]))
            near, far = max(near, ts[0]), min(far, ts[1])
    return max(0.0, float(near)) if far >= max(near, 0) else float('inf')


def swept_box(a, b, box, radius):
    """Exact clearance for a line segment against an axis-aligned rectangle."""
    lo, hi = np.array(box[:2]), np.array(box[2:])
    for p in (a, b):
        if np.linalg.norm(p - np.clip(p, lo, hi)) <= radius:
            return True
    delta = b - a
    length = np.linalg.norm(delta)
    if length > 0 and ray_box(a, delta / length, box) <= length:
        return True
    corners = [lo, np.array([lo[0], hi[1]]), hi, np.array([hi[0], lo[1]])]
    return any(segment_distance(corner, a, b) <= radius for corner in corners)


def collision(old, new, scene, radius, tolerance=1e-6):
    r = radius + tolerance
    for i, (a, b) in enumerate(zip(old[:, :2], new[:, :2])):
        if np.any(np.minimum(a, b) <= r) or np.any(np.maximum(a, b) >= np.array(scene['arena']) - r):
            return {'type': 'wall', 'robots': [i]}
        for j, obstacle in enumerate(scene['obstacles']):
            hit = (segment_distance(np.array(obstacle['center']), a, b) <= r + obstacle['radius']
                   if obstacle['kind'] == 'circle' else swept_box(a, b, obstacle['box'], r))
            if hit:
                return {'type': 'obstacle', 'robots': [i], 'obstacle': j}
    for i in range(len(old)):
        for j in range(i):
            # Relative swept segment detects robots exchanging places within one substep.
            if segment_distance(np.zeros(2), old[i, :2] - old[j, :2], new[i, :2] - new[j, :2]) <= 2 * radius + tolerance:
                return {'type': 'peer', 'robots': [j, i]}
    return None
