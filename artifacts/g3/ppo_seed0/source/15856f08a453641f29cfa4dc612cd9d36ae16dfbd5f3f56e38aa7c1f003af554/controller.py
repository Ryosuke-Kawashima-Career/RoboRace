"""Classical goal steering and local range avoidance; no map/state access."""
import numpy as np


class Classical:
    def reset(self, seed=0):
        pass

    def act(self, observations):
        actions = []
        for row in observations:
            distance = float(row[0]) * np.sqrt(200)
            bearing = np.arctan2(row[1], row[2])
            if distance <= 0.27:
                actions.append([0, 0])
                continue
            ranges = row[5:29] * 5
            angles = np.arange(24) * 2 * np.pi / 24
            # Seek a locally clear heading aligned with the goal; fixed left bias breaks symmetric ties.
            clearance = np.minimum(np.roll(ranges, 1), np.minimum(ranges, np.roll(ranges, -1)))
            angle_error = np.arctan2(np.sin(angles - bearing), np.cos(angles - bearing))
            scores = np.cos(angle_error) + 0.45 * np.minimum(clearance, 2) - 0.02 * np.sin(-angles)
            scores[clearance < 0.5] -= 4
            chosen = angles[int(np.argmax(scores))]
            turn = np.arctan2(np.sin(chosen), np.cos(chosen))
            if abs(bearing) < 0.2 and min(ranges[0], ranges[1], ranges[-1]) > 0.8:
                turn = bearing
            omega = np.clip(2.5 * turn, -2, 2)
            front = min(ranges[0], ranges[1], ranges[-1])
            speed = min(0.8, distance) * max(0, np.cos(turn)) * np.clip((front - 0.35) / 0.7, 0, 1)
            # Yield to an approaching peer on the right; no global ID or map privilege.
            for slot in row[29:].reshape(3, 5):
                px, py = slot[:2] * 5
                if slot[4] and 0 < px < 1.1 and -0.7 < py < 0.15:
                    speed *= 0.15
            # Local dynamic-window safety filter. Reconstruct sensor hits only,
            # never consult the simulator's hidden obstacle geometry.
            points = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
            points = points[ranges < 2.0]
            candidates = [(speed, omega)]
            candidates += [(v, w) for v in (0.0, 0.25, 0.5, 0.75) for w in (-2., -1., 0., 1., 2.)]
            best, best_score = (0.0, omega), -float('inf')
            for v, w in candidates:
                times = np.arange(1, 9) * 0.1
                if abs(w) < 1e-9:
                    trajectory = np.column_stack((v * times, np.zeros(8)))
                else:
                    trajectory = np.column_stack((v / w * np.sin(w * times), v / w * (1 - np.cos(w * times))))
                clearance_min = float(np.min(np.linalg.norm(trajectory[:, None, :] - points[None, :, :], axis=-1))) if len(points) else 2.0
                if v > 0 and clearance_min < 0.34:
                    continue
                progress = trajectory[-1] @ np.array([np.cos(bearing), np.sin(bearing)])
                alignment = np.cos(turn - w * .8)
                score = 2 * progress + .4 * alignment + .15 * min(clearance_min, 1) - .03 * abs(w)
                if score > best_score:
                    best_score, best = score, (v, w)
            speed, omega = best
            actions.append([speed, omega])
        return np.asarray(actions)
