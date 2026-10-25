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
            actions.append([speed, omega])
        return np.asarray(actions)
