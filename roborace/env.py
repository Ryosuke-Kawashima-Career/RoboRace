"""Simultaneous differential-drive world with local observations."""
from collections import deque
import copy
import numpy as np
from .geometry import collision, ray_box, ray_circle
from .scenarios import generate


class World:
    def __init__(self, config):
        self.cfg = copy.deepcopy(config)

    def reset(self, seed=0, course='open', scene=None):
        self.scene = copy.deepcopy(scene if scene is not None else generate(seed, course))
        self.state = np.array(self.scene['starts'], dtype=float)
        self.goals = np.array(self.scene['goals'], dtype=float)
        self.n = len(self.state)
        self.velocity = np.zeros((self.n, 2))
        self.finished = np.zeros(self.n, dtype=bool)
        self.dwell = np.zeros(self.n, dtype=int)
        self.finish_times = [None] * self.n
        self.path_length = np.zeros(self.n)
        self.steps = 0
        self.time = 0.0
        self.terminated = self.truncated = self.deadlock = False
        self.contact = None
        self.history = deque([self._distance()], maxlen=self.cfg['deadlock']['window_steps'] + 1)
        if collision(self.state, self.state, self.scene, self.cfg['robot_radius_m'], self.cfg['contact_tolerance_m']):
            raise ValueError('Initial state is in collision')
        return self.observe()

    def _distance(self):
        return float(np.linalg.norm(self.goals - self.state[:, :2], axis=1).sum())

    def observe(self):
        cfg = self.cfg
        rows = []
        count = cfg['range_sensor']['rays']
        maximum = cfg['range_sensor']['max_range_m']
        world_velocity = self.velocity[:, :1] * np.column_stack((np.cos(self.state[:, 2]), np.sin(self.state[:, 2])))
        for i, (x, y, heading) in enumerate(self.state):
            position = np.array([x, y])
            c, s = np.cos(heading), np.sin(heading)
            rotation = np.array([[c, s], [-s, c]])
            goal = self.goals[i] - position
            bearing = np.arctan2(goal[1], goal[0]) - heading
            row = [np.linalg.norm(goal) / np.linalg.norm(self.scene['arena']), np.sin(bearing), np.cos(bearing),
                   self.velocity[i, 0] / cfg['linear_velocity_limits_m_s'][1], self.velocity[i, 1] / cfg['angular_velocity_limits_rad_s'][1]]
            for k in range(count):
                angle = heading + k * cfg['range_sensor']['field_of_view_rad'] / count
                direction = np.array([np.cos(angle), np.sin(angle)])
                distance = maximum
                for axis in range(2):
                    if abs(direction[axis]) > 1e-12:
                        wall = self.scene['arena'][axis] if direction[axis] > 0 else 0
                        distance = min(distance, (wall - position[axis]) / direction[axis])
                for obstacle in self.scene['obstacles']:
                    hit = (ray_circle(position, direction, np.array(obstacle['center']), obstacle['radius'])
                           if obstacle['kind'] == 'circle' else ray_box(position, direction, obstacle['box']))
                    distance = min(distance, hit)
                for j in range(self.n):
                    if i != j:
                        distance = min(distance, ray_circle(position, direction, self.state[j, :2], cfg['robot_radius_m']))
                row.append(max(0.0, distance) / maximum)
            peers = sorted((j for j in range(self.n) if j != i), key=lambda j: (np.linalg.norm(self.state[j, :2] - position), j))
            for slot in range(cfg['maximum_observed_peers']):
                if slot < len(peers):
                    j = peers[slot]
                    row.extend((rotation @ (self.state[j, :2] - position) / 5).tolist())
                    row.extend((rotation @ (world_velocity[j] - world_velocity[i]) / 2).tolist())
                    row.append(1.0)
                else:
                    row.extend([0.0] * 5)
            rows.append(row)
        observation = np.asarray(rows, dtype=np.float32)
        if observation.shape != (self.n, cfg['observation']['dimension']) or not np.isfinite(observation).all():
            raise RuntimeError('Observation contract violation')
        return observation

    def step(self, actions):
        if self.terminated or self.truncated:
            raise RuntimeError('Reset before stepping a completed episode')
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.n, 2) or not np.isfinite(actions).all():
            raise ValueError(f'Actions must be finite with shape {(self.n, 2)}')
        actions = actions.copy()
        actions[:, 0] = np.clip(actions[:, 0], *self.cfg['linear_velocity_limits_m_s'])
        actions[:, 1] = np.clip(actions[:, 1], *self.cfg['angular_velocity_limits_rad_s'])
        actions[self.finished] = 0
        before = np.linalg.norm(self.goals - self.state[:, :2], axis=1)
        self.velocity = actions.copy()
        dt = self.cfg['control_dt_s'] / self.cfg['physics_substeps']
        for _ in range(self.cfg['physics_substeps']):
            old = self.state.copy()
            new = old.copy()
            new, self.contact = self.advance(old, actions, dt)
            self.time += dt
            if self.contact:
                # 2D conservatively retains the safe pose; physics backends retain measured contact poses.
                if getattr(self, "collision_retains_actual_pose", False):
                    self.path_length += np.linalg.norm(new[:, :2] - old[:, :2], axis=1)
                    self.state = new
                self.terminated = True
                self.velocity[:] = 0
                self.contact['simulated_time_s'] = self.time
                break
            self.path_length += np.linalg.norm(new[:, :2] - old[:, :2], axis=1)
            self.state = new
        self.steps += 1
        distances = np.linalg.norm(self.goals - self.state[:, :2], axis=1)
        newly = np.zeros(self.n, dtype=bool)
        if not self.contact:
            eligible = (distances <= self.cfg['goal_tolerance_m']) & (actions[:, 0] <= self.cfg['arrival_max_speed_m_s'])
            self.dwell = np.where(eligible, self.dwell + 1, 0)
            newly = (self.dwell >= self.cfg['arrival_dwell_steps']) & ~self.finished
            for i in np.flatnonzero(newly):
                self.finish_times[i] = round(self.time, 8)
            self.finished |= newly
            self.velocity[self.finished] = 0
            self.terminated = bool(self.finished.all())
        self.truncated = bool(self.steps >= self.cfg['horizon_steps'] and not self.terminated)
        self.history.append(self._distance())
        if len(self.history) == self.history.maxlen and self.history[0] - self.history[-1] < self.cfg['deadlock']['minimum_team_distance_reduction_m'] and not self.finished.all():
            self.deadlock = True
        # Provisional diagnostic reward; final PPO reward configuration belongs to G3.
        reward = before - distances + newly.astype(float) * 5 - 0.01
        if self.contact:
            reward -= 10
        return self.observe(), reward.astype(np.float32), self.terminated, self.truncated, self.summary()

    def advance(self, old, actions, dt):
        """Backend integration hook; returns actual pose and first contact."""
        new = old.copy()
        heading = old[:, 2] + actions[:, 1] * dt / 2
        new[:, :2] += actions[:, :1] * np.column_stack((np.cos(heading), np.sin(heading))) * dt
        new[:, 2] = (old[:, 2] + actions[:, 1] * dt + np.pi) % (2 * np.pi) - np.pi
        return new, collision(old, new, self.scene, self.cfg['robot_radius_m'], self.cfg['contact_tolerance_m'])

    def summary(self):
        success = bool(self.finished.all() and not self.contact)
        horizon = self.cfg['control_dt_s'] * self.cfg['horizon_steps']
        times = [t for t in self.finish_times if t is not None]
        return {'team_success': success, 'per_robot_success': self.finished.tolist(),
                'finish_times_s': self.finish_times.copy(), 'simulated_time_s': round(self.time, 8),
                'steps': self.steps, 'collision': copy.deepcopy(self.contact), 'timeout': self.truncated,
                'deadlock_observed': self.deadlock, 'path_length_m': self.path_length.tolist(),
                'makespan_s': max(times) if success else None,
                'capped_completion_s': max(times) if success else horizon}
