"""G0 runtime probe, not the collision-aware G1 simulator.

Run using the selected Python, from any directory. Writes local evidence only.
"""
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import PIL
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "g0"
OUT.mkdir(parents=True, exist_ok=True)
contract_bytes = (ROOT / "configs" / "g0_contract.json").read_bytes()
cfg = json.loads(contract_bytes)
torch.set_num_threads(cfg["compute_budget"]["torch_threads"])


def reset(seed):
    rng = np.random.default_rng(seed)
    state = np.zeros((3, 3), dtype=np.float64)
    state[:, 0] = 1.0
    state[:, 1] = [2.0, 5.0, 8.0]
    state[:, 2] = rng.uniform(-0.03, 0.03, 3)
    return state


def step(state, actions):
    # Simultaneous kinematic update; deliberately no collision claims at G0.
    result = state.copy()
    v = np.clip(actions[:, 0], 0, 1)
    w = np.clip(actions[:, 1], -2, 2)
    for _ in range(cfg["physics_substeps"]):
        dt = cfg["control_dt_s"] / cfg["physics_substeps"]
        result[:, 0] += v * np.cos(result[:, 2]) * dt
        result[:, 1] += v * np.sin(result[:, 2]) * dt
        result[:, 2] = (result[:, 2] + w * dt + np.pi) % (2 * np.pi) - np.pi
    return result


def frame(state):
    canvas = Image.new("RGB", (500, 540), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((10, 10, 490, 490), outline="black", width=2)
    for i, color in enumerate(("blue", "green", "purple")):
        gx, gy = 10 + 8 * 48, 490 - (2 + 3 * i) * 48
        draw.ellipse((gx-14, gy-14, gx+14, gy+14), outline=color, width=2)
        x, y = 10 + state[i, 0] * 48, 490 - state[i, 1] * 48
        draw.ellipse((x-10, y-10, x+10, y+10), fill=color)
    draw.text((12, 505), "G0 kinematics smoke: no collision/sensor implementation", fill="black")
    return canvas


initial = reset(42)
assert np.array_equal(initial, reset(42))
assert not np.array_equal(initial, reset(43))
actions = np.tile([0.5, 0.0], (3, 1))
state = initial.copy()
frames = [frame(state)]
for i in range(100):
    state = step(state, actions)
    if i % 5 == 4:
        frames.append(frame(state))
replay = reset(42)
for _ in range(100):
    replay = step(replay, actions)
assert np.array_equal(state, replay)
assert np.allclose(np.linalg.norm(state[:, :2] - initial[:, :2], axis=1), 5.0)
frames[-1].save(OUT / "smoke_final.png")
frames[0].save(OUT / "smoke.gif", save_all=True, append_images=frames[1:], duration=500, loop=0)

start = time.perf_counter()
for _ in range(100):
    probe = reset(42)
    for _ in range(100):
        probe = step(probe, actions)
elapsed = time.perf_counter() - start

torch.manual_seed(42)
model = torch.nn.Sequential(torch.nn.Linear(44, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2), torch.nn.Tanh())
x = torch.randn(256, 44)
target = torch.zeros(256, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
start = time.perf_counter()
losses = []
for _ in range(200):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
train_elapsed = time.perf_counter() - start
assert np.isfinite(losses).all() and losses[-1] < losses[0]
weights = OUT / "smoke_weights.pt"
torch.save(model.state_dict(), weights)
loaded = torch.nn.Sequential(torch.nn.Linear(44, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2), torch.nn.Tanh())
loaded.load_state_dict(torch.load(weights, weights_only=True))
with torch.no_grad():
    assert torch.equal(model(x), loaded(x))

result = {
    "status": "passed",
    "scope": "kinematic/reset/render and synthetic neural-network training/save-load only; not G1 or task-performance evidence",
    "python": sys.version.split()[0], "platform": platform.platform(),
    "versions": {"numpy": np.__version__, "torch": torch.__version__, "pillow": PIL.__version__},
    "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads(),
    "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
    "kinematic_world_steps": 10000, "kinematic_wall_s": elapsed,
    "kinematic_world_steps_per_s": 10000 / elapsed,
    "synthetic_train_updates": 200, "synthetic_train_wall_s": train_elapsed,
    "synthetic_initial_loss": losses[0], "synthetic_final_loss": losses[-1],
    "deterministic_reset_replay": True, "weight_reload_exact": True,
    "limitations": ["No collision geometry, sensors, reward, task controller or MP4 tested", "Kinematic throughput is not full-environment/PPO throughput", "Project venv inherits installed base packages and is not portable"]
}
(OUT / "runtime.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
