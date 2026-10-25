# RoboRace

G1–G3 implementation: deterministic 2D differential-drive robot races, local sensing,
collision detection, classical, behavior-cloning and independent PPO controllers,
and reproducible recordings. Final comparison/submission remain later phases.
See `dev/plans/README.md` for measured results and limitations.

## Run on the verified local environment

From this directory in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe -m roborace.cli demo --course open --seed 20000 --output artifacts/g1/demo_open_final
& .\.venv\Scripts\python.exe -m roborace.cli demo --course crossing --seed 20003 --output artifacts/g1/demo_crossing
& .\.venv\Scripts\python.exe -m roborace.cli evaluate --output artifacts/g1/validation_final
```

The existing virtual environment inherits installed base packages. For a new isolated
environment, use Python 3.13 and install `requirements-runtime.txt`, then run from the
repository root (or install this project using `pip install -e .`). A fresh downloaded
dependency installation has not been exercised; the pinned versions above passed the
local import and simulator checks. No GPU or Torch is required for G1. Torch is used
only by the G0 synthetic probe and future learning phases.

`python -m roborace.cli scenarios` rebuilds the 200/40/100 train/validation/test
manifests and individual static-route certificates. The checked-in manifests are
already generated; do not regenerate them after tuning without a versioned reason.
The G1 `evaluate` command deliberately uses only the first eight validation layouts,
two per course family, and never evaluates held-out test policies.

Each run saves an episode summary, config, raw state/action trajectory, scene and
checksums. Source snapshots resolve recorded source hashes. A demo also writes a
1x GIF and final PNG. GIF is G1 inspection media, not the final MP4 submission.
Recording uses actual simulator states with no interpolated path animation.

## Implemented contract

- Three robots by default; local 44-value observations, speed/turn actions.
- Simultaneous midpoint differential-drive integration with ten substeps per command.
- Swept robot–robot, wall and static circle/rectangle contact checks on each substep.
- First collision terminates the team; retain last safe state as conservative response.
- Goal dwell, stopped finished robots, per-robot finish times, success, horizon truncation,
  path length, deadlock flag and failure-capped completion score.
- Seeded open/slalom/bottleneck/crossing maps and geometry-based feasibility checks.
- Goal/range steering, right-side yielding and local dynamic-window safety filtering.

The controller consumes observations only; it has no privileged map or hidden state.
The sparse-sensor safety filter is a heuristic, not a collision-free guarantee.
Individual route certificates do not prove joint multi-robot solvability. The reward
returned by `step` is provisional diagnostic shaping, not the frozen PPO objective.
Metric units are metres and simulated seconds; controller timings are wall-clock.

## Current limitations

The baseline stalls in some slalom/bottleneck/crossing cases. G2 must filter expert
failures and measure coverage before cloning. The G2 learned policy currently collides
on obstacle/crossing validation courses; it is not a reliable navigation controller.
No 3D/physical validation, MP4 encoder or final course submission is implemented. The inherited
environment must be replaced or independently verified for the final reproducible
release. `dev/` and `docs/` remain ignored by the pre-existing `.gitignore`; explicitly
decide what planning/assignment material to publish when preparing the public repo.

## G2 imitation learning

Verified locally with CPU PyTorch 2.9.1+cpu. `requirements-learning.txt` records direct
versions; a complete isolated release lock is still pending. From the repository root:

```powershell
# Reuse the existing filtered dataset/checkpoint, or explicitly rebuild them:
& .\.venv\Scripts\python.exe -m roborace.cli collect-bc --dataset artifacts/g2/dataset
& .\.venv\Scripts\python.exe -m roborace.cli train-bc --dataset artifacts/g2/dataset --output artifacts/g2/bc_seed0
& .\.venv\Scripts\python.exe -m roborace.cli evaluate --controller bc --checkpoint artifacts/g2/bc_seed0/policy.pt --output artifacts/g2/validation
& .\.venv\Scripts\python.exe -m roborace.cli demo --controller bc --checkpoint artifacts/g2/bc_seed0/policy.pt --course open --seed 20000 --output artifacts/g2/demo_open
& .\.venv\Scripts\python.exe -m roborace.cli demo --controller bc --checkpoint artifacts/g2/bc_seed0/policy.pt --course slalom --seed 20001 --output artifacts/g2/demo_slalom_failure
& .\.venv\Scripts\python.exe scripts/verify_g2.py
```

Collection uses only the first 40 training layouts with seeded initial-heading
perturbations. It keeps pre-arrival trajectories only for robots that finish, and
rejects every trajectory from a team-collision episode. Partial successes in timeout
episodes are retained and explicitly counted. No validation/test trajectories become
demonstrations. Internal fit/holdout uses whole training episodes; normalization fits
only the fit partition. Validation selects checkpoints and is not an unbiased test.

The MLP learns bounded speed/turn actions; the same deterministic 0.27 m arrival stop
as the classical controller is applied explicitly. No classical obstacle-avoidance
fallback runs during learned inference. Checkpoints store architecture, statistics,
training configuration and data/environment/source provenance. Default G2 uses one
training seed; the multi-seed final comparison remains G4 work.

## G3 PPO pilot

Uses the existing CPU PyTorch dependencies in `requirements-learning.txt`; no new
external RL library is required. The custom implementation has tests for action
probabilities, clipping, gradients and GAE termination/truncation behavior.

```powershell
& .\.venv\Scripts\python.exe -m roborace.cli train-ppo --ppo-config configs/ppo.json --output artifacts/g3/ppo_seed0
& .\.venv\Scripts\python.exe -m roborace.cli evaluate --controller ppo --checkpoint artifacts/g3/ppo_seed0/policy.pt --output artifacts/g3/validation
& .\.venv\Scripts\python.exe -m roborace.cli demo --controller ppo --checkpoint artifacts/g3/ppo_seed0/policy.pt --course open --seed 20000 --output artifacts/g3/demo_open
& .\.venv\Scripts\python.exe -m roborace.cli demo --controller ppo --checkpoint artifacts/g3/ppo_seed0/policy.pt --course slalom --seed 20001 --output artifacts/g3/demo_slalom
& .\.venv\Scripts\python.exe scripts/verify_g3.py
```

The policy shares actor parameters across robots but each actor and value function
receives only that robot's 44 local inputs. It trains from scratch on randomly
shuffled training layouts, with no BC weights or expert actions. This is independent
PPO with parameter sharing, not MAPPO. Frozen physical input scaling is used without
fitted observation statistics.

The bounded Beta distribution avoids clipping stochastic actions: sampled unit-space
coordinates map to speed `[0,1]` and turn `[-2,2]`. PPO ratios use the same unit-space
density for old/new policies; the constant physical-transform Jacobian cancels.
Reported entropy is in unit space and may be negative (differential entropy).
Deterministic inference uses the distribution mean. The explicit 0.27 m arrival stop
matches BC/classical; masked arrival/finished samples do not train the actor.

True terminal transitions disable value bootstrap; horizon truncation bootstraps the
final observation, while GAE recursion stops at all resets/robot arrivals. Finished
robots contribute no subsequent actor/critic samples. Reward coefficients and all
optimization/selection settings are in `configs/ppo.json`. Checkpoints store model,
runtime, source, environment, training config and manifest provenance. Last optimizer
state is saved for inspection; seamless resume/RNG restoration is not implemented.

Default run is a seed-0 pilot capped at 32,768 world steps or 900 seconds, below the
million-step ceiling. Every 32 updates, four validation layouts select the checkpoint.
All failures are retained; this pilot does not establish superiority or reliability.
Do not regenerate scenario splits or tune against the test manifest. Full multi-seed
training and frozen held-out comparison remain G4 work.
