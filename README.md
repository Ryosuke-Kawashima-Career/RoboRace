# RoboRace

G1 implementation: deterministic 2D differential-drive robot races, local sensing,
collision detection, a classical controller and reproducible recordings. AI training
and the final submission are later phases. See `dev/plans/README.md` for the roadmap.

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
failures and measure coverage before cloning. No learned navigation, 3D/physical
validation, MP4 encoder or final course submission is implemented. The inherited
environment must be replaced or independently verified for the final reproducible
release. `dev/` and `docs/` remain ignored by the pre-existing `.gitignore`; explicitly
decide what planning/assignment material to publish when preparing the public repo.
