# PaddockPsychRL

F1-informed cooperative multi-agent reinforcement learning project that models driver
experience and a simple psychological state in PettingZoo `simple_spread_v3`.

This repo contains:
- A telemetry pipeline using FastF1 (2025 season) to build driver-specific profiles.
- A custom `F1PsychologyWrapper` that injects experience-scaled action noise and
  tracks per-agent `psych_state`.
- Tabular independent Q-learning experiments (baseline vs psych-aware).
- MAPPO-style training (parameter-shared PPO with joint observations via RLlib).

## Project Structure

- `helper.py` - data collection, wrappers, tabular training, and SRS utilities.
- `scripts/train_mappo.py` - RLlib PPO training entrypoint for benchmark/F1 runs.
- `results/mappo/*.json` - saved MAPPO learning curves and run summaries.
- `images/` - generated figures used in the write-up.
- `final_project_kritarth.ipynb` - notebook workflow and experiment orchestration.
- `final_project.tex` - final report source.

## Environment Setup

Recommended Python: `3.10+`

Create and activate a virtual environment, then install MARL dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-marl.txt
```

For notebook workflows, make sure your Jupyter environment uses the same venv.

## Data Pipeline (FastF1)

Driver profiles are derived from race lap-time consistency:
- `collect_season_driver_data(...)` pools per-driver lap times and points.
- `spread_experience_from_peers(...)` maps consistency to an experience range.
- `build_experience_profiles(...)` maps these values to PettingZoo agent IDs.

The `f1_cache/` directory stores FastF1 cache files for reproducibility and speed.

## Run MAPPO Training

From repo root:

```bash
python scripts/train_mappo.py --env benchmark --iterations 150 --out-json results/mappo/benchmark.json
python scripts/train_mappo.py --env f1_baseline --driver-csv checkpoint_driver_stats.csv --iterations 150 --out-json results/mappo/f1_baseline.json
python scripts/train_mappo.py --env f1_psych --driver-csv checkpoint_driver_stats.csv --iterations 150 --out-json results/mappo/f1_psych.json
```

Key options:
- `--num-env-runners` for parallel rollout workers.
- `--train-batch`, `--minibatch`, `--lr` for PPO tuning.
- `--no-joint-obs` to disable joint observation concatenation.

## Core Idea

Each agent gets:
- A driver-dependent experience level from real F1 telemetry.
- A psychological state that degrades after poor rewards and recovers over time.

Psych-aware runs append `psych_state` to observation vectors, while baselines do not.

## Reported MAPPO Snapshot

Using the JSON outputs in `results/mappo/`:
- Benchmark (`SpreadBenchmark-v0`): SRS `0.945`, final mean return `-72.55`.
- F1 baseline: SRS `0.117`, final mean return `-621.64`.
- F1 psych-aware: SRS `0.342`, final mean return `-585.38`.

## Reproducibility Notes

- Set explicit seeds in notebook/script runs for stable comparisons.
- FastF1 availability can vary by event/session data completeness.
- Keep cache and dependency versions fixed for closest metric parity.

## License

Course project repository for CSE 4/546 (University at Buffalo).
