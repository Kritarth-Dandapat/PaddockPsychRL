#!/usr/bin/env python3
"""
MAPPO-style training: multi-agent PPO with parameter sharing + joint observations
(centralized state) on PettingZoo parallel envs, via Ray RLlib (new RLModule API).

Environments (register with tune before building the algorithm):
  SpreadBenchmark-v0  — vanilla simple_spread_v3, N=3 (standard MPE benchmark).
  SpreadF1Psych-v0    — F1-backed spread + F1PsychologyWrapper; env_config selects
                        psych_in_obs and action_noise.

Example (from repo root, after pip install -r requirements-marl.txt):
  python scripts/train_mappo.py --env benchmark --iterations 50 --num-env-runners 4
  python scripts/train_mappo.py --env f1_baseline --driver-csv checkpoint_driver_stats.csv \\
      --iterations 80 --num-env-runners 4
  python scripts/train_mappo.py --env f1_psych --driver-csv checkpoint_driver_stats.csv \\
      --iterations 80 --num-env-runners 4

SRS: this script logs mean episode return per training iteration; feed that list to
helper.strategy_resilience_score in a notebook for the same SRS as tabular runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from helper import (
    build_mappo_pettingzoo_parallel,
    load_driver_stats_csv,
    spread_experience_from_peers,
    strategy_resilience_score,
)


def _register_envs() -> None:
    def benchmark_creator(cfg):
        cfg = cfg or {}
        return ParallelPettingZooEnv(
            build_mappo_pettingzoo_parallel(
                benchmark=True,
                joint_obs_for_mappo=bool(cfg.get("joint_obs_for_mappo", True)),
                max_cycles=int(cfg.get("max_cycles", 25)),
            )
        )

    def f1_creator(cfg):
        cfg = cfg or {}
        path = cfg.get("driver_csv")
        if not path:
            raise ValueError("SpreadF1Psych-v0 requires env_config['driver_csv']")
        raw = load_driver_stats_csv(path)
        stats = spread_experience_from_peers(raw)
        return ParallelPettingZooEnv(
            build_mappo_pettingzoo_parallel(
                benchmark=False,
                driver_stats=stats,
                psych_in_obs=bool(cfg.get("psych_in_obs", False)),
                action_noise=bool(cfg.get("action_noise", False)),
                joint_obs_for_mappo=bool(cfg.get("joint_obs_for_mappo", True)),
                max_cycles=int(cfg.get("max_cycles", 25)),
            )
        )

    register_env("SpreadBenchmark-v0", benchmark_creator)
    register_env("SpreadF1Psych-v0", f1_creator)


def _episode_return_mean(result: Dict[str, Any]) -> float:
    """Best-effort extraction across Ray RLlib result layouts."""
    if not isinstance(result, dict):
        return float("nan")
    v = result.get("episode_reward_mean")
    if v is not None:
        return float(v)
    er = result.get("env_runners", {})
    if isinstance(er, dict):
        for key in (
            "episode_return_mean",
            "episode_reward_mean",
        ):
            if key in er and er[key] is not None:
                return float(er[key])
        hs = er.get("hist_stats", {})
        if isinstance(hs, dict):
            for key in ("episode_return", "episode_reward"):
                seq = hs.get(key)
                if isinstance(seq, (list, tuple)) and len(seq) > 0:
                    return float(np.mean(seq[-1]))
    return float("nan")


def build_ppo_config(
    env_name: str,
    env_config: Dict[str, Any],
    num_env_runners: int,
    num_gpus: float,
    lr: float,
    train_batch_size_per_learner: int,
    minibatch_size: int,
) -> PPOConfig:
    def _flatten_obs_connector(env, spaces=None, device=None):
        # Accept latest RLlib connector callback signature.
        return FlattenObservations(multi_agent=True)

    cfg = PPOConfig()
    # New API stack (RLModule + Learner); Ray 2.37+.
    if hasattr(cfg, "api_stack"):
        cfg = cfg.api_stack(enable_rl_module_and_learner=True)
    cfg = (
        cfg.environment(env=env_name, env_config=env_config)
        .env_runners(
            num_env_runners=num_env_runners,
            env_to_module_connector=_flatten_obs_connector,
        )
        .multi_agent(
            policies={"shared"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared",
        )
        .training(
            lr=lr,
            train_batch_size_per_learner=train_batch_size_per_learner,
            minibatch_size=minibatch_size,
            vf_loss_coeff=0.05,
        )
        .rl_module(
            model_config=DefaultModelConfig(
                fcnet_hiddens=[256, 256],
                vf_share_layers=True,
            ),
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={"shared": RLModuleSpec()},
            ),
        )
    )
    # New RLlib API stack places learner-side GPU assignment here.
    if hasattr(cfg, "learners"):
        cfg = cfg.learners(
            num_learners=1,
            num_gpus_per_learner=float(num_gpus),
        )
    else:
        cfg = cfg.resources(num_gpus=num_gpus)
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MAPPO-style PPO on PettingZoo (Ray RLlib).")
    p.add_argument(
        "--env",
        choices=("benchmark", "f1_baseline", "f1_psych"),
        default="benchmark",
        help="benchmark: vanilla MPE spread N=3; f1_* uses SpreadF1Psych-v0.",
    )
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--num-env-runners", type=int, default=4)
    p.add_argument(
        "--driver-csv",
        type=str,
        default=os.path.join(ROOT, "checkpoint_driver_stats.csv"),
        help="Driver rows for F1 env (see helper.export_driver_stats_csv).",
    )
    p.add_argument("--max-cycles", type=int, default=25)
    p.add_argument("--no-joint-obs", action="store_true", help="Disable JointObservationWrapper.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-batch", type=int, default=4000)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--num-gpus", type=float, default=0.0, help="GPUs to allocate to PPO learner.")
    p.add_argument("--out-json", type=str, default="", help="Write summary metrics + SRS to this path.")
    args = p.parse_args(argv)

    _register_envs()

    if args.env == "benchmark":
        env_name = "SpreadBenchmark-v0"
        env_config: Dict[str, Any] = {
            "max_cycles": args.max_cycles,
            "joint_obs_for_mappo": not args.no_joint_obs,
        }
    else:
        env_name = "SpreadF1Psych-v0"
        env_config = {
            "driver_csv": os.path.abspath(args.driver_csv),
            "max_cycles": args.max_cycles,
            "joint_obs_for_mappo": not args.no_joint_obs,
            "psych_in_obs": args.env == "f1_psych",
            # Baseline: isolate psych-in-observation; psych run may keep telemetry-linked noise.
            "action_noise": args.env == "f1_psych",
        }

    ray.init(ignore_reinit_error=True)
    try:
        config = build_ppo_config(
            env_name,
            env_config,
            num_env_runners=args.num_env_runners,
            num_gpus=args.num_gpus,
            lr=args.lr,
            train_batch_size_per_learner=args.train_batch,
            minibatch_size=args.minibatch,
        )
        algo = PPO(config=config)
        returns: List[float] = []
        for it in range(args.iterations):
            result = algo.train()
            m = _episode_return_mean(result)
            returns.append(m)
            print(f"iter {it + 1}/{args.iterations}  episode_return_mean ~ {m:.4f}")

        srs = strategy_resilience_score([r for r in returns if np.isfinite(r)])
        summary = {
            "env": args.env,
            "env_name": env_name,
            "env_config": {k: v for k, v in env_config.items() if k != "driver_csv"},
            "driver_csv": env_config.get("driver_csv", ""),
            "iterations": args.iterations,
            "episode_return_mean_per_iter": returns,
            "srs_on_iter_means": srs,
        }
        if args.out_json:
            out_dir = os.path.dirname(os.path.abspath(args.out_json))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print("wrote", args.out_json)
        algo.stop()
    finally:
        ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
