import os
from typing import Any, Dict, List, Optional, Tuple, Union

import collections

import csv
import fastf1
import numpy as np
import pandas as pd
from gymnasium import spaces
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils.wrappers import BaseWrapper

MIN_POOLED_LAPS_DEFAULT = 5
SEASON_TOP_N_DEFAULT = 10


def _race_lap_times_seconds(session: Any, driver_code: str) -> List[float]:
    """
    Return this driver's completed race lap times in seconds (empty if none).

    FastF1 renamed the lap filter API: newer versions expose ``pick_drivers``,
    older ones only ``pick_driver``. We branch on attribute presence instead of
    try/except so behavior stays obvious.
    """
    laps = session.laps
    # Prefer the plural API when present (current FastF1).
    pick_many = getattr(laps, "pick_drivers", None)
    if callable(pick_many):
        sub = pick_many(driver_code)
    else:
        # Legacy single-driver selector.
        sub = laps.pick_driver(driver_code)
    # No laps table rows for this driver in this session.
    if sub is None or len(sub) == 0:
        return []
    # ``LapTime`` is a timedelta; convert to seconds and drop incomplete/invalid rows.
    ser = sub["LapTime"].dt.total_seconds().dropna()
    return [float(x) for x in ser.tolist()]


# --- TASK 1: DATA LOADING & CACHE FIX ---
def download_driver_data(year, event, driver_code, cache_path='f1_cache'):
    """
    Load one race session and derive simple per-driver stats for RL agent profiles.
    """
    # Persist FastF1 API responses so repeat runs do not re-download everything.
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    fastf1.Cache.enable_cache(cache_path)

    # ``event`` is the round identifier FastF1 expects (number or name, per docs).
    session = fastf1.get_session(year, event, 'R')
    # We only need timing data, not telemetry matrices (faster, smaller cache).
    session.load(telemetry=False, weather=False)

    lap_times_list = _race_lap_times_seconds(session, driver_code)
    arr = np.asarray(lap_times_list, dtype=float)
    # Spread of lap times: lower std ⇒ more consistent ⇒ stronger "consistency" signal.
    lap_std = float(np.std(arr)) if arr.size else 0.0
    # Inverse spread (+ epsilon) maps consistency to a positive unbounded score.
    consistency_inv = 1.0 / (lap_std + 1e-6)
    # Squash into a bounded trait used as experience/resilience in the env wrapper.
    resilience = float(np.clip(consistency_inv / 10, 0.5, 1.0))
    experience = resilience
    return {
        'code': driver_code,
        'lap_std': lap_std,
        'consistency_inv': consistency_inv,
        'resilience': resilience,
        'experience': experience,
    }


def collect_season_driver_data(
    year: int,
    cache_path: str = "f1_cache",
    top_n: int = SEASON_TOP_N_DEFAULT,
    min_pooled_laps: int = MIN_POOLED_LAPS_DEFAULT,
    skip_event_formats: Tuple[str, ...] = ("testing",),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Scan all race weekends in ``year``: sum championship points from each Race ``results``
    table and pool every driver's race lap times. Rank by total points, then take ``top_n``
    drivers with at least ``min_pooled_laps`` pooled laps. If fewer than ``top_n`` qualify,
    fill from the remaining championship order even when below ``min_pooled_laps`` (see
    ``meta['padded_drivers']``).

    Partial seasons: only successfully loaded events contribute; top 10 is from available
    data (not the full calendar if future races are missing).

    Returns:
        (stats_for_top_n, meta) where each stats dict matches ``download_driver_data`` keys
        plus ``season_points`` and ``n_laps``.
    """
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    fastf1.Cache.enable_cache(cache_path)

    # One row per Grand Prix weekend (and similar); includes round index and format.
    schedule = fastf1.get_event_schedule(year)
    # Championship points accumulated across all successfully loaded races.
    points: Dict[str, float] = collections.defaultdict(float)
    # Pool all race lap times per three-letter code across the season.
    laps_by_driver: Dict[str, List[float]] = collections.defaultdict(list)
    # Rounds we fully processed (for meta / debugging).
    rounds_processed: List[int] = []
    # Human-readable reasons we skipped a weekend (missing data, load failure, etc.).
    skipped_events: List[str] = []

    for _, ev in schedule.iterrows():
        # Skip test sessions / non-championship formats the caller does not want.
        fmt = str(ev.get("EventFormat", "") or "").lower()
        if fmt in {x.lower() for x in skip_event_formats}:
            continue
        rnd = ev.get("RoundNumber", None)
        # Pandas/JSON often uses NaN for missing numbers; treat as unusable round index.
        if rnd is None or (isinstance(rnd, float) and np.isnan(rnd)):
            skipped_events.append(str(ev.get("EventName", "?")))
            continue
        # Normalize round index: schedule rows are usually numeric; ``to_numeric`` also accepts "12".
        rnd_int: Optional[int] = None
        if isinstance(rnd, (int, np.integer)):
            rnd_int = int(rnd)
        elif isinstance(rnd, float) and not np.isnan(rnd):
            rnd_int = int(rnd)
        else:
            rnum = pd.to_numeric(rnd, errors="coerce")
            if not pd.isna(rnum):
                rnd_int = int(rnum)
        if rnd_int is None:
            skipped_events.append(str(ev.get("EventName", "?")))
            continue
        event_label = f"{year} R{rnd_int} {ev.get('EventName', '')}"
        # Isolated failure per round: calendar may include future races or bad cache state.
        # FastF1 can raise for network, missing files, or invalid session; we skip and continue.
        try:
            session = fastf1.get_session(year, rnd_int, "R")
            session.load(telemetry=False, weather=False)
        except Exception:
            skipped_events.append(event_label)
            continue

        res = session.results
        if res is None or len(res) == 0:
            skipped_events.append(f"{event_label} (no results)")
            continue

        abbr_col = "Abbreviation" if "Abbreviation" in res.columns else None
        if abbr_col is None:
            skipped_events.append(f"{event_label} (no Abbreviation column)")
            continue
        pts_col = "Points" if "Points" in res.columns else None

        for _, rrow in res.iterrows():
            abbr = str(rrow.get(abbr_col, "") or "").strip()
            if not abbr:
                continue
            if pts_col is not None:
                # Coerce championship points without try/parse: non-numeric → NaN → 0.
                raw_pts = pd.to_numeric(rrow.get(pts_col, 0), errors="coerce")
                pts_f = 0.0 if pd.isna(raw_pts) else float(raw_pts)
                points[abbr] += pts_f
            else:
                # Results table exists but no Points column: keep driver keys at zero.
                points.setdefault(abbr, 0.0)

        for abbr in res[abbr_col].dropna().unique():
            ac = str(abbr).strip()
            if not ac:
                continue
            laps_by_driver[ac].extend(_race_lap_times_seconds(session, ac))

        rounds_processed.append(rnd_int)

    # Championship order: highest total points first (only from loaded races).
    ranked = sorted(points.keys(), key=lambda a: points[a], reverse=True)
    selected: List[str] = []
    padded: List[str] = []

    # Prefer drivers with enough pooled laps so stats are stable.
    for abbr in ranked:
        if len(laps_by_driver.get(abbr, [])) >= min_pooled_laps:
            selected.append(abbr)
        if len(selected) >= top_n:
            break

    # If the grid is thin, pad from championship order even when lap count is low.
    if len(selected) < top_n:
        for abbr in ranked:
            if abbr in selected:
                continue
            selected.append(abbr)
            padded.append(abbr)
            if len(selected) >= top_n:
                break

    if len(selected) < top_n:
        raise ValueError(
            f"Only {len(selected)} drivers available for top_n={top_n} "
            f"(check schedule data for {year})."
        )

    top_stats: List[Dict[str, Any]] = []
    for code in selected[:top_n]:
        lt = laps_by_driver[code]
        arr = np.asarray(lt, dtype=float)
        # Same psychology metrics as ``download_driver_data``, plus season aggregates.
        lap_std = float(np.std(arr)) if arr.size else 0.0
        consistency_inv = 1.0 / (lap_std + 1e-6)
        resilience = float(np.clip(consistency_inv / 10, 0.5, 1.0))
        top_stats.append(
            {
                "code": code,
                "season_points": float(points[code]),
                "n_laps": int(arr.size),
                "lap_std": lap_std,
                "consistency_inv": consistency_inv,
                "resilience": resilience,
                "experience": resilience,
            }
        )

    meta: Dict[str, Any] = {
        "year": year,
        "rounds_processed": rounds_processed,
        "skipped_events": skipped_events,
        "points_by_driver": {k: float(points[k]) for k in sorted(points.keys())},
        "padded_drivers": padded,
        "min_pooled_laps": min_pooled_laps,
        "note": "Top drivers by points from loaded races only; partial season if rounds missing.",
    }
    return top_stats, meta


def spread_experience_from_peers(
    driver_stats: List[Dict[str, Any]],
    low: float = 0.35,
    high: float = 0.95,
) -> List[Dict[str, Any]]:
    """
    Re-map experience (and resilience for plotting) from consistency_inv min–max
    across the listed drivers. Avoids both agents sitting at the clip floor (0.5)
    when per-driver scaling is identical.
    """
    # Pull raw consistency scores from every driver profile.
    vals = [float(d['consistency_inv']) for d in driver_stats]
    vmin, vmax = min(vals), max(vals)
    out: List[Dict[str, Any]] = []
    n = len(driver_stats)
    for i, d in enumerate(driver_stats):
        dd = dict(d)
        # Degenerate case: everyone identical → spread by list index so agents still differ.
        if vmax - vmin < 1e-12:
            t = i / max(1, n - 1) if n > 1 else 0.5
        else:
            # Linear rank in [0, 1] from worst to best consistency within this grid.
            t = (float(d['consistency_inv']) - vmin) / (vmax - vmin)
        # Affine map into [low, high] for the wrapper's noise scaling.
        exp = low + (high - low) * t
        dd['experience'] = float(exp)
        dd['resilience'] = float(exp)
        out.append(dd)
    return out


def parallel_simple_spread_env(driver_stats: List[Dict[str, Any]], **kwargs):
    """simple_spread_v3.parallel_env with N = len(driver_stats) (default N=3 would mismatch)."""
    return simple_spread_v3.parallel_env(N=len(driver_stats), **kwargs)


def build_experience_profiles(
    possible_agents: List[str], driver_stats: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Map PettingZoo agent ids to experience levels from telemetry-derived stats."""
    if len(driver_stats) != len(possible_agents):
        raise ValueError(
            f"driver_stats length ({len(driver_stats)}) must equal "
            f"len(possible_agents) ({len(possible_agents)}). "
            "Use helper.parallel_simple_spread_env(driver_stats, ...) or "
            "parallel_env(N=len(driver_stats), ...)."
        )
    return {
        a: float(s['experience']) for a, s in zip(possible_agents, driver_stats)
    }

# --- TASK 2: BEHAVIORAL WRAPPER ---
class F1PsychologyWrapper(BaseWrapper):
    """
    Injects experience-dependent action noise and a simple scalar ``psych_state``
    into PettingZoo infos so tabular Q can optionally augment the state.

    Set ``action_noise=False`` for MAPPO baselines that should only differ from the
    psych-in-observation run by the extra scalar in the observation vector.
    """

    def __init__(self, env, experience_levels, action_noise: bool = True):
        super().__init__(env)
        # Per-agent baseline skill (from telemetry); higher ⇒ less exploratory noise.
        self.experience_levels = experience_levels
        self.action_noise = bool(action_noise)
        # Mood / confidence proxy, updated from rewards; starts fully "up" each episode.
        self.psych_states = {agent: 1.0 for agent in env.possible_agents}

    def reset(self, seed=None, options=None):
        # PettingZoo parallel API: reset returns team observations plus per-agent info dicts.
        obs, infos = self.env.reset(seed=seed, options=options)
        self.psych_states = {agent: 1.0 for agent in self.env.possible_agents}
        for agent in self.env.possible_agents:
            if agent not in infos:
                infos[agent] = {}
            # Expose psych state to training loop for state augmentation or logging.
            infos[agent]['psych_state'] = self.psych_states[agent]
        return obs, infos

    def step(self, actions):
        modified_actions = {}
        for agent, action in actions.items():
            if not self.action_noise:
                modified_actions[agent] = int(
                    np.clip(int(np.asarray(action).item()), 0, 4)
                )
                continue
            exp = self.experience_levels.get(agent, 0.5)
            # Gaussian perturbation: inexperienced agents get wider noise on discrete actions.
            noise = np.random.normal(0, (1.0 - exp) * 0.1, size=np.array(action).shape)
            # simple_spread uses a small discrete action set; clip after rounding.
            modified_actions[agent] = int(np.clip(np.round(action + noise), 0, 4))

        obs, rewards, terminations, truncations, infos = self.env.step(modified_actions)

        for agent, reward in rewards.items():
            # Large negative team reward ⇒ drop psych state (floor 0.5); else slowly recover.
            if reward < -0.5:
                self.psych_states[agent] = max(0.5, self.psych_states[agent] - 0.2)
            else:
                self.psych_states[agent] = min(1.0, self.psych_states[agent] + 0.05)
            if agent not in infos:
                infos[agent] = {}
            infos[agent]['psych_state'] = self.psych_states[agent]
        # Agents that did not receive a reward key this step still need infos filled.
        for agent in self.psych_states:
            if agent in rewards:
                continue
            if agent not in infos:
                infos[agent] = {}
            infos[agent]['psych_state'] = self.psych_states[agent]

        return obs, rewards, terminations, truncations, infos


class PsychStateObsWrapper(BaseWrapper):
    """
    Append ``infos[agent]['psych_state']`` to each agent's observation so RLlib policies
    can consume it (PettingZoo ``infos`` are not part of the default policy observation).
    Expects an inner env that already writes ``psych_state`` (e.g. ``F1PsychologyWrapper``).
    """

    def observation_space(self, agent):
        base = self.env.observation_space(agent)
        low = np.concatenate([np.asarray(base.low, dtype=np.float32).reshape(-1), [0.5]])
        high = np.concatenate([np.asarray(base.high, dtype=np.float32).reshape(-1), [1.0]])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        out: Dict[str, np.ndarray] = {}
        for agent in self.env.possible_agents:
            vec = np.asarray(obs[agent], dtype=np.float32).reshape(-1)
            p = float(infos.get(agent, {}).get("psych_state", 1.0))
            out[agent] = np.concatenate([vec, np.asarray([p], dtype=np.float32)])
        return out, infos

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        out: Dict[str, np.ndarray] = {}
        for agent in obs:
            vec = np.asarray(obs[agent], dtype=np.float32).reshape(-1)
            p = float(infos.get(agent, {}).get("psych_state", 1.0))
            out[agent] = np.concatenate([vec, np.asarray([p], dtype=np.float32)])
        return out, rewards, terms, truncs, infos


class JointObservationWrapper(BaseWrapper):
    """
    MAPPO-style global information: every agent receives the same vector, the
    concatenation of all agents' (possibly already wrapped) observations in
    ``possible_agents`` order. Enables a shared critic to condition on the joint state
    without custom RLlib model code.
    """

    def __init__(self, env):
        super().__init__(env)
        agents = list(env.possible_agents)
        base = env.observation_space(agents[0])
        d = int(np.prod(base.shape))
        n = len(agents)
        low = np.tile(np.asarray(base.low, dtype=np.float32).reshape(-1), n)
        high = np.tile(np.asarray(base.high, dtype=np.float32).reshape(-1), n)
        self._joint_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._agents_order = agents

    def observation_space(self, agent):
        return self._joint_space

    def _joint_vec(self, obs: Dict[str, Any]) -> np.ndarray:
        parts = [
            np.asarray(obs[a], dtype=np.float32).reshape(-1) for a in self._agents_order
        ]
        return np.concatenate(parts)

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        j = self._joint_vec(obs)
        out = {a: j.copy() for a in self.env.possible_agents}
        return out, infos

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        j = self._joint_vec(obs)
        out = {a: j.copy() for a in obs}
        return out, rewards, terms, truncs, infos


def build_mappo_pettingzoo_parallel(
    *,
    benchmark: bool,
    driver_stats: Optional[List[Dict[str, Any]]] = None,
    psych_in_obs: bool = False,
    action_noise: bool = True,
    joint_obs_for_mappo: bool = True,
    max_cycles: int = 25,
):
    """
    Build a PettingZoo ``ParallelEnv`` for RLlib's ``ParallelPettingZooEnv`` wrapper.

    If ``benchmark`` is True, uses vanilla ``simple_spread_v3`` with ``N=3`` (standard
    MPE cooperative benchmark, no FastF1). Otherwise ``driver_stats`` must match
    ``N`` agents and the F1 psychology stack is applied.

    ``joint_obs_for_mappo`` repeats the concatenation of all agents' observations to
    each agent (centralized state for MAPPO-style training with a shared policy).
    """
    if benchmark:
        env = simple_spread_v3.parallel_env(N=3, max_cycles=max_cycles)
    else:
        if driver_stats is None:
            raise ValueError("driver_stats required when benchmark=False")
        env = parallel_simple_spread_env(driver_stats, max_cycles=max_cycles)
        profiles = build_experience_profiles(env.possible_agents, driver_stats)
        env = F1PsychologyWrapper(env, profiles, action_noise=action_noise)
        if psych_in_obs:
            env = PsychStateObsWrapper(env)
    if joint_obs_for_mappo:
        env = JointObservationWrapper(env)
    return env


def load_driver_stats_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load per-driver rows written by ``export_driver_stats_csv`` (or compatible).
    Returns dicts with numeric fields; caller should run ``spread_experience_from_peers``
    before wiring into the env if experience must be peer-normalized like the notebook.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d: Dict[str, Any] = {"code": row["code"].strip()}
            for k in ("season_points", "n_laps", "lap_std", "consistency_inv", "resilience"):
                if k in row and row[k] != "":
                    d[k] = float(row[k])
            rows.append(d)
    return rows


# --- TASK 3: TABULAR Q-LEARNER ---
class TabularQLearner:
    """One independent Q-table agent (default dict of vectors over actions)."""

    def __init__(self, action_size, lr=0.1, gamma=0.95):
        # Lazy rows: visiting a state creates a zero vector of length ``action_size``.
        self.q_table = collections.defaultdict(lambda: np.zeros(action_size))
        self.lr, self.gamma = lr, gamma

    def get_state(
        self, obs: np.ndarray, psych_state: Optional[float] = None
    ) -> Tuple[Any, ...]:
        # Coarse discretization keeps the table finite for continuous observations.
        base = tuple(np.round(obs, 1).flatten())
        if psych_state is None:
            return base
        # Optional extra dimension when the wrapper exposes psych_state (psych-aware run).
        return base + (round(float(psych_state), 2),)

    def choose(self, state, action_space, eps=0.1):
        # Epsilon-greedy exploration against the PettingZoo action space sampler.
        if np.random.random() < eps:
            return action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def learn(self, s, a, r, next_s):
        # One-step TD backup toward r + gamma * max_a' Q(s', a').
        target = r + self.gamma * np.max(self.q_table[next_s])
        self.q_table[s][a] += self.lr * (target - self.q_table[s][a])


def _agent_state(
    learner: TabularQLearner,
    obs: np.ndarray,
    infos: Dict[str, Dict[str, Any]],
    agent: str,
    use_psych_in_state: bool,
) -> Tuple[Any, ...]:
    # Build the tabular key for this agent at the current decision point.
    if use_psych_in_state:
        psych = infos.get(agent, {}).get('psych_state', 1.0)
        return learner.get_state(obs, psych)
    return learner.get_state(obs)


def train_tabular_parallel_marl(
    env,
    episodes: int,
    use_psych_in_state: bool,
    action_size: int = 5,
    lr: float = 0.1,
    gamma: float = 0.95,
    eps: float = 0.1,
    seed: Optional[int] = None,
    per_agent_rewards: bool = False,
) -> Union[
    Tuple[List[float], Dict[str, TabularQLearner]],
    Tuple[List[float], Dict[str, TabularQLearner], Dict[str, List[float]]],
]:
    """
    Independent Q-learners on a PettingZoo ParallelEnv.
    use_psych_in_state=False: baseline (obs only).
    use_psych_in_state=True: psych_state from infos appended to discretized obs.
    If per_agent_rewards=True, also returns dict mapping agent -> list of episode sums.
    """
    if seed is not None:
        np.random.seed(seed)
    # Independent learners share the environment but not parameters.
    agents = {
        a: TabularQLearner(action_size, lr=lr, gamma=gamma)
        for a in env.possible_agents
    }
    reward_history: List[float] = []
    per_hist: Optional[Dict[str, List[float]]] = None
    if per_agent_rewards:
        per_hist = {a: [] for a in env.possible_agents}

    for ep in range(episodes):
        # Vary reset seed per episode when a base seed is set (reproducible but not identical).
        reset_seed = None if seed is None else int(seed + ep)
        obs, infos = env.reset(seed=reset_seed)
        total_r = 0.0
        ep_by_agent = {a: 0.0 for a in env.possible_agents}
        while env.agents:
            actions = {}
            for a in env.agents:
                st = _agent_state(agents[a], obs[a], infos, a, use_psych_in_state)
                actions[a] = agents[a].choose(st, env.action_space(a), eps=eps)
            n_obs, rewards, terms, truncs, n_infos = env.step(actions)
            for a in actions:
                s = _agent_state(agents[a], obs[a], infos, a, use_psych_in_state)
                next_s = _agent_state(
                    agents[a], n_obs[a], n_infos, a, use_psych_in_state
                )
                agents[a].learn(s, actions[a], rewards[a], next_s)
                total_r += rewards[a]
                ep_by_agent[a] += rewards[a]
            obs, infos = n_obs, n_infos
        reward_history.append(total_r)
        if per_hist is not None:
            for a in env.possible_agents:
                per_hist[a].append(float(ep_by_agent[a]))

    if per_hist is None:
        return reward_history, agents
    return reward_history, agents, per_hist


def strategy_resilience_score(
    reward_history: List[float],
    early_frac: float = 0.2,
    late_frac: float = 0.2,
) -> float:
    """
    Normalized late-vs-early improvement in [0, 1]. Compares mean return in the last
    late_frac episodes to the first early_frac; maps (late_mean - early_mean) / std
    through tanh so mixed-sign windows (where late/early ratio is meaningless) do not
    collapse to 0. No change ~ 0.5; sustained improvement ~ > 0.5.
    """
    n = len(reward_history)
    if n < 5:
        return 0.0
    e = max(1, int(n * early_frac))
    l = max(1, int(n * late_frac))
    early = float(np.mean(reward_history[:e]))
    late = float(np.mean(reward_history[-l:]))
    spread = float(np.std(reward_history))
    if spread < 1e-8:
        spread = 1e-8
    # Standardize improvement by run volatility; tanh maps to a bounded score near 0.5.
    z = (late - early) / spread
    return float(np.clip(0.5 + 0.5 * np.tanh(z), 0.0, 1.0))


# --- CHECKPOINT EXPERIMENTS & REPORTING ---


def subset_driver_stats(driver_stats: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """First k drivers in list order (e.g. championship order from collect_season_driver_data)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > len(driver_stats):
        raise ValueError(f"k={k} exceeds len(driver_stats)={len(driver_stats)}")
    return [dict(d) for d in driver_stats[:k]]


def learning_curve_summary(
    reward_history: List[float],
    early_frac: float = 0.2,
    late_frac: float = 0.2,
    tail: int = 20,
) -> Dict[str, float]:
    """Scalar summaries for tables / ablation writeups."""
    arr = np.asarray(reward_history, dtype=float)
    n = arr.size
    if n == 0:
        return {}
    # Window sizes for "early" and "late" segments (same idea as SRS, but raw means here).
    e = max(1, int(n * early_frac))
    l = max(1, int(n * late_frac))
    tail_k = min(tail, n)
    return {
        "n_episodes": float(n),
        "mean_all": float(np.mean(arr)),
        "std_all": float(np.std(arr)),
        "mean_early": float(np.mean(arr[:e])),
        "mean_late": float(np.mean(arr[-l:])),
        "mean_last_k": float(np.mean(arr[-tail_k:])),
        "best_episode": float(np.max(arr)),
        "worst_episode": float(np.min(arr)),
    }


def bootstrap_srs_interval(
    reward_history: List[float],
    n_bootstrap: int = 400,
    seed: Optional[int] = None,
    early_frac: float = 0.2,
    late_frac: float = 0.2,
) -> Dict[str, float]:
    """
    Resample episodes with replacement; distribution of strategy_resilience_score.
    Returns mean and 5th/95th percentiles of SRS over bootstrap draws.
    """
    rh = list(reward_history)
    n = len(rh)
    if n < 5:
        return {"srs_mean": 0.0, "srs_p05": 0.0, "srs_p95": 0.0, "n_bootstrap": 0.0}
    rng = np.random.default_rng(seed)
    scores: List[float] = []
    for _ in range(n_bootstrap):
        # Resample entire episode indices with replacement → distribution over SRS.
        sample = [rh[i] for i in rng.integers(0, n, size=n)]
        scores.append(strategy_resilience_score(sample, early_frac, late_frac))
    sa = np.asarray(scores, dtype=float)
    return {
        "srs_mean": float(np.mean(sa)),
        "srs_p05": float(np.percentile(sa, 5)),
        "srs_p95": float(np.percentile(sa, 95)),
        "n_bootstrap": float(n_bootstrap),
    }


def make_wrapped_env_factory(
    driver_stats_spread: List[Dict[str, Any]],
    max_cycles: int = 25,
):
    """
    Returns a zero-arg factory: each call builds a fresh ParallelEnv + profiles + wrapper.
    Use this for multi-seed runs so each seed gets a new env and Q-tables via train_*.
    """
    # Open once to read canonical agent names; experience map is fixed across seeds.
    probe = parallel_simple_spread_env(driver_stats_spread, max_cycles=max_cycles)
    profiles = build_experience_profiles(probe.possible_agents, driver_stats_spread)
    probe.close()

    def _factory():
        # Fresh env each call so parallel training runs do not share internal RNG/state.
        inner = parallel_simple_spread_env(driver_stats_spread, max_cycles=max_cycles)
        return F1PsychologyWrapper(inner, profiles)

    return _factory


def run_baseline_vs_psych_once(
    env_factory,
    episodes: int,
    seed: int,
    **train_kw: Any,
) -> Dict[str, Any]:
    """One seed: train baseline then psych-aware; return histories, SRS, summaries."""
    train_kw = dict(train_kw)
    # Same seed, same factory: baseline uses obs-only Q-states; psych uses augmented states.
    env_b = env_factory()
    h_b, _ = train_tabular_parallel_marl(
        env_b, episodes, False, seed=seed, **train_kw
    )
    env_b.close()
    env_p = env_factory()
    h_p, _ = train_tabular_parallel_marl(
        env_p, episodes, True, seed=seed, **train_kw
    )
    env_p.close()
    return {
        "seed": seed,
        "reward_baseline": h_b,
        "reward_psych": h_p,
        "srs_baseline": strategy_resilience_score(h_b),
        "srs_psych": strategy_resilience_score(h_p),
        "summary_baseline": learning_curve_summary(h_b),
        "summary_psych": learning_curve_summary(h_p),
        "bootstrap_baseline": bootstrap_srs_interval(h_b, seed=seed),
        "bootstrap_psych": bootstrap_srs_interval(h_p, seed=seed + 1),
    }


def multi_seed_experiment(
    env_factory,
    seeds: Tuple[int, ...],
    episodes: int,
    **train_kw: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Repeat run_baseline_vs_psych_once for each seed.
    Returns (per_run_dicts, aggregates) where aggregates has mean/std SRS for B and P.
    """
    runs: List[Dict[str, Any]] = []
    sb: List[float] = []
    sp: List[float] = []
    for s in seeds:
        r = run_baseline_vs_psych_once(env_factory, episodes, int(s), **train_kw)
        runs.append(r)
        sb.append(r["srs_baseline"])
        sp.append(r["srs_psych"])
    agg = {
        "srs_baseline_mean": float(np.mean(sb)),
        "srs_baseline_std": float(np.std(sb)),
        "srs_psych_mean": float(np.mean(sp)),
        "srs_psych_std": float(np.std(sp)),
        "n_seeds": float(len(seeds)),
    }
    return runs, agg


def epsilon_comparison_run(
    env_factory,
    episodes: int,
    seed: int,
    epsilons: Tuple[float, ...] = (0.05, 0.1, 0.2),
    psych: bool = False,
    **train_kw: Any,
) -> List[Dict[str, Any]]:
    """Sweep exploration rate for one condition (baseline or psych-aware)."""
    rows: List[Dict[str, Any]] = []
    for eps in epsilons:
        env = env_factory()
        h, _ = train_tabular_parallel_marl(
            env,
            episodes,
            psych,
            seed=seed,
            eps=float(eps),
            **{k: v for k, v in train_kw.items() if k != "eps"},
        )
        env.close()
        rows.append(
            {
                "eps": float(eps),
                "psych": psych,
                "srs": strategy_resilience_score(h),
                "mean_last_20": learning_curve_summary(h, tail=20)["mean_last_k"],
            }
        )
    return rows


def export_driver_stats_csv(
    driver_stats: List[Dict[str, Any]],
    path: str,
) -> None:
    """Write code, season_points, n_laps, lap_std, resilience to CSV (no pandas)."""
    keys = ["code", "season_points", "n_laps", "lap_std", "consistency_inv", "resilience"]
    lines = [",".join(keys)]
    for d in driver_stats:
        parts = []
        for k in keys:
            v = d.get(k, "")
            if isinstance(v, float):
                parts.append(f"{v:.6g}")
            else:
                parts.append(str(v))
        lines.append(",".join(parts))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def q_table_size(learners: Dict[str, TabularQLearner]) -> Dict[str, int]:
    """Number of visited states per agent (tabular capacity proxy)."""
    return {a: len(learners[a].q_table) for a in learners}