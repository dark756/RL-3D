"""Episode logs (brief sections 6 and 9, D10, D22): one JSON file per episode plus an .npz of arrays.

JSON (readable; floats written exactly): log format, when it was written, the code version (git commit
and whether there were uncommitted changes, D22), Python / numpy / MuJoCo / Gymnasium versions, every fact
of the config with its status and origin (overrides included, D5), the env seed, the episode's resolved
start (`MarsLanderEnv` info["episode"]: initial state, conditions, sim seed), the outcome and the return.
NPZ: the actions exactly as the agent sent them, the rewards, and optionally the true trajectory
(`TRAJECTORY_COLUMNS`), so a replay can be checked number for number.
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_ENV_PATH, Config, Param, build_config, parse_env_file

LOG_FORMAT = 1
ROOT = Path(__file__).resolve().parent.parent
TRAJECTORY_COLUMNS = ("t", "x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz", "propellant_kg")


@functools.lru_cache(maxsize=None)
def code_version(root: str = str(ROOT)) -> dict[str, Any]:
    """{'commit': hash, 'dirty': uncommitted changes?} from git, or {'commit': 'unversioned', 'dirty': None} (D22).

    Read once per process: a run does not change its own code.
    """
    try:
        run = functools.partial(subprocess.run, cwd=root, capture_output=True, text=True, timeout=10, check=True)
        commit = run(["git", "rev-parse", "HEAD"]).stdout.strip()
        dirty = bool(run(["git", "status", "--porcelain"]).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unversioned", "dirty": None}


def versions() -> dict[str, str]:
    """Interpreter, platform and package versions (a replay is bit-identical only with the same ones, D10)."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("mujoco", "numpy", "gymnasium"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "missing"
    return out


def trajectory_row(telemetry) -> np.ndarray:
    """One row of TRAJECTORY_COLUMNS from a Telemetry snapshot (the truth)."""
    tl = telemetry
    return np.concatenate([[tl.t], tl.cg, tl.vel, tl.quat, tl.omega_b, [tl.propellant_kg]])


def to_jsonable(x: Any) -> Any:
    """numpy arrays and scalars, tuples and nested containers -> plain JSON types."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def episode_record(cfg: Config, seed: int | None, episode: dict, outcome: dict | None, ret: float, steps: int,
                   **extra: Any) -> dict[str, Any]:
    """Everything the JSON file holds about one episode."""
    return {"log_format": LOG_FORMAT, "written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "code": code_version(), "versions": versions(), "env_path": cfg.env_path,
            "config": cfg.to_log_dict(), "seed": seed, "episode": to_jsonable(episode),
            "outcome": to_jsonable(outcome), "return": float(ret), "steps": int(steps), **to_jsonable(extra)}


def write_episode(stem: str | Path, record: dict, arrays: dict[str, np.ndarray]) -> Path:
    """Write <stem>.json and <stem>.npz; returns the JSON path."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    npz = stem.with_name(stem.name + ".npz")
    np.savez_compressed(npz, **arrays)
    path = stem.with_name(stem.name + ".json")
    path.write_text(json.dumps(to_jsonable({**record, "arrays": npz.name}), indent=1), encoding="utf-8")
    return path


def read_episode(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """The record and arrays of a logged episode, from its JSON path."""
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    with np.load(path.parent / record["arrays"]) as z:
        arrays = {k: z[k] for k in z.files}
    return record, arrays


def keys_missing_from_log(record: dict, env_path: str | Path | None = None) -> list[str]:
    """Facts in the current `.env` that the log does not have, because they were added after it was written."""
    return sorted(set(parse_env_file(env_path or DEFAULT_ENV_PATH)) - set(record["config"]))


def config_from_log(record: dict, env_path: str | Path | None = None) -> Config:
    """The exact config an episode ran with: every value, status, source and origin as logged.

    Facts added to `.env` after the log was written (e.g. M3's controller settings in an M2 log) come from the
    current `.env`; `keys_missing_from_log` names them, and the replay script warns about them.
    """
    params = {}
    for key, d in record["config"].items():
        value = tuple(float(v) for v in d["value"]) if isinstance(d["value"], list) else d["value"]
        params[key] = Param(key, value, d["unit"], d["status"], d["source"], d["note"], d["origin"])
    current = parse_env_file(env_path or DEFAULT_ENV_PATH)
    for key in sorted(set(current) - set(params)):
        params[key] = current[key]
    return build_config(params, record.get("env_path", ""))
