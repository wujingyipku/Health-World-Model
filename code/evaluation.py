"""Comprehensive health world model evaluation"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from code.data import build_loader
from code.nn import JEPAAgent
from code.specs import (
    ModelSpec,
    continuous_reward_standardize_info,
    materialize_pooled_binary_rewards,
    resolve_pooled_binary_rewards,
)
from code.trainer import (
    Trainer,
    encode_batch_observation,
    encode_batch_static,
    encode_batch_trajectory,
)

try:
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import average_precision_score
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for evaluate_and_plot_hrs_jepa.py. "
        "Install with: pip install scikit-learn"
    ) from exc


# Outcomes in the released model. Worsening flags are next-wave 0/1 events.
WORSENING_REWARDS: tuple[str, ...] = (
    "adl_worsening",
    "iadl_worsening",
)
CONTINUOUS_REWARDS: tuple[str, ...] = ()
# Acute-event group: 1 = the event occurred in the interval.
CLINICAL_EVENT_REWARDS: tuple[str, ...] = ("death_event",)
FIG6_SIMULATION_EVENTS: tuple[str, ...] = ("death_event",)
# Kaplan–Meier row order (only events present in the table are drawn).
KM_PANEL_ORDER: tuple[str, ...] = ("death_event",)
EVENT_LABELS = {
    "death_event": "Death",
    "adl_worsening": "ADL worsened",
    "iadl_worsening": "IADL worsened",
}
# Backward-compat alias: older plot/collector code imported LEVEL_OUTCOMES.
LEVEL_OUTCOMES = EVENT_LABELS


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(name)


def selected_checkpoint(run_dir: Path | None, explicit: str | Path | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.is_absolute():
            resolved = path.resolve()
        elif run_dir is not None:
            in_run = Path(run_dir).resolve() / path
            if in_run.exists():
                resolved = in_run
            elif path.exists():
                resolved = path.resolve()
            else:
                resolved = in_run
        elif path.exists():
            resolved = path.resolve()
        else:
            resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {resolved}"
                + (f" (run_dir={Path(run_dir).resolve()})" if run_dir else "")
            )
        return resolved
    if run_dir is None:
        raise ValueError("Provide --checkpoint or --run-dir")
    run_dir = Path(run_dir).resolve()
    for name in (
        "best_final.pt",
        "best_clinical.pt",
        "best_jepa.pt",
        "best_death_finetune.pt",
        # Legacy names from older runs.
        "checkpoint_world_best.pt",
        "checkpoint_final.pt",
    ):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No best_final.pt / best_clinical.pt / best_jepa.pt in {run_dir}"
    )


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def parquet_columns(path: Path) -> set[str]:
    """Schema column names without loading the full table."""
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        return set(pd.read_parquet(path).columns)


# Bumped whenever the reward definitions change in a way that invalidates a
# fitted baseline. This release evaluates death, ADL worsening, and IADL worsening.
REWARD_REPRESENTATION = "death_adl_iadl"
# Match JEPA ``world_loss.binary_reward_pos_weight_max``: class weight for the
# positive class ≈ n_neg/n_pos, clamped to [1, max]. <=0 disables weighting.
DEFAULT_BINARY_POS_WEIGHT_MAX = 10.0


def baseline_cache_matches_features(
    manifest: Mapping[str, Any] | None,
    allowed_feature_names: Sequence[str] | None,
) -> bool:
    """True if the cache used the same ModelSpec feature restriction."""
    if allowed_feature_names is None:
        return True
    if not manifest:
        return False
    cached = manifest.get("allowed_feature_names")
    if cached is None:
        return False
    return set(str(x) for x in cached) == set(str(x) for x in allowed_feature_names)


def baseline_cache_matches_rewards(manifest: Mapping[str, Any] | None) -> bool:
    """True if a cached baseline was fitted against the current reward semantics.

    Pre-``binary_worsening`` caches fitted ``baseline_prob__adl_worsening`` on a
    *level* cutoff (ADL > 0), not on "worsened vs not", so the column name alone
    cannot tell the two apart — only the manifest can.
    """
    if not manifest:
        return False
    return str(manifest.get("reward_representation", "")) == REWARD_REPRESENTATION


def binary_pos_class_weight(
    y: np.ndarray,
    *,
    max_weight: float = DEFAULT_BINARY_POS_WEIGHT_MAX,
) -> dict[int, float] | None:
    """Sklearn ``class_weight`` dict matching JEPA BCE ``pos_weight``.

    ``{0: 1.0, 1: clip(n_neg/n_pos, 1, max_weight)}``. Returns ``None`` when
    ``max_weight <= 0`` (unweighted fit).
    """
    if float(max_weight) <= 0:
        return None
    y_i = np.asarray(y, dtype=int).reshape(-1)
    n_pos = int((y_i == 1).sum())
    n_neg = int((y_i == 0).sum())
    if n_pos <= 0:
        return {0: 1.0, 1: float(max_weight)}
    weight = float(np.clip(n_neg / max(n_pos, 1), 1.0, float(max_weight)))
    return {0: 1.0, 1: weight}


def binary_pos_sample_weights(
    y: np.ndarray,
    *,
    max_weight: float = DEFAULT_BINARY_POS_WEIGHT_MAX,
) -> np.ndarray | None:
    """Per-row sample weights for estimators without ``class_weight`` (MLP)."""
    cw = binary_pos_class_weight(y, max_weight=max_weight)
    if cw is None:
        return None
    y_i = np.asarray(y, dtype=int).reshape(-1)
    weights = np.ones(y_i.shape[0], dtype=np.float64)
    weights[y_i == 1] = float(cw[1])
    weights[y_i == 0] = float(cw[0])
    return weights


def expand_binary_by_pos_weight(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_weight: float = DEFAULT_BINARY_POS_WEIGHT_MAX,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Duplicate positive rows ≈ ``pos_weight`` times (MLP has no sample_weight).

    Returns ``(x_exp, y_exp, pos_weight_or_None)``. Negatives stay once; each
    positive is repeated ``round(clip(n_neg/n_pos, 1, max))`` times, then
    rows are shuffled. Matches Logistic ``class_weight`` ratio on older sklearn
    where ``MLPClassifier.fit`` rejects ``sample_weight``.
    """
    cw = binary_pos_class_weight(y, max_weight=max_weight)
    if cw is None:
        return x, y, None
    y_i = np.asarray(y, dtype=int).reshape(-1)
    pos_w = float(cw[1])
    repeats = max(int(round(pos_w)), 1)
    pos_idx = np.flatnonzero(y_i == 1)
    neg_idx = np.flatnonzero(y_i == 0)
    if pos_idx.size == 0:
        return x, y, pos_w
    pos_exp = np.repeat(pos_idx, repeats)
    idx = np.concatenate([neg_idx, pos_exp])
    rng = np.random.RandomState(int(seed))
    rng.shuffle(idx)
    return x[idx], y_i[idx], pos_w


def baseline_cache_matches_pos_weight(
    manifest: Mapping[str, Any] | None,
    max_weight: float,
) -> bool:
    """True if cache was fit with the same capped positive-class weight policy."""
    if not manifest or "binary_pos_weight_max" not in manifest:
        return False
    try:
        cached = float(manifest["binary_pos_weight_max"])
    except (TypeError, ValueError):
        return False
    return abs(cached - float(max_weight)) < 1e-9


def inverse_standardized(value: np.ndarray | pd.Series, info: Mapping[str, Any]) -> np.ndarray:
    value_np = np.asarray(value, dtype=float)
    transformed = value_np * float(info.get("std", 1.0)) + float(info.get("mean", 0.0))
    transform = str(info.get("transform", "identity"))
    if transform == "log1p":
        return np.expm1(transformed)
    if transform in {"signed_log1p", "signed-log1p"}:
        return np.sign(transformed) * np.expm1(np.abs(transformed))
    return transformed


def load_agent(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[JEPAAgent, ModelSpec, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    spec = ModelSpec.from_dict(checkpoint["model_spec"])
    cfg = checkpoint["config"]
    agent = JEPAAgent(spec, cfg["model"]).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    return agent, spec, cfg


def schema_compatibility_warnings(test_path: Path, spec: ModelSpec) -> list[str]:
    columns = set(read_table(test_path).columns)
    warnings: list[str] = []
    pooled = spec.pooled_binary_rewards or {}
    for name in spec.reward_names:
        if name in pooled:
            missing_src = [
                f"reward__{src}"
                for src in pooled[name]
                if f"reward__{src}" not in columns
            ]
            if missing_src:
                warnings.append(
                    f"Checkpoint pooled reward '{name}' needs {', '.join(missing_src)} "
                    "on the test table; sample-level evaluation for this outcome "
                    "is omitted."
                )
            continue
        missing = [
            column
            for column in (f"reward__{name}", f"reward_mask__{name}")
            if column not in columns
        ]
        if missing:
            warnings.append(
                f"Checkpoint expects reward '{name}', but the current test table "
                f"does not contain {', '.join(missing)}; sample-level evaluation "
                "for this outcome is omitted."
            )
    return warnings


def transition_key_frame(batch: Any, t: int, active: np.ndarray) -> pd.DataFrame:
    pids = np.asarray(batch.person_id, dtype=object)[active]
    wave = batch.wave[:, t].detach().cpu().numpy()[active].astype(int)
    next_wave = batch.next_wave[:, t].detach().cpu().numpy()[active].astype(int)
    return pd.DataFrame({"person_id": pids.astype(str), "wave": wave, "next_wave": next_wave})


def _event_labels_for_spec(spec: ModelSpec) -> dict[str, str]:
    return {name: EVENT_LABELS.get(name, name) for name in spec.reward_binary}


@torch.inference_mode()
def collect_one_step_predictions(
    agent: JEPAAgent,
    loader: DataLoader,
    preprocessing: Mapping[str, Any],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export teacher-forced one-step JEPA predictions.

    Temporal Transformer → z_t → action-conditioned predictor → z_future → heads.
    Returns ``(levels, events)``: continuous rewards in original units, binary
    events as probabilities.
    """
    world = agent.world
    event_rows: list[pd.DataFrame] = []
    level_rows: list[pd.DataFrame] = []
    reward_binary_lookup = {name: idx for idx, name in enumerate(agent.spec.reward_binary)}
    event_labels = _event_labels_for_spec(agent.spec)
    n_bin = len(agent.spec.reward_binary)

    for raw_batch in loader:
        batch = raw_batch.to(device)
        _, steps = batch.valid.shape
        z_seq = encode_batch_trajectory(world, batch)
        static_embed = encode_batch_static(world, batch, t=0)
        for t in range(steps):
            valid = batch.valid[:, t]
            feature = world.predict_next(
                z_seq[:, t],
                batch.action_cont[:, t],
                batch.action_cont_mask[:, t],
                batch.action_cat[:, t],
                batch.delta_t_norm[:, t],
                static_embed=static_embed,
            )
            reward_pred = world.clinical_heads.mean_dict(
                feature,
                state_cont=batch.state_cont[:, t],
                state_cont_mask=batch.state_cont_mask[:, t],
            )

            for name, label in event_labels.items():
                if name not in reward_binary_lookup or name not in reward_pred:
                    continue
                idx = reward_binary_lookup[name]
                active_t = (valid > 0.5) & (batch.reward_mask[:, t, idx] > 0.5)
                active = active_t.detach().cpu().numpy().astype(bool)
                if not active.any():
                    continue
                keys = transition_key_frame(batch, t, active)
                keys["event"] = name
                keys["label"] = label
                keys["observed"] = batch.reward[:, t, idx].detach().cpu().numpy()[active]
                keys["predicted_prob"] = reward_pred[name].detach().cpu().numpy()[active]
                event_rows.append(keys)

            for j, name in enumerate(agent.spec.reward_continuous):
                if name not in reward_pred:
                    continue
                ridx = n_bin + j
                _, info = continuous_reward_standardize_info(
                    name,
                    preprocessing,
                    agent.spec.state_continuous,
                )
                active_t = (valid > 0.5) & (batch.reward_mask[:, t, ridx] > 0.5)
                active = active_t.detach().cpu().numpy().astype(bool)
                if not active.any():
                    continue
                obs_std = batch.reward[:, t, ridx].detach().cpu().numpy()[active]
                pred_std = reward_pred[name].detach().cpu().numpy()[active]
                keys = transition_key_frame(batch, t, active)
                keys["outcome"] = name
                keys["label"] = EVENT_LABELS.get(name, name)
                if info:
                    keys["observed_raw"] = inverse_standardized(obs_std, info)
                    keys["predicted_raw"] = inverse_standardized(pred_std, info)
                else:
                    keys["observed_raw"] = obs_std
                    keys["predicted_raw"] = pred_std
                level_rows.append(keys)

    empty_keys = pd.DataFrame(columns=["person_id", "wave", "next_wave"])
    return (
        pd.concat(level_rows, ignore_index=True) if level_rows else empty_keys.copy(),
        pd.concat(event_rows, ignore_index=True) if event_rows else empty_keys.copy(),
    )


@torch.inference_mode()
def collect_free_rollouts(
    agent: JEPAAgent,
    loader: DataLoader,
    preprocessing: Mapping[str, Any],
    device: torch.device,
    max_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Latent JEPA rollout for events only (no state decoder).

    z0 from Temporal Transformer on Wave1; then
    z_{t+1}=predict_next(z_t, a_t, Δt) in embedding space.
    """
    world = agent.world
    event_rows: list[pd.DataFrame] = []
    event_lookup = {name: idx for idx, name in enumerate(agent.spec.reward_binary)}
    event_labels = _event_labels_for_spec(agent.spec)
    _ = preprocessing  # kept for API compatibility with Dreamer eval

    for raw_batch in loader:
        batch = raw_batch.to(device)
        batch_size, steps = batch.valid.shape
        use_steps = min(steps, max_horizon)
        z_seq = encode_batch_trajectory(world, batch)
        z = z_seq[:, 0]
        static_embed = encode_batch_static(world, batch, t=0)
        elapsed = torch.zeros(batch_size, device=device)

        for t in range(use_steps):
            valid = batch.valid[:, t]
            feature = world.predict_next(
                z,
                batch.action_cont[:, t],
                batch.action_cont_mask[:, t],
                batch.action_cat[:, t],
                batch.delta_t_norm[:, t],
                static_embed=static_embed,
            )
            reward_pred = world.clinical_heads.mean_dict(
                feature,
                state_cont=batch.state_cont[:, t],
                state_cont_mask=batch.state_cont_mask[:, t],
            )
            elapsed = elapsed + batch.delta_t_raw[:, t, 0]
            z = feature

            for name, label in event_labels.items():
                if name not in event_lookup or name not in reward_pred:
                    continue
                idx = event_lookup[name]
                active_t = (valid > 0.5) & (batch.reward_mask[:, t, idx] > 0.5)
                active = active_t.detach().cpu().numpy().astype(bool)
                if not active.any():
                    continue
                event_rows.append(
                    pd.DataFrame(
                        {
                            "person_id": np.asarray(batch.person_id, dtype=object)[active].astype(str),
                            "start_wave": batch.wave[:, 0].detach().cpu().numpy()[active].astype(int),
                            "target_wave": batch.next_wave[:, t].detach().cpu().numpy()[active].astype(int),
                            "horizon": t + 1,
                            "elapsed_years": elapsed.detach().cpu().numpy()[active],
                            "event": name,
                            "label": label,
                            "observed": batch.reward[:, t, idx].detach().cpu().numpy()[active],
                            "predicted_prob": reward_pred[name].detach().cpu().numpy()[active],
                        }
                    )
                )

    empty = pd.DataFrame()
    return empty, (
        pd.concat(event_rows, ignore_index=True) if event_rows else empty
    )


def baseline_feature_columns(
    columns: Iterable[str],
    allowed_names: Sequence[str] | None = None,
) -> list[str]:
    prefixes = ("state__", "state_mask__", "action__", "action_mask__")
    cols = [c for c in columns if c.startswith(prefixes)]
    if allowed_names is None:
        return sorted(cols)
    allowed = {str(x) for x in allowed_names}
    kept: list[str] = []
    for col in cols:
        for prefix in prefixes:
            if col.startswith(prefix) and col[len(prefix) :] in allowed:
                kept.append(col)
                break
    return sorted(kept)


def discover_binary_event_names(
    columns: Iterable[str],
    preferred: Sequence[str] | None = None,
) -> list[str]:
    """Binary event names present in a transition table (optionally filtered by spec)."""
    column_set = set(columns)
    skip = set(CONTINUOUS_REWARDS)
    candidates = list(preferred) if preferred else list(EVENT_LABELS)
    return [
        name
        for name in candidates
        if name not in skip and f"reward__{name}" in column_set
    ]


def _looks_bernoulli(values: np.ndarray, mask: np.ndarray) -> bool:
    """True if masked values are 0/1 (allowing float 0.0/1.0)."""
    active = np.asarray(mask, dtype=float) > 0.5
    y = np.asarray(values, dtype=float)[active]
    y = y[np.isfinite(y)]
    if y.size == 0:
        return False
    uniq = np.unique(np.round(y, 6))
    return bool(np.all(np.isin(uniq, [0.0, 1.0])))


def baseline_cache_paths(baseline_dir: Path) -> dict[str, Path]:
    baseline_dir = Path(baseline_dir)
    return {
        "predictions": baseline_dir / "linear_baseline_predictions.parquet",
        "manifest": baseline_dir / "baseline_manifest.json",
    }


def legacy_baseline_prediction_path(output_dir: Path) -> Path:
    """Previous location used before baseline_cache was split out."""
    return Path(output_dir) / "prediction_cache" / "linear_baseline_predictions.parquet"


def fit_linear_baselines(
    train_path: Path,
    test_path: Path,
    preprocessing: Mapping[str, Any],
    seed: int,
    skip: bool,
    event_names: Sequence[str],
    pooled_binary_rewards: Mapping[str, Sequence[str]] | None = None,
    binary_pos_weight_max: float = DEFAULT_BINARY_POS_WEIGHT_MAX,
    allowed_feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    _ = preprocessing
    train = read_table(train_path)
    test = read_table(test_path)
    keys = ["person_id", "wave", "next_wave"]
    result = test[keys].copy()
    result["person_id"] = result["person_id"].astype(str)
    result["wave"] = result["wave"].astype(int)
    result["next_wave"] = result["next_wave"].astype(int)
    features = baseline_feature_columns(
        train.columns, allowed_names=allowed_feature_names
    )
    if allowed_feature_names is not None:
        print(
            f"Logistic features restricted to ModelSpec "
            f"({len(allowed_feature_names)} names, {len(features)} columns)",
            flush=True,
        )
    x_train = np.nan_to_num(train[features].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(test[features].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pooled = resolve_pooled_binary_rewards(event_names, explicit=pooled_binary_rewards)
    train = materialize_pooled_binary_rewards(train, pooled)
    test = materialize_pooled_binary_rewards(test, pooled)
    weight_notes: list[str] = []

    def fit_binary(observed: np.ndarray, mask: np.ndarray, name: str) -> np.ndarray | None:
        if not _looks_bernoulli(observed, mask):
            return None
        active = (np.asarray(mask, dtype=float) > 0.5) & np.isfinite(observed)
        y = np.round(np.asarray(observed[active], dtype=float)).astype(int)
        if y.size == 0 or int(y.min()) < 0 or int(y.max()) > 1:
            return None
        prevalence = float(y.mean()) if len(y) else 0.0
        if skip or len(np.unique(y)) < 2 or int(np.bincount(y).min()) < 10:
            return np.full(len(test), prevalence, dtype=np.float32)
        class_weight = binary_pos_class_weight(y, max_weight=binary_pos_weight_max)
        if class_weight is not None:
            weight_notes.append(f"{name}={class_weight[1]:.3f}")
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=2e-4,
            max_iter=2000,
            tol=1e-4,
            average=True,
            random_state=seed,
            class_weight=class_weight,
        )
        clf.fit(x_train[active], y)
        return clf.predict_proba(x_test)[:, 1].astype(np.float32)

    for name in event_names:
        if name in CONTINUOUS_REWARDS:
            continue
        value_col = f"reward__{name}"
        mask_col = f"reward_mask__{name}"
        if value_col not in train or mask_col not in train:
            continue
        observed = pd.to_numeric(train[value_col], errors="coerce").fillna(0).to_numpy()
        mask = pd.to_numeric(train[mask_col], errors="coerce").fillna(0).to_numpy()
        fitted = fit_binary(observed, mask, name)
        if fitted is None:
            continue
        result[f"baseline_prob__{name}"] = fitted

    if weight_notes:
        print(
            "Logistic class_weight pos "
            f"(max={binary_pos_weight_max}): " + ", ".join(weight_notes),
            flush=True,
        )
    elif float(binary_pos_weight_max) <= 0:
        print("Logistic class_weight disabled (binary_pos_weight_max<=0)", flush=True)

    return result


def prepare_baselines(
    *,
    train_data: Path,
    test_data: Path,
    preprocessing_path: Path,
    baseline_dir: Path,
    seed: int = 2026,
    skip_linear_baseline: bool = False,
    event_names: Sequence[str] | None = None,
    binary_pos_weight_max: float = DEFAULT_BINARY_POS_WEIGHT_MAX,
    allowed_feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit linear baselines once and persist them under ``baseline_dir``."""
    train_data = Path(train_data).resolve()
    test_data = Path(test_data).resolve()
    preprocessing_path = Path(preprocessing_path).resolve()
    baseline_dir = Path(baseline_dir).resolve()
    baseline_dir.mkdir(parents=True, exist_ok=True)
    paths = baseline_cache_paths(baseline_dir)

    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    if event_names is None:
        event_names = discover_binary_event_names(read_table(train_data).columns)
    else:
        event_names = [n for n in event_names if n not in CONTINUOUS_REWARDS]

    print(f"Fitting linear baselines -> {paths['predictions']}", flush=True)
    baseline = fit_linear_baselines(
        train_data,
        test_data,
        preprocessing,
        seed,
        skip_linear_baseline,
        event_names=list(event_names),
        binary_pos_weight_max=float(binary_pos_weight_max),
        allowed_feature_names=allowed_feature_names,
    )
    baseline.to_parquet(paths["predictions"], index=False)

    prob_columns = sorted(c for c in baseline.columns if c.startswith("baseline_prob__"))
    manifest = {
        "train_data": str(train_data),
        "test_data": str(test_data),
        "preprocessing": str(preprocessing_path),
        "seed": seed,
        "skip_linear_baseline": bool(skip_linear_baseline),
        "event_names": list(event_names),
        "probability_columns": prob_columns,
        "reward_representation": REWARD_REPRESENTATION,
        "binary_pos_weight_max": float(binary_pos_weight_max),
        "class_weight_scheme": "neg_over_pos_capped",
        "n_rows": int(len(baseline)),
        "predictions_path": str(paths["predictions"]),
        "allowed_feature_names": (
            list(allowed_feature_names) if allowed_feature_names is not None else None
        ),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote baseline manifest {paths['manifest']}", flush=True)
    return {
        "baseline_dir": str(baseline_dir),
        "predictions_path": str(paths["predictions"]),
        "manifest_path": str(paths["manifest"]),
        "manifest": manifest,
        "baseline": baseline,
    }


def load_or_prepare_baselines(
    *,
    train_data: Path | None,
    test_data: Path,
    preprocessing_path: Path,
    baseline_dir: Path,
    output_dir: Path,
    seed: int,
    skip_linear_baseline: bool,
    refit_baselines: bool,
    event_names: Sequence[str] | None,
    binary_pos_weight_max: float = DEFAULT_BINARY_POS_WEIGHT_MAX,
    allowed_feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load cached baselines; fit only when missing or ``refit_baselines`` is set."""
    paths = baseline_cache_paths(baseline_dir)
    legacy = legacy_baseline_prediction_path(output_dir)
    pos_max = float(binary_pos_weight_max)

    if not refit_baselines and paths["predictions"].exists():
        manifest: dict[str, Any] = {}
        if paths["manifest"].exists():
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        cached = pd.read_parquet(paths["predictions"])
        have_events = {
            c[len("baseline_prob__") :]
            for c in cached.columns
            if c.startswith("baseline_prob__")
        }
        need_events = {str(x) for x in (event_names or [])}
        events_ok = not need_events or need_events.issubset(have_events)
        pos_ok = baseline_cache_matches_pos_weight(manifest, pos_max)
        features_ok = baseline_cache_matches_features(manifest, allowed_feature_names)
        if (
            baseline_cache_matches_rewards(manifest)
            and events_ok
            and pos_ok
            and features_ok
        ):
            print(f"Reusing cached linear baselines: {paths['predictions']}", flush=True)
            return cached, manifest
        reasons: list[str] = []
        if not baseline_cache_matches_rewards(manifest) or not events_ok:
            reasons.append(
                f"events/reward defs (need {sorted(need_events) or REWARD_REPRESENTATION!r})"
            )
        if not pos_ok:
            reasons.append(f"binary_pos_weight_max={pos_max}")
        if not features_ok:
            reasons.append("ModelSpec feature restriction")
        print(
            "Cached linear baselines outdated ("
            + "; ".join(reasons)
            + "); refitting...",
            flush=True,
        )

    # Caches written before the all-binary reward bundle carry no usable manifest,
    # so the legacy baseline_cache/ migration path is intentionally not reused.
    if legacy.exists() and not refit_baselines:
        print(
            f"Ignoring legacy linear baselines at {legacy}: predates the "
            "binary-worsening rewards.",
            flush=True,
        )

    if train_data is None:
        raise ValueError(
            "Linear baselines are missing. Provide --train-data and run with "
            "--baselines-only (or omit --baseline-dir freeze and allow fitting)."
        )
    prepared = prepare_baselines(
        train_data=train_data,
        test_data=test_data,
        preprocessing_path=preprocessing_path,
        baseline_dir=baseline_dir,
        seed=seed,
        skip_linear_baseline=skip_linear_baseline,
        event_names=event_names,
        binary_pos_weight_max=pos_max,
        allowed_feature_names=allowed_feature_names,
    )
    return prepared["baseline"], prepared["manifest"]


def attach_baselines(
    state: pd.DataFrame,
    events: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["person_id", "wave", "next_wave"]

    def _attach_prob(
        frame: pd.DataFrame,
        name_col: str,
        observed_binary: Callable[[pd.DataFrame], pd.Series],
    ) -> pd.DataFrame:
        if frame.empty or name_col not in frame.columns:
            return frame
        chunks = []
        for name, group in frame.groupby(name_col, sort=False):
            column = f"baseline_prob__{name}"
            use = baseline[keys + ([column] if column in baseline else [])].copy()
            if column not in use:
                use[column] = float(observed_binary(group).mean())
            merged = group.merge(use, on=keys, how="left", validate="many_to_one")
            merged = merged.rename(columns={column: "baseline_prob"})
            chunks.append(merged)
        return pd.concat(chunks, ignore_index=True) if chunks else frame

    def _attach_level_baselines(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "outcome" not in frame.columns:
            return frame
        chunks = []
        for name, group in frame.groupby("outcome", sort=False):
            cols = keys.copy()
            mean_col = f"baseline_mean__{name}"
            prob_col = f"baseline_prob__{name}"
            keep = [c for c in (mean_col, prob_col) if c in baseline.columns]
            use = baseline[keys + keep].copy()
            if mean_col not in use:
                use[mean_col] = float(
                    pd.to_numeric(group["observed_raw"], errors="coerce").mean()
                )
            if prob_col not in use:
                if "observed_clinical" in group.columns:
                    use[prob_col] = float(group["observed_clinical"].mean())
                else:
                    use[prob_col] = float(
                        clinical_binary_from_level(group["observed_raw"], str(name)).mean()
                    )
            merged = group.merge(use, on=keys, how="left", validate="many_to_one")
            merged = merged.rename(
                columns={mean_col: "baseline_mean", prob_col: "baseline_prob"}
            )
            chunks.append(merged)
        return pd.concat(chunks, ignore_index=True) if chunks else frame

    events_out = _attach_prob(events, "event", lambda g: g["observed"])
    _ = _attach_level_baselines  # kept for callers that still pass empty level frames
    return state, events_out


def safe_auprc(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    active = np.isfinite(y) & np.isfinite(score)
    if active.sum() == 0 or len(np.unique(y[active])) < 2:
        return float("nan")
    return float(
        average_precision_score(
            y[active],
            score[active],
            sample_weight=None if weight is None else np.asarray(weight)[active],
        )
    )


def binary_rates_at_threshold(
    y: np.ndarray,
    prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Accuracy / sensitivity / specificity at a fixed probability threshold."""
    y = (np.asarray(y, dtype=float) > 0.5).astype(int)
    prob = np.asarray(prob, dtype=float)
    active = np.isfinite(y) & np.isfinite(prob)
    if active.sum() == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
        }
    pred = (prob[active] >= float(threshold)).astype(int)
    true = y[active]
    tp = float(((pred == 1) & (true == 1)).sum())
    tn = float(((pred == 0) & (true == 0)).sum())
    fp = float(((pred == 1) & (true == 0)).sum())
    fn = float(((pred == 0) & (true == 1)).sum())
    total = tp + tn + fp + fn
    pos = tp + fn
    neg = tn + fp
    sensitivity = tp / pos if pos > 0 else float("nan")
    specificity = tn / neg if neg > 0 else float("nan")
    accuracy = (tp + tn) / total if total > 0 else float("nan")
    balanced_accuracy = (
        0.5 * (sensitivity + specificity)
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else float("nan")
    )
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def sensitivity_at_min_specificity(
    y: np.ndarray,
    prob: np.ndarray,
    *,
    min_specificity: float = 0.8,
    grid_size: int = 199,
) -> dict[str, float]:
    """Max sensitivity among thresholds with specificity ≥ ``min_specificity``.

    Sweeps a probability grid (plus an all-negative extreme). Returns the
    operating point with the highest sensitivity that still meets the
    specificity floor; ties break toward the threshold closer to 0.5.
    """
    nan = {
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "threshold": float("nan"),
    }
    y_arr = (np.asarray(y, dtype=float) > 0.5).astype(int)
    p_arr = np.asarray(prob, dtype=float)
    active = np.isfinite(y_arr) & np.isfinite(p_arr)
    if int(active.sum()) < 2:
        return nan
    y_use = y_arr[active]
    p_use = p_arr[active]
    if len(np.unique(y_use)) < 2:
        return nan

    lo = 1.0 / (grid_size + 1)
    hi = 1.0 - lo
    grid = np.linspace(lo, hi, grid_size)
    # All-negative prediction (threshold above every score) always has spec=1.
    candidates = np.concatenate([grid, np.asarray([float(np.max(p_use)) + 1e-6])])

    best_sen = -1.0
    best_spec = float("nan")
    best_t = float("nan")
    best_dist = float("inf")
    for threshold in candidates:
        rates = binary_rates_at_threshold(y_use, p_use, threshold=float(threshold))
        sen = float(rates["sensitivity"])
        spec = float(rates["specificity"])
        if not (np.isfinite(sen) and np.isfinite(spec)):
            continue
        if spec + 1e-12 < float(min_specificity):
            continue
        dist = abs(float(threshold) - 0.5)
        if sen > best_sen + 1e-12 or (
            abs(sen - best_sen) <= 1e-12 and dist < best_dist
        ):
            best_sen = sen
            best_spec = spec
            best_t = float(threshold)
            best_dist = dist

    if best_sen < 0.0:
        return nan
    return {
        "sensitivity": float(best_sen),
        "specificity": float(best_spec),
        "threshold": float(best_t),
    }


def brier(y: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    active = np.isfinite(y) & np.isfinite(prob)
    return float(np.mean((prob[active] - y[active]) ** 2)) if active.any() else float("nan")


def cluster_bootstrap_ci(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame, np.ndarray], float],
    reps: int,
    seed: int,
) -> tuple[float, float]:
    if reps <= 1 or "person_id" not in frame.columns or frame["person_id"].nunique() < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    people = frame["person_id"].astype(str).unique()
    estimates = []
    person_series = frame["person_id"].astype(str)
    for _ in range(reps):
        sampled = rng.choice(people, size=len(people), replace=True)
        counts = pd.Series(sampled).value_counts()
        active = person_series.isin(counts.index)
        use = frame.loc[active]
        weights = use["person_id"].astype(str).map(counts).to_numpy(float)
        value = metric(use, weights)
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    return tuple(np.quantile(estimates, [0.025, 0.975]).astype(float))


def calibration_slope(frame: pd.DataFrame, prob_col: str) -> tuple[float, float, float]:
    use = frame[["observed", prob_col]].dropna()
    if use.empty or use["observed"].nunique() < 2:
        return float("nan"), float("nan"), float("nan")
    p = np.clip(use[prob_col].to_numpy(float), 1e-5, 1 - 1e-5)
    x = np.log(p / (1 - p))[:, None]
    y = use["observed"].to_numpy(int)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, y)
    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0, 0])
    q = model.predict_proba(x)[:, 1]
    design = np.column_stack([np.ones(len(x)), x[:, 0]])
    information = design.T @ (design * (q * (1 - q))[:, None])
    covariance = np.linalg.pinv(information)
    se = float(np.sqrt(max(covariance[1, 1], 0)))
    return intercept, slope, se


def compute_state_performance(
    state: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if state.empty or "outcome" not in state.columns:
        return pd.DataFrame()
    rows = []
    for name, group in state.groupby("outcome", sort=False):
        if len(group) < 200:
            continue
        model_mae = float(np.mean(np.abs(group["predicted_std"] - group["observed_std"])))
        persistence_mae = float(np.mean(np.abs(group["current_std"] - group["observed_std"])))
        low, high = cluster_bootstrap_ci(
            group,
            lambda x, w: float(np.average(np.abs(x["predicted_std"] - x["observed_std"]), weights=w)),
            bootstrap,
            seed,
        )
        rows.append(
            {
                "outcome": name,
                "label": group["label"].iloc[0],
                "model_mae": model_mae,
                "persistence_mae": persistence_mae,
                "improvement_vs_persistence_pct": 100.0
                * (persistence_mae - model_mae)
                / persistence_mae
                if persistence_mae
                else float("nan"),
                "ci_low": low,
                "ci_high": high,
                "n": len(group),
            }
        )
    return pd.DataFrame(rows)


def compute_level_performance(
    levels: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Metrics for next-wave continuous *level* rewards (not wave-to-wave deltas)."""
    if levels.empty or "outcome" not in levels.columns:
        empty = pd.DataFrame()
        return empty, empty, empty
    level_rows = []
    calibration_rows = []
    risk_rows = []
    for name, group in levels.groupby("outcome", sort=False):
        use = group.copy()
        if "observed_clinical" not in use.columns:
            use["observed_clinical"] = clinical_binary_from_level(
                use["observed_raw"], str(name)
            )
        if "risk_score" not in use.columns:
            use["risk_score"] = level_risk_score(use["predicted_raw"], str(name))
        obs = pd.to_numeric(use["observed_raw"], errors="coerce").to_numpy(float)
        pred = pd.to_numeric(use["predicted_raw"], errors="coerce").to_numpy(float)
        finite = np.isfinite(obs) & np.isfinite(pred)
        if finite.sum() < 50:
            continue
        model_mae = float(np.mean(np.abs(pred[finite] - obs[finite])))
        model_rmse = float(np.sqrt(np.mean((pred[finite] - obs[finite]) ** 2)))
        if "baseline_mean" in use.columns:
            base = pd.to_numeric(use["baseline_mean"], errors="coerce").to_numpy(float)
            baseline_mae = float(np.mean(np.abs(base[finite] - obs[finite])))
        else:
            baseline_mae = float(np.mean(np.abs(obs[finite] - obs[finite].mean())))
        if np.std(obs[finite]) > 1e-8 and np.std(pred[finite]) > 1e-8:
            corr = float(np.corrcoef(obs[finite], pred[finite])[0, 1])
        else:
            corr = float("nan")
        y_bin = use["observed_clinical"].to_numpy(int)
        model_auprc = safe_auprc(y_bin, use["risk_score"].to_numpy(float))
        baseline_auprc = (
            safe_auprc(y_bin, use["baseline_prob"].to_numpy(float))
            if "baseline_prob" in use.columns
            else float("nan")
        )
        low, high = cluster_bootstrap_ci(
            use,
            lambda x, w: float(
                np.average(
                    np.abs(
                        pd.to_numeric(x["predicted_raw"], errors="coerce")
                        - pd.to_numeric(x["observed_raw"], errors="coerce")
                    ),
                    weights=w,
                )
            ),
            bootstrap,
            seed + 11,
        )
        level_rows.append(
            {
                "outcome": name,
                "label": use["label"].iloc[0],
                "model_mae": model_mae,
                "baseline_mae": baseline_mae,
                "improvement_vs_mean_pct": 100.0
                * (baseline_mae - model_mae)
                / baseline_mae
                if baseline_mae
                else float("nan"),
                "model_rmse": model_rmse,
                "pearson_r": corr,
                "clinical_prevalence": float(np.mean(y_bin)) if len(y_bin) else float("nan"),
                "model_clinical_auprc": model_auprc,
                "baseline_clinical_auprc": baseline_auprc,
                "mae_ci_low": low,
                "mae_ci_high": high,
                "n": int(finite.sum()),
                "clinical_events": int(np.sum(y_bin)),
            }
        )
        rank = use["predicted_raw"].rank(method="first")
        use["decile"] = pd.qcut(rank, q=10, labels=False, duplicates="drop") + 1
        cal = use.groupby("decile", as_index=False).agg(
            predicted_level=("predicted_raw", "mean"),
            observed_level=("observed_raw", "mean"),
            observed_clinical=("observed_clinical", "mean"),
            n=("observed_clinical", "size"),
        )
        cal["outcome"] = name
        cal["label"] = use["label"].iloc[0]
        calibration_rows.append(cal)
        use["quintile"] = pd.qcut(rank, q=5, labels=False, duplicates="drop") + 1
        risk = use.groupby("quintile", as_index=False).agg(
            observed_clinical=("observed_clinical", "mean"),
            n=("observed_clinical", "size"),
        )
        risk["outcome"] = name
        risk["label"] = use["label"].iloc[0]
        risk_rows.append(risk)
    return (
        pd.DataFrame(level_rows),
        pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame(),
        pd.concat(risk_rows, ignore_index=True) if risk_rows else pd.DataFrame(),
    )


# Deprecated alias.
compute_worsening_performance = compute_level_performance


def compute_event_performance(
    events: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if events.empty or "event" not in events.columns:
        return pd.DataFrame()
    rows = []
    for name, group in events.groupby("event", sort=False):
        y = group["observed"].to_numpy(int)
        p = group["predicted_prob"].to_numpy(float)
        b = group["baseline_prob"].to_numpy(float)
        low, high = cluster_bootstrap_ci(
            group,
            lambda x, w: safe_auprc(x["observed"], x["predicted_prob"], w),
            bootstrap,
            seed + 23,
        )
        intercept, slope, slope_se = calibration_slope(group, "predicted_prob")
        rows.append(
            {
                "event": name,
                "label": group["label"].iloc[0],
                "prevalence": float(y.mean()),
                "events": int(y.sum()),
                "n": len(y),
                "model_auprc": safe_auprc(y, p),
                "baseline_auprc": safe_auprc(y, b),
                "model_brier": brier(y, p),
                "baseline_brier": brier(y, b),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "calibration_slope_se": slope_se,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def compute_rollout_performance(
    rollout_state: pd.DataFrame,
    rollout_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rollout_state.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    error = (
        rollout_state.assign(
            model_error=lambda x: np.abs(x["predicted_std"] - x["observed_std"]),
            persistence_error=lambda x: np.abs(x["baseline_std"] - x["observed_std"]),
        )
        .groupby("horizon", as_index=False)
        .agg(
            elapsed_years=("elapsed_years", "mean"),
            model_mae=("model_error", "mean"),
            persistence_mae=("persistence_error", "mean"),
            model_se=("model_error", lambda x: x.std(ddof=1) / math.sqrt(max(len(x), 1))),
            n=("model_error", "size"),
        )
    )
    trajectory_rows = []
    for outcome in ["adl_total_score", "total_cognition_score"]:
        use = rollout_state[rollout_state["outcome"] == outcome].copy()
        if use.empty:
            continue
        if outcome == "adl_total_score":
            use["baseline_stratum"] = np.where(
                use["baseline_raw"] <= 0, "No baseline ADL limitation", "Baseline ADL limitation"
            )
        else:
            median = use.loc[use["horizon"] == use["horizon"].min(), "baseline_raw"].median()
            use["baseline_stratum"] = np.where(
                use["baseline_raw"] >= median, "Higher baseline cognition", "Lower baseline cognition"
            )
        agg = use.groupby(["outcome", "label", "baseline_stratum", "horizon"], as_index=False).agg(
            elapsed_years=("elapsed_years", "mean"),
            observed=("observed_raw", "mean"),
            predicted=("predicted_raw", "mean"),
            n=("person_id", "nunique"),
        )
        trajectory_rows.append(agg)

    cumulative_rows = []
    if not rollout_events.empty:
        for event in ["death_event"]:
            use = rollout_events[rollout_events["event"] == event].sort_values(["person_id", "horizon"]).copy()
            if use.empty:
                continue
            use["cum_predicted"] = use.groupby("person_id")["predicted_prob"].transform(
                lambda x: 1 - (1 - x.clip(0, 1)).cumprod()
            )
            use["cum_observed"] = use.groupby("person_id")["observed"].cummax()
            agg = use.groupby(["event", "label", "horizon"], as_index=False).agg(
                elapsed_years=("elapsed_years", "mean"),
                predicted=("cum_predicted", "mean"),
                observed=("cum_observed", "mean"),
                n=("person_id", "nunique"),
            )
            cumulative_rows.append(agg)
    return (
        error,
        pd.concat(trajectory_rows, ignore_index=True) if trajectory_rows else pd.DataFrame(),
        pd.concat(cumulative_rows, ignore_index=True) if cumulative_rows else pd.DataFrame(),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [_jsonable(row) for row in frame.to_dict(orient="records")]


def write_report(
    output_dir: Path,
    checkpoint: Path,
    aggregate_metrics: Mapping[str, float],
    state_performance: pd.DataFrame,
    level_metrics: pd.DataFrame,
    event_metrics: pd.DataFrame,
    rollout_error: pd.DataFrame,
    compatibility_warnings: list[str],
) -> Path:
    lines = [
        "# Health world model evaluation report",
        "",
        f"- Checkpoint: `{checkpoint}`",
        "- Primary result: one-step JEPA **predict_next** (context encoder + predictor; the next observation is not used to reconstruct the state).",
        "- Confidence intervals: person-level cluster bootstrap (default).",
        "",
        "## Aggregate world-model metrics (Trainer.evaluate_world)",
        "",
    ]
    for key in (
        "posterior_world_loss",
        "posterior_jepa_loss",
        "posterior_reward_loss",
        "rare_event_mean_balanced_accuracy",
    ):
        if key in aggregate_metrics:
            lines.append(f"- `{key}`: {aggregate_metrics[key]:.6f}")

    lines.extend(["", "## Continuous clinical states (prior vs persistence)", ""])
    if state_performance.empty:
        lines.append("- This model has no StateDecoder and does not reconstruct the full state.")
    else:
        improved = int((state_performance["model_mae"] < state_performance["persistence_mae"]).sum())
        lines.append(
            f"- {improved}/{len(state_performance)} states have a lower standardized MAE than persistence."
        )
        for _, row in state_performance.sort_values("improvement_vs_persistence_pct").iterrows():
            lines.append(
                f"- {row['label']}: model MAE={row['model_mae']:.3f}, "
                f"persistence={row['persistence_mae']:.3f} "
                f"({row['improvement_vs_persistence_pct']:+.1f}%)"
            )

    lines.extend(["", "## Major clinical events (AUPRC / Brier)", ""])
    if event_metrics.empty:
        lines.append("- No usable event predictions (the checkpoint may not match fields in this data bundle).")
    else:
        for _, row in event_metrics.sort_values("model_auprc", ascending=False).iterrows():
            lines.append(
                f"- {row['label']}: AUPRC={row['model_auprc']:.3f} "
                f"(baseline {row['baseline_auprc']:.3f}), "
                f"Brier={row['model_brier']:.3f}, n={int(row['n'])}, events={int(row['events'])}"
            )

    lines.extend(
        [
            "",
            "## Next-wave continuous levels (reward levels, not between-wave differences)",
            "",
            "- Primary metrics: raw-scale MAE and Pearson r, relative to the training-set mean baseline.",
            "- Also reported: binary AUPRC at the clinical cut-point of each continuous level, when a continuous reward is present.",
            "",
        ]
    )
    if level_metrics.empty:
        lines.append("- No usable continuous-level predictions.")
    else:
        for _, row in level_metrics.sort_values("model_mae").iterrows():
            lines.append(
                f"- {row['label']}: MAE={row['model_mae']:.3f} "
                f"(mean-baseline {row['baseline_mae']:.3f}, "
                f"{row['improvement_vs_mean_pct']:+.1f}%), "
                f"r={row['pearson_r']:.3f}, "
                f"clinical AUPRC={row['model_clinical_auprc']:.3f} "
                f"(baseline {row['baseline_clinical_auprc']:.3f}), "
                f"clinical prev={row['clinical_prevalence']:.3f}"
            )

    lines.extend(["", "## Free-running rollout", ""])
    if rollout_error.empty:
        lines.append("- No rollout results were produced.")
    else:
        longest = rollout_error.sort_values("horizon").iloc[-1]
        lines.append(
            f"- Longest horizon={int(longest['horizon'])} "
            f"(~{longest['elapsed_years']:.1f}y): "
            f"model MAE={longest['model_mae']:.3f}, "
            f"persistence={longest['persistence_mae']:.3f}"
        )
        lines.append("- Later steps are conditioned on the observed action path and are not an unconditional natural-history forecast.")

    lines.extend(
        [
            "",
            "## Checkpoint / data-bundle compatibility",
            "",
            *(
                [f"- {message}" for message in compatibility_warnings]
                if compatibility_warnings
                else ["- Every reward field required by the checkpoint is present in the test table."]
            ),
            "",
            "## Limitations",
            "",
            "1. Linear baselines use the current state, mask, and action; they are not formal survival models.",
            "2. Rollouts use observed future actions and do not support a causal-intervention interpretation.",
            "3. Continuous rewards are next-wave state levels; binary clinical cut-points are auxiliary ranking metrics.",
            "4. Thresholded event-classification metrics are in `Trainer.evaluate_world`. This report focuses on ranking and calibration.",
        ]
    )
    path = output_dir / "EVALUATION_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot_evaluation_summary(
    figures_dir: Path,
    state_performance: pd.DataFrame,
    event_metrics: pd.DataFrame,
    rollout_error: pd.DataFrame,
    level_metrics: pd.DataFrame | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if level_metrics is None:
        level_metrics = pd.DataFrame()

    if not event_metrics.empty:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        order = event_metrics.sort_values("model_auprc")
        y = np.arange(len(order))
        ax.barh(y - 0.15, order["model_auprc"], height=0.3, label="JEPA prior", color="#0072B2")
        ax.barh(y + 0.15, order["baseline_auprc"], height=0.3, label="Linear baseline", color="#6E6E6E")
        ax.set_yticks(y, order["label"])
        ax.set_xlabel("AUPRC")
        ax.set_title("Binary event ranking performance")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        path = figures_dir / "events_auprc.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    if not state_performance.empty:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        order = state_performance.sort_values("model_mae", ascending=False)
        y = np.arange(len(order))
        ax.barh(y - 0.15, order["model_mae"], height=0.3, label="JEPA prior", color="#0072B2")
        ax.barh(y + 0.15, order["persistence_mae"], height=0.3, label="Persistence", color="#D55E00")
        ax.set_yticks(y, order["label"])
        ax.set_xlabel("Standardized MAE")
        ax.set_title("Continuous state one-step error")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        path = figures_dir / "state_mae.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    if not level_metrics.empty and {"model_mae", "baseline_mae"}.issubset(level_metrics.columns):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        order = level_metrics.sort_values("model_mae", ascending=False)
        y = np.arange(len(order))
        ax.barh(y - 0.15, order["model_mae"], height=0.3, label="JEPA prior", color="#0072B2")
        ax.barh(
            y + 0.15,
            order["baseline_mae"],
            height=0.3,
            label="Train-mean level",
            color="#6E6E6E",
        )
        ax.set_yticks(y, order["label"])
        ax.set_xlabel("Raw-scale MAE")
        ax.set_title("Next-wave continuous level error")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        path = figures_dir / "level_mae.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    if not rollout_error.empty:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(rollout_error["horizon"], rollout_error["model_mae"], marker="o", label="JEPA")
        ax.plot(
            rollout_error["horizon"],
            rollout_error["persistence_mae"],
            marker="s",
            label="Persistence",
        )
        ax.set_xlabel("Horizon (steps)")
        ax.set_ylabel("Standardized MAE")
        ax.set_title("Free-running rollout error")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        path = figures_dir / "rollout_mae.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written


def run_evaluation(
    *,
    checkpoint_path: Path,
    test_data: Path,
    train_data: Path | None,
    preprocessing_path: Path | None,
    output_dir: Path,
    device_name: str = "auto",
    batch_size: int | None = None,
    num_workers: int | None = None,
    bootstrap: int = 100,
    rollout_horizon: int = 5,
    seed: int = 2026,
    reuse_predictions: bool = False,
    skip_linear_baseline: bool = False,
    refit_baselines: bool = False,
    baseline_dir: Path | None = None,
    skip_rollout: bool = False,
    skip_aggregate: bool = True,
    plot: bool = True,
    figures_only: bool = True,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir).resolve()
    cache_dir = output_dir / "prediction_cache"
    source_dir = output_dir / "source_data"
    figures_dir = output_dir / "figures"
    baseline_dir = Path(baseline_dir).resolve() if baseline_dir else (output_dir / "baseline_cache")
    for path in (output_dir, cache_dir, source_dir, figures_dir, baseline_dir):
        path.mkdir(parents=True, exist_ok=True)

    agent, spec, cfg = load_agent(checkpoint_path, device)
    if preprocessing_path is None:
        raise ValueError("--preprocessing is required for sample-level evaluation")
    preprocessing = json.loads(Path(preprocessing_path).read_text(encoding="utf-8"))
    compatibility_warnings = schema_compatibility_warnings(test_data, spec)
    for message in compatibility_warnings:
        print(f"WARNING: {message}", flush=True)

    pos_max = float(
        (cfg.get("world_loss") or {}).get(
            "binary_reward_pos_weight_max",
            DEFAULT_BINARY_POS_WEIGHT_MAX,
        )
    )
    baseline, baseline_manifest = load_or_prepare_baselines(
        train_data=train_data,
        test_data=test_data,
        preprocessing_path=Path(preprocessing_path),
        baseline_dir=baseline_dir,
        output_dir=output_dir,
        seed=seed,
        skip_linear_baseline=skip_linear_baseline,
        refit_baselines=refit_baselines,
        event_names=list(spec.reward_binary),
        binary_pos_weight_max=pos_max,
        allowed_feature_names=(
            list(spec.state_names) + list(spec.static_names) + list(spec.action_names)
        ),
    )

    bs = int(batch_size if batch_size is not None else cfg["data"]["batch_size"])
    workers = int(num_workers if num_workers is not None else 0)
    loader_kwargs = dict(
        table_path=test_data,
        spec=spec,
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        seed=seed,
    )

    aggregate_metrics: dict[str, float] = {}
    if not skip_aggregate and not figures_only:
        print("Computing aggregate Trainer.evaluate_world metrics...", flush=True)
        trainer = Trainer(agent, spec, cfg, device, output_dir)
        loader = build_loader(**loader_kwargs)
        aggregate_metrics = {
            str(k): float(v) for k, v in trainer.evaluate_world(loader).items()
        }
        (output_dir / "aggregate_world_metrics.json").write_text(
            json.dumps(aggregate_metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    cache_paths = {
        "state": cache_dir / "one_step_state_predictions.parquet",
        "events": cache_dir / "one_step_event_predictions.parquet",
        "levels": cache_dir / "one_step_level_predictions.parquet",
        "rollout_state": cache_dir / "rollout_state_predictions.parquet",
        "rollout_events": cache_dir / "rollout_event_predictions.parquet",
    }
    legacy_levels = cache_dir / "one_step_worsening_predictions.parquet"
    required = list(cache_paths.values())
    if skip_rollout:
        required = [p for k, p in cache_paths.items() if not k.startswith("rollout_")]
    # Old caches used delta>0 semantics; require level schema before reuse.
    reuse = (
        reuse_predictions
        and all(p.exists() for p in required)
        and "observed_clinical" in parquet_columns(cache_paths["levels"])
    )

    if reuse:
        print("Reusing cached model prediction tables...", flush=True)
        state = pd.read_parquet(cache_paths["state"])
        events = pd.read_parquet(cache_paths["events"])
        levels = pd.read_parquet(cache_paths["levels"])
        if skip_rollout:
            rollout_state = pd.DataFrame()
            rollout_events = pd.DataFrame()
        else:
            rollout_state = pd.read_parquet(cache_paths["rollout_state"])
            rollout_events = pd.read_parquet(cache_paths["rollout_events"])
    else:
        print(f"Device: {device}", flush=True)
        print(f"Checkpoint: {checkpoint_path}", flush=True)
        loader = build_loader(**loader_kwargs)
        state, events = collect_one_step_predictions(
            agent, loader, preprocessing, device
        )
        levels = pd.DataFrame()
        if skip_rollout:
            rollout_state = pd.DataFrame()
            rollout_events = pd.DataFrame()
        else:
            loader = build_loader(**loader_kwargs)
            rollout_state, rollout_events = collect_free_rollouts(
                agent, loader, preprocessing, device, max_horizon=rollout_horizon
            )
        state.to_parquet(cache_paths["state"], index=False)
        events.to_parquet(cache_paths["events"], index=False)
        levels.to_parquet(cache_paths["levels"], index=False)
        if legacy_levels.exists():
            try:
                legacy_levels.unlink()
            except OSError:
                pass
        if not skip_rollout:
            rollout_state.to_parquet(cache_paths["rollout_state"], index=False)
            rollout_events.to_parquet(cache_paths["rollout_events"], index=False)

    state, events = attach_baselines(state, events, baseline)
    levels = pd.DataFrame()
    state_perf = compute_state_performance(state, bootstrap, seed)
    event_metrics = compute_event_performance(events, bootstrap, seed)
    rollout_error, rollout_trajectories, rollout_cumulative = compute_rollout_performance(
        rollout_state, rollout_events
    )

    level_metrics, level_calibration, level_risk = compute_level_performance(
        levels, bootstrap, seed
    )
    source_tables = {
        "state_performance.csv": state_perf,
        "level_performance.csv": level_metrics,
        "level_calibration.csv": level_calibration,
        "level_clinical_risk_quintiles.csv": level_risk,
        "event_performance.csv": event_metrics,
        "rollout_error.csv": rollout_error,
        "rollout_trajectories.csv": rollout_trajectories,
        "rollout_cumulative_events.csv": rollout_cumulative,
    }
    csv_paths: list[Path] = []
    for filename, frame in source_tables.items():
        path = source_dir / filename
        frame.to_csv(path, index=False)
        csv_paths.append(path)
        print(f"Wrote {path}", flush=True)

    report_path: Path | None = None
    if not figures_only:
        report_path = write_report(
            output_dir,
            checkpoint_path,
            aggregate_metrics,
            state_perf,
            level_metrics,
            event_metrics,
            rollout_error,
            compatibility_warnings,
        )

    figure_paths: list[Path] = []
    if figures_only or plot:
        figure_paths = plot_evaluation_summary(
            figures_dir, state_perf, event_metrics, rollout_error, level_metrics
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "test_data": str(test_data),
        "train_data": str(train_data) if train_data else None,
        "preprocessing": str(preprocessing_path),
        "output_dir": str(output_dir),
        "baseline_dir": str(baseline_dir),
        "baseline_manifest": _jsonable(baseline_manifest),
        "device": str(device),
        "bootstrap": bootstrap,
        "rollout_horizon": None if skip_rollout else rollout_horizon,
        "compatibility_warnings": compatibility_warnings,
        "figures_only": figures_only,
        "aggregate_world_metrics": _jsonable(aggregate_metrics),
        "state_performance": dataframe_records(state_perf),
        "event_performance": dataframe_records(event_metrics),
        "level_performance": dataframe_records(level_metrics),
        "worsening_performance": dataframe_records(level_metrics),  # deprecated alias
        "rollout_error": dataframe_records(rollout_error),
        "report": str(report_path) if report_path else None,
        "figures": [str(p) for p in figure_paths],
        "source_data_dir": str(source_dir),
        "source_csv": [str(p) for p in csv_paths],
        "prediction_cache_dir": str(cache_dir),
        "continuous_reward_representation": "next_state_level",
    }
    if not figures_only:
        summary_path = output_dir / "evaluation_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote {summary_path}", flush=True)
        if report_path is not None:
            print(f"Wrote {report_path}", flush=True)
    return summary
