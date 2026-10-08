"""Build the model variable specification from the training table and upstream config.

The module separates continuous and categorical states, continuous and categorical
actions, and binary and continuous rewards. It also estimates category sets,
continuous-action support, and the standardization parameters for the time interval.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


# Explicit event-reward prefixes and suffixes. Other rewards are typed from the training values.
# *_worsening remains a next-wave severe 0/1 flag.
BINARY_REWARD_PREFIXES = ("death_event",)
BINARY_REWARD_SUFFIXES = (
    "_event",
    "_worsening",
)
BINARY_REWARD_NAMES: frozenset[str] = frozenset()
# Derived Bernoulli rewards built at load time from parquet columns. Empty in this release.
DEFAULT_POOL_BINARY_REWARDS: dict[str, tuple[str, ...]] = {}


def read_model_table(path: Path) -> pd.DataFrame:
    """Load model-ready train/validation/test tables (parquet preferred, CSV accepted)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            low_memory=False,
            dtype={"person_id": "string"},
        )
    else:
        raise ValueError(f"Unsupported model table format: {path}")
    if "person_id" in frame.columns:
        frame["person_id"] = frame["person_id"].astype("string")
    return frame


@dataclass
class CategoricalFeature:
    """Categorical variable name and the valid category values seen in training."""
    name: str
    values: list[float]
    # When True, StateDecoder uses cumulative (CORAL) ordinal loss instead of CE.
    ordered: bool = False

    @property
    def classes(self) -> int:
        """Number of valid categories. Index 0 is reserved for the missing category."""
        return len(self.values)


@dataclass
class ContinuousActionFeature:
    """Continuous action name and its support on the training data."""
    name: str
    low: float
    high: float


# ======================== Final model variable specification ========================

# Legacy reward -> source state map, kept for older bundles whose rewards were
# next-wave levels. Current bundles derive these as binary worsening flags, so
# they land in reward_binary and this map is not consulted.
CONTINUOUS_REWARD_TO_STATE: dict[str, str] = {
    "adl_worsening": "adl_total_score",
    "iadl_worsening": "iadl_total_score",
}
CONTINUOUS_REWARD_TO_STATE_FALLBACKS: dict[str, tuple[str, ...]] = {}


def mapped_state_for_continuous_reward(
    reward_name: str,
    available_states: Sequence[str] | None = None,
) -> str | None:
    """State column that a continuous reward residual head should track."""
    candidates: list[str] = []
    primary = CONTINUOUS_REWARD_TO_STATE.get(reward_name)
    if primary:
        candidates.append(primary)
    candidates.extend(CONTINUOUS_REWARD_TO_STATE_FALLBACKS.get(reward_name, ()))
    if available_states is None:
        return candidates[0] if candidates else None
    avail = set(available_states)
    for name in candidates:
        if name in avail:
            return name
    return None


def continuous_reward_standardize_info(
    reward_name: str,
    preprocessing: Mapping[str, Any] | None,
    available_states: Sequence[str] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Return ``(state_name, mean/std info)`` for decoding a continuous reward."""
    payload = preprocessing or {}
    state = mapped_state_for_continuous_reward(reward_name, available_states)
    info = dict((payload.get("reward_continuous") or {}).get(reward_name) or {})
    if not info and state:
        info = dict(
            (payload.get("continuous") or {}).get(state)
            or (payload.get("state_continuous") or {}).get(state)
            or {}
        )
    return state, info


@dataclass
class ModelSpec:
    """States, actions, rewards, and time-interval parameters used by the model."""
    state_continuous: list[str] = field(default_factory=list)
    state_categorical: list[CategoricalFeature] = field(default_factory=list)
    static_continuous: list[str] = field(default_factory=list)
    static_categorical: list[CategoricalFeature] = field(default_factory=list)
    action_continuous: list[ContinuousActionFeature] = field(default_factory=list)
    action_categorical: list[CategoricalFeature] = field(default_factory=list)
    reward_binary: list[str] = field(default_factory=list)
    reward_continuous: list[str] = field(default_factory=list)
    # Per binary reward: BCE positive-class weight ≈ n_neg / n_pos (clamped).
    reward_binary_pos_weight: list[float] = field(default_factory=list)
    # Binary rewards kept in ClinicalHeads / eval, but stop-grad into z
    # (representation is not trained on these labels; heads act as probes).
    probe_only_rewards: list[str] = field(default_factory=list)
    # name -> parquet source reward names; OR among rows with all sources observed.
    pooled_binary_rewards: dict[str, list[str]] = field(default_factory=dict)
    delta_t_mean: float = 0.0
    delta_t_std: float = 1.0
    source_config: str = ""
    source_preprocessing: str = ""

    @property
    def state_names(self) -> list[str]:
        """Dynamic state names in model input order."""
        return self.state_continuous + [x.name for x in self.state_categorical]

    @property
    def static_names(self) -> list[str]:
        """Static context names in model input order."""
        return self.static_continuous + [x.name for x in self.static_categorical]

    @property
    def has_static_context(self) -> bool:
        """Whether at least one static context variable is configured."""
        return bool(self.static_names)

    @property
    def action_names(self) -> list[str]:
        """Action names in model input order."""
        return [x.name for x in self.action_continuous] + [
            x.name for x in self.action_categorical
        ]

    @property
    def reward_names(self) -> list[str]:
        """Reward names in model output order."""
        return self.reward_binary + self.reward_continuous

    def to_dict(self) -> dict[str, Any]:
        """Convert the specification to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelSpec":
        """Restore a model specification from a checkpoint or JSON dictionary."""
        return cls(
            state_continuous=list(payload.get("state_continuous", [])),
            state_categorical=[
                CategoricalFeature(
                    name=str(x["name"]),
                    values=[float(v) for v in x["values"]],
                    ordered=bool(x.get("ordered", False)),
                )
                for x in payload.get("state_categorical", [])
            ],
            static_continuous=list(payload.get("static_continuous", [])),
            static_categorical=[
                CategoricalFeature(
                    name=str(x["name"]),
                    values=[float(v) for v in x["values"]],
                    ordered=bool(x.get("ordered", False)),
                )
                for x in payload.get("static_categorical", [])
            ],
            action_continuous=[
                ContinuousActionFeature(**x)
                for x in payload.get("action_continuous", [])
            ],
            action_categorical=[
                CategoricalFeature(
                    name=str(x["name"]),
                    values=[float(v) for v in x["values"]],
                    ordered=bool(x.get("ordered", False)),
                )
                for x in payload.get("action_categorical", [])
            ],
            reward_binary=list(payload.get("reward_binary", [])),
            reward_continuous=list(payload.get("reward_continuous", [])),
            reward_binary_pos_weight=[
                float(x) for x in payload.get("reward_binary_pos_weight", [])
            ],
            probe_only_rewards=[
                str(x) for x in payload.get("probe_only_rewards", [])
            ],
            pooled_binary_rewards={
                str(name): [str(x) for x in sources]
                for name, sources in (
                    payload.get("pooled_binary_rewards") or {}
                ).items()
            },
            delta_t_mean=float(payload.get("delta_t_mean", 0.0)),
            delta_t_std=float(payload.get("delta_t_std", 1.0)),
            source_config=str(payload.get("source_config", "")),
            source_preprocessing=str(payload.get("source_preprocessing", "")),
        )

    def save(self, path: Path) -> None:
        """Write the specification to JSON so training and evaluation can be reproduced."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ModelSpec":
        """Load a model specification from a JSON file."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _read_json(path: Path) -> dict[str, Any]:
    """Read JSON and raise a clear error when the file is missing."""
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_values(
    frame: pd.DataFrame,
    value_col: str,
    mask_col: str,
) -> pd.Series:
    """Keep training values with mask=1 and drop missing placeholders."""
    if value_col not in frame.columns:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    if mask_col in frame.columns:
        mask = pd.to_numeric(frame[mask_col], errors="coerce").eq(1)
        values = values.loc[mask]
    return values.dropna()


def _categories(
    frame: pd.DataFrame,
    value_col: str,
    mask_col: str,
) -> list[float]:
    """Collect every valid category of a categorical variable from the training set."""
    values = _valid_values(frame, value_col, mask_col)
    unique = sorted(float(x) for x in values.unique())
    if not unique:
        raise ValueError(f"Categorical variable has no valid training categories: {value_col}")
    return unique


def parse_pool_binary_rewards(raw: Any) -> dict[str, list[str]]:
    """YAML ``pool_binary_rewards``: mapping name → list of source reward names."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("data.pool_binary_rewards must be a mapping of name → sources")
    out: dict[str, list[str]] = {}
    for name, sources in raw.items():
        if isinstance(sources, (list, tuple)):
            src = [str(x) for x in sources if str(x)]
        else:
            src = [str(sources)] if str(sources) else []
        if len(src) < 2:
            raise ValueError(
                f"pool_binary_rewards[{name!r}] needs ≥2 sources, got {src}"
            )
        out[str(name)] = src
    return out


def pooled_binary_arrays(
    frame: pd.DataFrame,
    sources: Sequence[str],
    *,
    how: str = "any",
    mask_how: str = "all",
) -> tuple[np.ndarray, np.ndarray]:
    """Pool source Bernoulli columns into ``(values, mask)`` float32 arrays.

    ``mask_how='all'``: only rows where every source is observed (CVD-free).
    ``how='any'``: event = 1 if any observed source is 1.
    """
    n = len(frame)
    if n == 0 or not sources:
        empty = np.zeros((n,), dtype=np.float32)
        return empty, empty.copy()
    values = np.stack(
        [
            pd.to_numeric(frame.get(f"reward__{src}"), errors="coerce")
            .to_numpy(dtype=float)
            if f"reward__{src}" in frame.columns
            else np.full(n, np.nan)
            for src in sources
        ],
        axis=1,
    )
    masks = np.stack(
        [
            pd.to_numeric(frame.get(f"reward_mask__{src}"), errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
            if f"reward_mask__{src}" in frame.columns
            else np.zeros(n, dtype=float)
            for src in sources
        ],
        axis=1,
    )
    observed = np.isfinite(values) & (masks > 0.5)
    values_01 = np.where(np.isfinite(values), values, 0.0) > 0.5
    if mask_how == "any":
        mask = observed.any(axis=1)
    else:
        mask = observed.all(axis=1)
    if how == "all":
        event = (values_01 | ~observed).all(axis=1)
    else:
        event = (values_01 & observed).any(axis=1)
    event = event & mask
    return event.astype(np.float32), mask.astype(np.float32)


def materialize_pooled_binary_rewards(
    frame: pd.DataFrame,
    pooled: Mapping[str, Sequence[str]] | None,
) -> pd.DataFrame:
    """Write ``reward__*`` / ``reward_mask__*`` for derived pooled events."""
    if not pooled:
        return frame
    out = frame
    copied = False
    for name, sources in pooled.items():
        value_col = f"reward__{name}"
        mask_col = f"reward_mask__{name}"
        if value_col in out.columns and mask_col in out.columns:
            continue
        values, masks = pooled_binary_arrays(out, list(sources))
        if not copied:
            out = out.copy()
            copied = True
        out[value_col] = values
        out[mask_col] = masks
    return out


def resolve_pooled_binary_rewards(
    event_names: Sequence[str] | None = None,
    explicit: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """Merge yaml pooling with defaults when a derived name is requested."""
    wanted = {str(x) for x in (event_names or [])}
    out: dict[str, list[str]] = {}
    for name, sources in DEFAULT_POOL_BINARY_REWARDS.items():
        if not wanted or name in wanted:
            out[name] = list(sources)
    if explicit:
        for name, sources in explicit.items():
            out[str(name)] = [str(x) for x in sources]
    if wanted:
        out = {k: v for k, v in out.items() if k in wanted}
    return out


def _binary_pos_weights(
    frame: pd.DataFrame,
    reward_binary: list[str],
    *,
    max_weight: float = 50.0,
) -> list[float]:
    """Estimate BCE pos_weight for rare binary rewards from masked train labels.

    Uses ``pos_weight = n_neg / n_pos`` among rows with ``reward_mask==1``,
    clamped to ``[1, max_weight]``. Set ``max_weight <= 0`` to disable (all 1.0).
    """
    if max_weight <= 0:
        return [1.0 for _ in reward_binary]
    weights: list[float] = []
    for name in reward_binary:
        values = _valid_values(
            frame,
            f"reward__{name}",
            f"reward_mask__{name}",
        )
        if values.empty:
            weights.append(1.0)
            continue
        positives = float((values > 0.5).sum())
        negatives = float((values <= 0.5).sum())
        if positives <= 0:
            weights.append(float(max_weight))
            continue
        weight = negatives / positives
        weights.append(float(np.clip(weight, 1.0, max_weight)))
    return weights


def _action_bounds(
    frame: pd.DataFrame,
    value_col: str,
    mask_col: str,
) -> tuple[float, float]:
    """Estimate robust continuous-action support from the training 1st and 99th percentiles."""
    values = _valid_values(frame, value_col, mask_col)
    if values.empty:
        return -1.0, 1.0
    low = float(values.quantile(0.01))
    high = float(values.quantile(0.99))
    if not np.isfinite(low):
        low = float(values.min())
    if not np.isfinite(high):
        high = float(values.max())
    if high - low < 1e-4:
        center = float(values.median())
        return center - 1.0, center + 1.0
    margin = 0.05 * (high - low)
    return low - margin, high + margin


# ======================== Infer the specification from training data ========================

def build_model_spec(
    train_table: Path,
    model_config_json: Path,
    preprocessing_json: Path,
    *,
    binary_pos_weight_max: float = 50.0,
    exclude_actions: Sequence[str] | None = None,
    exclude_states: Sequence[str] | None = None,
    exclude_rewards: Sequence[str] | None = None,
    force_continuous_rewards: Sequence[str] | None = None,
    probe_only_rewards: Sequence[str] | None = None,
    pool_binary_rewards: Mapping[str, Sequence[str]] | None = None,
) -> ModelSpec:
    config = _read_json(model_config_json)
    preprocessing = _read_json(preprocessing_json)
    frame = read_model_table(train_table)
    excluded_actions = {str(x) for x in (exclude_actions or [])}
    excluded_states = {str(x) for x in (exclude_states or [])}
    excluded_rewards = {str(x) for x in (exclude_rewards or [])}
    forced_continuous = {str(x) for x in (force_continuous_rewards or [])}
    pooled = parse_pool_binary_rewards(pool_binary_rewards)
    pooled_sources = {src for sources in pooled.values() for src in sources}
    excluded_rewards |= pooled_sources

    state_names = [
        str(x)
        for x in config.get("state_columns_model", [])
        if f"state__{x}" in frame.columns and str(x) not in excluded_states
    ]
    static_names = [
        str(x)
        for x in config.get("static_context_columns", [])
        if f"state__{x}" in frame.columns
        and str(x) not in state_names
        and str(x) not in excluded_states
    ]
    action_names = [
        str(x)
        for x in config.get("action_columns_model", [])
        if f"action__{x}" in frame.columns and str(x) not in excluded_actions
    ]
    reward_names = [
        str(x)
        for x in config.get("reward_columns", [])
        if f"reward__{x}" in frame.columns and str(x) not in excluded_rewards
    ]
    for pooled_name, sources in pooled.items():
        missing = [s for s in sources if f"reward__{s}" not in frame.columns]
        if missing:
            raise ValueError(
                f"pool_binary_rewards[{pooled_name!r}] missing parquet columns: {missing}"
            )
        # Sources are already dropped; skip the pooled name if it is excluded too
        # (e.g. v13: keep heart/stroke out of ModelSpec, and do not train CVD).
        if pooled_name in excluded_rewards:
            continue
        if pooled_name not in reward_names:
            reward_names.append(pooled_name)

    if not state_names:
        raise ValueError("model_config has no state_columns_model that match the training table")
    if not action_names:
        raise ValueError("model_config has no action_columns_model that match the training table")
    if not reward_names:
        raise ValueError("model_config has no reward_columns that match the training table")

    # The continuous list in the upstream preprocessing.json decides continuous vs categorical.
    continuous_names = set(preprocessing.get("continuous", {}).keys())
    ordinal_names = {
        str(x)
        for x in preprocessing.get(
            "ordinal",
            config.get("ordinal_state_columns", []),
        )
    }

    state_cont = [x for x in state_names if x in continuous_names]
    state_cat = [
        CategoricalFeature(
            name=x,
            values=_categories(
                frame,
                f"state__{x}",
                f"state_mask__{x}",
            ),
            ordered=x in ordinal_names,
        )
        for x in state_names
        if x not in continuous_names
    ]
    static_cont = [x for x in static_names if x in continuous_names]
    static_cat = [
        CategoricalFeature(
            name=x,
            values=_categories(
                frame,
                f"state__{x}",
                f"state_mask__{x}",
            ),
            ordered=x in ordinal_names,
        )
        for x in static_names
        if x not in continuous_names
    ]

    action_cont = []
    action_cat = []
    for name in action_names:
        if name in continuous_names:
            low, high = _action_bounds(
                frame,
                f"action__{name}",
                f"action_mask__{name}",
            )
            action_cont.append(
                ContinuousActionFeature(name=name, low=low, high=high)
            )
        else:
            action_cat.append(
                CategoricalFeature(
                    name=name,
                    values=_categories(
                        frame,
                        f"action__{name}",
                        f"action_mask__{name}",
                    ),
                )
            )

    # Rewards use different output distributions: Bernoulli for events, two-hot continuous for changes.
    reward_binary: list[str] = []
    reward_continuous: list[str] = []
    for name in reward_names:
        if (
            name in BINARY_REWARD_NAMES
            or name.startswith(BINARY_REWARD_PREFIXES)
            or name.endswith(BINARY_REWARD_SUFFIXES)
        ):
            reward_binary.append(name)
            continue
        values = _valid_values(
            frame,
            f"reward__{name}",
            f"reward_mask__{name}",
        )
        unique = set(float(x) for x in values.unique())
        if unique and unique.issubset({0.0, 1.0}) and not name.startswith("change_"):
            reward_binary.append(name)
        else:
            reward_continuous.append(name)

    if forced_continuous:
        keep_binary: list[str] = []
        for name in reward_binary:
            if name in forced_continuous:
                reward_continuous.append(name)
            else:
                keep_binary.append(name)
        reward_binary = keep_binary
        # Preserve upstream reward_columns order: binary then continuous.
        reward_order = {name: i for i, name in enumerate(reward_names)}
        reward_binary.sort(key=lambda n: reward_order.get(n, 10**9))
        reward_continuous.sort(key=lambda n: reward_order.get(n, 10**9))

    delta_series = frame["delta_t_years"] if "delta_t_years" in frame.columns else frame.get(
        "delta_time_years"
    )
    delta = pd.to_numeric(delta_series, errors="coerce").dropna()
    delta_mean = float(delta.mean()) if len(delta) else 0.0
    delta_std = float(delta.std(ddof=0)) if len(delta) else 1.0
    if not np.isfinite(delta_std) or delta_std < 1e-6:
        delta_std = 1.0

    if pooled:
        frame = materialize_pooled_binary_rewards(frame, pooled)

    reward_binary_pos_weight = _binary_pos_weights(
        frame,
        reward_binary,
        max_weight=float(binary_pos_weight_max),
    )
    probe_only = [
        str(x)
        for x in (probe_only_rewards or [])
        if str(x) in reward_binary
    ]

    return ModelSpec(
        state_continuous=state_cont,
        state_categorical=state_cat,
        static_continuous=static_cont,
        static_categorical=static_cat,
        action_continuous=action_cont,
        action_categorical=action_cat,
        reward_binary=reward_binary,
        reward_continuous=reward_continuous,
        reward_binary_pos_weight=reward_binary_pos_weight,
        probe_only_rewards=probe_only,
        pooled_binary_rewards=pooled,
        delta_t_mean=delta_mean,
        delta_t_std=delta_std,
        source_config=str(model_config_json),
        source_preprocessing=str(preprocessing_json),
    )
