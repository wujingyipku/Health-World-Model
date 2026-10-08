"""Read transition tables, group them into trajectories, and collate batches.

A cleaned transition table (parquet or CSV) is grouped by person_id into variable-length
trajectories. Continuous variables, categorical variables, missingness masks, rewards,
continuation, and the time interval are converted to the tensors the model expects.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .specs import (
    CategoricalFeature,
    ModelSpec,
    mapped_state_for_continuous_reward,
    pooled_binary_arrays,
    read_model_table,
)


# ======================== Batch data structure ========================

@dataclass
class TrajectoryBatch:
    """Variable-length person trajectories in one batch. valid masks padded positions."""
    person_id: list[str]
    wave: torch.Tensor
    next_wave: torch.Tensor
    valid: torch.Tensor
    state_cont: torch.Tensor
    state_cont_mask: torch.Tensor
    next_state_cont: torch.Tensor
    next_state_cont_mask: torch.Tensor
    state_cat: torch.Tensor
    state_cat_mask: torch.Tensor
    next_state_cat: torch.Tensor
    next_state_cat_mask: torch.Tensor
    static_cont: torch.Tensor
    static_cont_mask: torch.Tensor
    static_cat: torch.Tensor
    static_cat_mask: torch.Tensor
    action_cont: torch.Tensor
    action_cont_mask: torch.Tensor
    action_cat: torch.Tensor
    action_cat_mask: torch.Tensor
    reward: torch.Tensor
    reward_mask: torch.Tensor
    continuation: torch.Tensor
    continuation_mask: torch.Tensor
    delta_t_raw: torch.Tensor
    delta_t_norm: torch.Tensor

    def to(self, device: torch.device | str) -> "TrajectoryBatch":
        """Move every tensor in the batch to the given CPU or GPU device."""
        payload: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            payload[key] = value.to(device) if torch.is_tensor(value) else value
        return TrajectoryBatch(**payload)


# ======================== Table to person trajectories ========================

class TrajectoryDataset(Dataset[dict[str, Any]]):
    """Group a transition table by person_id into one variable-length trajectory per person."""
    def __init__(self, table_path: Path, spec: ModelSpec) -> None:
        """Read parquet or CSV, check required fields, sort by person and time, and cache trajectories."""
        frame = read_model_table(table_path)
        if "delta_t_years" not in frame.columns and "delta_time_years" in frame.columns:
            frame = frame.rename(columns={"delta_time_years": "delta_t_years"})
        required = {"person_id", "wave", "next_wave", "delta_t_years"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Data is missing required columns: {missing}")
        frame = frame.sort_values(
            ["person_id", "transition_index" if "transition_index" in frame else "wave"],
            kind="stable",
        )
        self.spec = spec
        self.trajectories: list[dict[str, Any]] = []
        for person_id, group in frame.groupby("person_id", sort=False):
            self.trajectories.append(self._convert_group(str(person_id), group))
        if not self.trajectories:
            raise ValueError(f"No usable trajectories: {table_path}")

    @staticmethod
    def _numeric(group: pd.DataFrame, col: str) -> np.ndarray:
        """Read a numeric column. Return an all-NaN array of the same length when the column is absent."""
        if col not in group.columns:
            return np.full(len(group), np.nan, dtype=np.float32)
        return pd.to_numeric(group[col], errors="coerce").to_numpy(np.float32)

    @staticmethod
    def _categorical_indices(
        values: np.ndarray,
        masks: np.ndarray,
        feature: CategoricalFeature,
    ) -> np.ndarray:
        # 0 is reserved for missing / not-applicable; valid categories are 1..K.
        """Map raw category values to embedding indices. Index 0 is reserved for missing or not applicable."""
        result = np.zeros(values.shape[0], dtype=np.int64)
        mapping = {float(value): idx + 1 for idx, value in enumerate(feature.values)}
        for row, (value, mask) in enumerate(zip(values, masks, strict=True)):
            if mask < 0.5 or not np.isfinite(value):
                continue
            if float(value) not in mapping:
                # Unseen test category is treated as missing rather than creating leakage.
                continue
            result[row] = mapping[float(value)]
        return result

    def _stack_cont(
        self,
        group: pd.DataFrame,
        prefix: str,
        mask_prefix: str,
        names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack a group of continuous variables and their observation masks."""
        if not names:
            empty = np.zeros((len(group), 0), dtype=np.float32)
            return empty, empty.copy()
        values = np.stack([self._numeric(group, f"{prefix}{x}") for x in names], axis=-1)
        masks = np.stack(
            [self._numeric(group, f"{mask_prefix}{x}") for x in names], axis=-1
        )
        masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
        values = np.nan_to_num(values, nan=0.0).astype(np.float32)
        return values, masks

    def _stack_cat(
        self,
        group: pd.DataFrame,
        prefix: str,
        mask_prefix: str,
        features: list[CategoricalFeature],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack embedding indices and observation masks for a group of categorical variables."""
        if not features:
            empty_i = np.zeros((len(group), 0), dtype=np.int64)
            empty_f = np.zeros((len(group), 0), dtype=np.float32)
            return empty_i, empty_f
        indices: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for feature in features:
            value = self._numeric(group, f"{prefix}{feature.name}")
            mask = self._numeric(group, f"{mask_prefix}{feature.name}")
            mask = np.nan_to_num(mask, nan=0.0).astype(np.float32)
            indices.append(self._categorical_indices(value, mask, feature))
            masks.append(mask)
        return np.stack(indices, axis=-1), np.stack(masks, axis=-1)

    def _reward_channel(self, group: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Binary rewards from ``reward__*``; forced-continuous from next-wave state."""
        if name in self.spec.reward_continuous:
            state = mapped_state_for_continuous_reward(name, self.spec.state_continuous)
            if state:
                value_col = f"next_state__{state}"
                mask_col = f"next_state_mask__{state}"
                if value_col in group.columns:
                    values = self._numeric(group, value_col)
                    masks = self._numeric(group, mask_col)
                    return values, masks
        pooled = self.spec.pooled_binary_rewards.get(name)
        if pooled:
            return pooled_binary_arrays(group, pooled)
        return (
            self._numeric(group, f"reward__{name}"),
            self._numeric(group, f"reward_mask__{name}"),
        )

    def _convert_group(self, person_id: str, group: pd.DataFrame) -> dict[str, Any]:
        """Convert one person's transitions into a single NumPy trajectory dictionary."""
        state_cont, state_cont_mask = self._stack_cont(
            group,
            "state__",
            "state_mask__",
            self.spec.state_continuous,
        )
        next_state_cont, next_state_cont_mask = self._stack_cont(
            group,
            "next_state__",
            "next_state_mask__",
            self.spec.state_continuous,
        )
        state_cat, state_cat_mask = self._stack_cat(
            group,
            "state__",
            "state_mask__",
            self.spec.state_categorical,
        )
        next_state_cat, next_state_cat_mask = self._stack_cat(
            group,
            "next_state__",
            "next_state_mask__",
            self.spec.state_categorical,
        )
        # Static context is person-level; reuse state__/state_mask__ columns and
        # do not ask the StateDecoder to reconstruct next_* counterparts.
        static_cont, static_cont_mask = self._stack_cont(
            group,
            "state__",
            "state_mask__",
            self.spec.static_continuous,
        )
        static_cat, static_cat_mask = self._stack_cat(
            group,
            "state__",
            "state_mask__",
            self.spec.static_categorical,
        )
        action_cont, action_cont_mask = self._stack_cont(
            group,
            "action__",
            "action_mask__",
            [x.name for x in self.spec.action_continuous],
        )
        action_cat, action_cat_mask = self._stack_cat(
            group,
            "action__",
            "action_mask__",
            self.spec.action_categorical,
        )

        # Rewards are concatenated in ModelSpec order. The mask keeps "cannot be judged" distinct from a true 0.
        # Continuous rewards supervise the residual head with the standardized next_state level, not the 0/1 flag in the parquet.
        reward_names = self.spec.reward_names
        if reward_names:
            channels = [self._reward_channel(group, x) for x in reward_names]
            reward = np.stack([c[0] for c in channels], axis=-1)
            reward_mask = np.stack([c[1] for c in channels], axis=-1)
        else:
            reward = np.zeros((len(group), 0), dtype=np.float32)
            reward_mask = np.zeros((len(group), 0), dtype=np.float32)
        reward = np.nan_to_num(reward, nan=0.0).astype(np.float32)
        reward_mask = np.nan_to_num(reward_mask, nan=0.0).astype(np.float32)

        continuation = self._numeric(group, "continuation")
        continuation_mask = np.isfinite(continuation).astype(np.float32)
        continuation = np.nan_to_num(continuation, nan=0.0).astype(np.float32)

        # Inter-wave intervals are irregular. Missing values use the training-set mean, and a standardized copy is also stored.
        delta_raw = self._numeric(group, "delta_t_years")
        delta_fallback = self.spec.delta_t_mean
        delta_raw = np.where(np.isfinite(delta_raw), delta_raw, delta_fallback).astype(np.float32)
        delta_norm = ((delta_raw - self.spec.delta_t_mean) / self.spec.delta_t_std).astype(
            np.float32
        )

        return {
            "person_id": person_id,
            "wave": self._numeric(group, "wave"),
            "next_wave": self._numeric(group, "next_wave"),
            "state_cont": state_cont,
            "state_cont_mask": state_cont_mask,
            "next_state_cont": next_state_cont,
            "next_state_cont_mask": next_state_cont_mask,
            "state_cat": state_cat,
            "state_cat_mask": state_cat_mask,
            "next_state_cat": next_state_cat,
            "next_state_cat_mask": next_state_cat_mask,
            "static_cont": static_cont,
            "static_cont_mask": static_cont_mask,
            "static_cat": static_cat,
            "static_cat_mask": static_cat_mask,
            "action_cont": action_cont,
            "action_cont_mask": action_cont_mask,
            "action_cat": action_cat,
            "action_cat_mask": action_cat_mask,
            "reward": reward,
            "reward_mask": reward_mask,
            "continuation": continuation[:, None],
            "continuation_mask": continuation_mask[:, None],
            "delta_t_raw": delta_raw[:, None],
            "delta_t_norm": delta_norm[:, None],
        }

    def __len__(self) -> int:
        """Number of person trajectories, not the number of transition rows."""
        return len(self.trajectories)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one respondent's full trajectory by index."""
        return self.trajectories[index]


# ======================== Variable-length trajectory batching ========================

def _pad_array(
    arrays: list[np.ndarray],
    max_len: int,
    dtype: np.dtype,
) -> torch.Tensor:
    """Zero-pad arrays of different lengths to a common time length and convert them to a tensor."""
    tail = arrays[0].shape[1:]
    out = np.zeros((len(arrays), max_len, *tail), dtype=dtype)
    for row, array in enumerate(arrays):
        out[row, : len(array)] = array
    return torch.from_numpy(out)


def collate_trajectories(items: list[dict[str, Any]]) -> TrajectoryBatch:
    """Pad variable-length trajectories and assemble them into a TrajectoryBatch."""
    max_len = max(len(x["wave"]) for x in items)
    valid = np.zeros((len(items), max_len), dtype=np.float32)
    for row, item in enumerate(items):
        valid[row, : len(item["wave"])] = 1.0

    def collect(name: str, dtype: np.dtype) -> torch.Tensor:
        """Collect one field and pad it to the maximum length in this batch."""
        return _pad_array([x[name] for x in items], max_len, dtype)

    return TrajectoryBatch(
        person_id=[x["person_id"] for x in items],
        wave=collect("wave", np.float32),
        next_wave=collect("next_wave", np.float32),
        valid=torch.from_numpy(valid),
        state_cont=collect("state_cont", np.float32),
        state_cont_mask=collect("state_cont_mask", np.float32),
        next_state_cont=collect("next_state_cont", np.float32),
        next_state_cont_mask=collect("next_state_cont_mask", np.float32),
        state_cat=collect("state_cat", np.int64),
        state_cat_mask=collect("state_cat_mask", np.float32),
        next_state_cat=collect("next_state_cat", np.int64),
        next_state_cat_mask=collect("next_state_cat_mask", np.float32),
        static_cont=collect("static_cont", np.float32),
        static_cont_mask=collect("static_cont_mask", np.float32),
        static_cat=collect("static_cat", np.int64),
        static_cat_mask=collect("static_cat_mask", np.float32),
        action_cont=collect("action_cont", np.float32),
        action_cont_mask=collect("action_cont_mask", np.float32),
        action_cat=collect("action_cat", np.int64),
        action_cat_mask=collect("action_cat_mask", np.float32),
        reward=collect("reward", np.float32),
        reward_mask=collect("reward_mask", np.float32),
        continuation=collect("continuation", np.float32),
        continuation_mask=collect("continuation_mask", np.float32),
        delta_t_raw=collect("delta_t_raw", np.float32),
        delta_t_norm=collect("delta_t_norm", np.float32),
    )


def build_loader(
    table_path: Path,
    spec: ModelSpec,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[TrajectoryBatch]:
    """Build a DataLoader that samples trajectories and uses the custom collate function."""
    dataset = TrajectoryDataset(table_path, spec)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_trajectories,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
