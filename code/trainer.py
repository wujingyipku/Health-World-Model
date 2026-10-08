"""Training and evaluation for the health world model.

Implements the JEPA embedding-prediction loss, clinical projection heads,
one-step open-loop evaluation (the prior prefix follows the Dreamer evaluation protocol),
training logs, and checkpoints.
"""
from __future__ import annotations

import contextlib
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import TrajectoryBatch
from .nn import (
    JEPAAgent,
    cosine_distance,
    embedding_variance,
    masked_mean,
)
from .specs import ModelSpec


# ======================== Training utilities ========================

class MetricAverager:
    """Accumulate finite values by metric name and average them across batches."""

    def __init__(self) -> None:
        self.total: dict[str, float] = defaultdict(float)
        self.count: dict[str, int] = defaultdict(int)

    def update(self, metrics: dict[str, float | torch.Tensor]) -> None:
        for key, value in metrics.items():
            number = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            if math.isfinite(number):
                self.total[key] += number
                self.count[key] += 1

    def mean(self) -> dict[str, float]:
        return {
            key: self.total[key] / max(self.count[key], 1)
            for key in sorted(self.total)
        }


class ReturnNormalizer:
    """Normalize imagined returns with a moving 5th–95th percentile range (optional actor-critic stub)."""

    def __init__(self, decay: float = 0.99) -> None:
        self.decay = decay
        self.low = 0.0
        self.high = 1.0
        self.initialized = False

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().float().reshape(-1)
        if flat.numel() == 0:
            return
        low = float(torch.quantile(flat, 0.05).cpu())
        high = float(torch.quantile(flat, 0.95).cpu())
        if not self.initialized:
            self.low, self.high = low, high
            self.initialized = True
        else:
            self.low = self.decay * self.low + (1 - self.decay) * low
            self.high = self.decay * self.high + (1 - self.decay) * high

    @property
    def scale(self) -> float:
        return max(self.high - self.low, 1.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "low": self.low,
            "high": self.high,
            "initialized": self.initialized,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.low = float(state["low"])
        self.high = float(state["high"])
        self.initialized = bool(state["initialized"])


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


@contextlib.contextmanager
def freeze_modules(*modules: nn.Module) -> Iterator[None]:
    states: list[tuple[nn.Parameter, bool]] = []
    for module in modules:
        for parameter in module.parameters():
            states.append((parameter, parameter.requires_grad))
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in states:
            parameter.requires_grad_(requires_grad)


def encode_batch_observation(
    world: Any,
    batch: TrajectoryBatch,
    t: int,
    *,
    next_state: bool = False,
) -> torch.Tensor:
    """Dynamic-only wave encoder (no temporal encoder and no static context)."""
    if next_state:
        cont = batch.next_state_cont[:, t]
        cont_mask = batch.next_state_cont_mask[:, t]
        cat = batch.next_state_cat[:, t]
        cat_mask = batch.next_state_cat_mask[:, t]
    else:
        cont = batch.state_cont[:, t]
        cont_mask = batch.state_cont_mask[:, t]
        cat = batch.state_cat[:, t]
        cat_mask = batch.state_cat_mask[:, t]
    return world.encode_observation(cont, cont_mask, cat, cat_mask)


def encode_batch_static(
    world: Any,
    batch: TrajectoryBatch,
    t: int = 0,
) -> torch.Tensor:
    """Static condition embedding (person-level; the t=0 slice by default)."""
    return world.encode_static(
        batch.static_cont[:, t],
        batch.static_cont_mask[:, t],
        batch.static_cat[:, t],
        batch.static_cat_mask[:, t],
    )


def encode_batch_target(
    world: Any,
    batch: TrajectoryBatch,
    t: int,
) -> torch.Tensor:
    """Last frame of EMA target_z = Temporal_EMA(s_1:t+1), without static context.

    Sequence layout: ``[s_0, …, s_t, s_{t+1}]``, where ``s_{t+1}`` is this step's ``next_state``.
    """
    cont = torch.cat(
        [batch.state_cont[:, : t + 1], batch.next_state_cont[:, t : t + 1]], dim=1
    )
    cont_mask = torch.cat(
        [batch.state_cont_mask[:, : t + 1], batch.next_state_cont_mask[:, t : t + 1]],
        dim=1,
    )
    cat = torch.cat(
        [batch.state_cat[:, : t + 1], batch.next_state_cat[:, t : t + 1]], dim=1
    )
    cat_mask = torch.cat(
        [batch.state_cat_mask[:, : t + 1], batch.next_state_cat_mask[:, t : t + 1]],
        dim=1,
    )
    # Whether the next wave is valid follows this transition's valid flag.
    valid = torch.cat([batch.valid[:, : t + 1], batch.valid[:, t : t + 1]], dim=1)
    return world.encode_target(cont, cont_mask, cat, cat_mask, valid)


def encode_batch_trajectory(
    world: Any,
    batch: TrajectoryBatch,
) -> torch.Tensor:
    """Dynamic Wave 1…T through the temporal transformer to a z sequence [B, T, D]."""
    return world.encode_trajectory_waves(
        batch.state_cont,
        batch.state_cont_mask,
        batch.state_cat,
        batch.state_cat_mask,
        batch.valid,
    )


# ======================== World-model learning ========================

def resolve_binary_event_scales(
    reward_binary: Sequence[str],
    scale_map: Mapping[str, Any] | None,
) -> list[float]:
    """Map event-name scales to the reward_binary channel order (default 1.0)."""
    mapping = {str(k): float(v) for k, v in dict(scale_map or {}).items()}
    return [float(mapping.get(name, 1.0)) for name in reward_binary]


def _jepa_embedding_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    """Masked JEPA embedding distance: ``mse`` or ``cosine``."""
    kind = str(kind).lower().strip()
    if kind == "cosine":
        dist = cosine_distance(pred, target)
    elif kind == "mse":
        dist = (pred - target).square().mean(dim=-1)
    else:
        raise ValueError(f"Unsupported jepa_loss={kind!r}; expected 'mse' or 'cosine'")
    return masked_mean(dist, valid)


def _resolve_multi_horizon_weights(cfg: Mapping[str, Any], horizon: int) -> list[float]:
    """Per-horizon weights for h=1..H (default geometric: 1, 1/2, 1/4, ...)."""
    horizon = max(int(horizon), 1)
    raw = cfg.get("multi_horizon_weights")
    if raw is None:
        return [0.5 ** (h - 1) for h in range(1, horizon + 1)]
    weights = [float(x) for x in list(raw)]
    if not weights:
        return [1.0] * horizon
    if len(weights) < horizon:
        weights = weights + [weights[-1]] * (horizon - len(weights))
    return weights[:horizon]


def resolve_ever_event_rewards(
    reward_binary: Sequence[str],
    names: Sequence[str] | None,
) -> list[tuple[str, int]]:
    """``(name, channel_index)`` for ever-event auxiliaries present in ModelSpec."""
    if not names:
        return []
    lookup = {str(n): i for i, n in enumerate(reward_binary)}
    out: list[tuple[str, int]] = []
    for raw in names:
        name = str(raw)
        if name in lookup:
            out.append((name, int(lookup[name])))
    return out


def ever_event_aggregate_prob(
    probs: torch.Tensor,
    step_mask: torch.Tensor,
    *,
    aggregate: str = "product",
) -> torch.Tensor:
    """Aggregate per-step probs ``[B, H]`` into ever-event probability ``[B]``.

    ``product``: ``1 - ∏_h (1 - p_h)`` over masked steps (discrete survival).
    ``max``: masked maximum of ``p_h``.
    """
    mask = step_mask.to(dtype=probs.dtype)
    kind = str(aggregate).lower().strip()
    if kind == "max":
        filled = torch.where(mask > 0.5, probs, probs.new_zeros(probs.shape))
        return filled.max(dim=-1).values
    one_minus = (1.0 - probs).clamp(min=1e-6, max=1.0)
    factors = torch.where(mask > 0.5, one_minus, torch.ones_like(one_minus))
    survival = factors.prod(dim=-1).clamp(min=1e-6, max=1.0)
    return (1.0 - survival).clamp(min=1e-6, max=1.0 - 1e-6)


def ever_event_bce_loss(
    p_ever: torch.Tensor,
    y_ever: torch.Tensor,
    valid: torch.Tensor,
    *,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Masked BCE on ever-event probabilities with optional positive weight."""
    valid_f = valid.to(dtype=p_ever.dtype)
    if float(valid_f.sum().detach()) <= 0:
        return p_ever.new_zeros(())
    p = p_ever.clamp(1e-6, 1.0 - 1e-6)
    y = y_ever.to(dtype=p.dtype)
    loss = F.binary_cross_entropy(p, y, reduction="none")
    pw = float(pos_weight)
    if pw > 1.0 + 1e-8:
        weights = torch.where(y > 0.5, p.new_tensor(pw), torch.ones_like(p))
        loss = loss * weights
    return masked_mean(loss, valid_f)


def _ever_event_pos_weight(
    spec: ModelSpec,
    channel: int,
    *,
    max_weight: float,
) -> float:
    """Positive-class weight for ever-BCE; ``max_weight<=0`` disables (1.0)."""
    if float(max_weight) <= 0:
        return 1.0
    weights = list(spec.reward_binary_pos_weight)
    if 0 <= channel < len(weights):
        raw = float(weights[channel])
    else:
        raw = 1.0
    return float(min(max(raw, 1.0), float(max_weight)))


def world_model_loss(
    agent: JEPAAgent,
    batch: TrajectoryBatch,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """JEPA + clinical loss with optional multi-horizon open-loop unroll.

    For each start step ``t`` and horizon ``h=1..H`` (``k=t+h-1``)::

        z <- Predictor^h (z_t, a_{t:k}, Δt)
        L_JEPA(h)  = d(z, EMA_Temporal(s_1:k+1))
        L_clin(h)  = ClinicalHeads(z; reward_k), residual base open-loop for h>1

    Optional ever-event auxiliary (v12; aligns Fig 4O/4P observed strata)::

        p_ever = 1 - ∏_h (1 - p_h)   (or max_h p_h)
        y_ever = 1{∃ k: reward_mask & reward}
        L_ever = BCE(p_ever, y_ever) for configured binary rewards

    Config::

        multi_horizon: H (1 = one-step only)
        multi_horizon_weights: [w1, w2, ...]  (default 1, 1/2, 1/4, ...)
        multi_horizon_clinical: bool (apply clinical on h>=2; default True)
        multi_horizon_clinical_scale: extra scale on clinical for h>=2 (default 0.5)
        ever_event_loss_scale / ever_event_rewards / ever_event_aggregate
    """
    world = agent.world
    _, steps = batch.valid.shape
    total = batch.valid.new_zeros(())
    sums: dict[str, torch.Tensor] = defaultdict(lambda: batch.valid.new_zeros(()))

    event_scales = resolve_binary_event_scales(
        agent.spec.reward_binary,
        cfg.get("reward_binary_event_scales")
        or cfg.get("prior_reward_binary_event_scales"),
    )
    jepa_kind = str(cfg.get("jepa_loss", "mse"))
    jepa_scale = float(cfg.get("jepa_loss_scale", 1.0))
    var_scale = float(cfg.get("variance_loss_scale", 0.0))
    var_floor = float(cfg.get("embedding_variance_floor", 1.0))

    default_reward_scale = float(cfg.get("reward_loss_scale", 1.0))
    reward_binary_scale = float(
        cfg.get(
            "reward_binary_loss_scale",
            cfg.get("prior_reward_binary_loss_scale", default_reward_scale),
        )
    )
    reward_cont_scale = float(
        cfg.get(
            "reward_continuous_loss_scale",
            cfg.get("prior_reward_continuous_loss_scale", default_reward_scale),
        )
    )
    reward_delta_scale = float(
        cfg.get(
            "reward_continuous_delta_loss_scale",
            cfg.get("prior_reward_continuous_delta_loss_scale", reward_cont_scale),
        )
    )

    multi_h = max(int(cfg.get("multi_horizon", 1)), 1)
    mh_weights = _resolve_multi_horizon_weights(cfg, multi_h)
    mh_clinical = bool(cfg.get("multi_horizon_clinical", True))
    mh_clin_scale = float(cfg.get("multi_horizon_clinical_scale", 0.5))

    ever_scale = float(cfg.get("ever_event_loss_scale", 0.0))
    ever_channels = resolve_ever_event_rewards(
        agent.spec.reward_binary,
        cfg.get("ever_event_rewards"),
    )
    ever_aggregate = str(cfg.get("ever_event_aggregate", "product"))
    ever_pw_max = float(cfg.get("ever_event_pos_weight_max", 10.0))
    use_ever = ever_scale > 0.0 and bool(ever_channels) and world.clinical_heads.has_binary

    z_seq = encode_batch_trajectory(world, batch)
    static_embed = encode_batch_static(world, batch, t=0)
    n_cont = len(agent.spec.reward_continuous)

    for t in range(steps):
        z_roll = z_seq[:, t]
        cur_levels: torch.Tensor | None = None
        cur_mask: torch.Tensor | None = None
        if n_cont > 0:
            cur_levels, cur_mask = world.clinical_heads._current_levels_for_rewards(
                batch.state_cont[:, t], batch.state_cont_mask[:, t]
            )

        step_loss = batch.valid.new_zeros(())
        jepa_step = batch.valid.new_zeros(())
        reward_term_step = batch.valid.new_zeros(())
        probe_term_step = batch.valid.new_zeros(())
        ever_term_step = batch.valid.new_zeros(())
        var_step = batch.valid.new_zeros(())
        emb_var_step = batch.valid.new_zeros(())

        ever_prob_lists: dict[str, list[torch.Tensor]] = {
            name: [] for name, _ in ever_channels
        }
        ever_pred_masks: list[torch.Tensor] = []
        ever_label_lists: dict[str, list[torch.Tensor]] = {
            name: [] for name, _ in ever_channels
        }
        ever_label_masks: dict[str, list[torch.Tensor]] = {
            name: [] for name, _ in ever_channels
        }

        for h, w_h in enumerate(mh_weights, start=1):
            k = t + h - 1
            if k >= steps:
                break
            valid_k = batch.valid[:, k]
            z_roll = world.predict_next(
                z_roll,
                batch.action_cont[:, k],
                batch.action_cont_mask[:, k],
                batch.action_cat[:, k],
                batch.delta_t_norm[:, k],
                static_embed=static_embed,
            )
            target = encode_batch_target(world, batch, k)
            jepa_h = _jepa_embedding_loss(z_roll, target, valid_k, jepa_kind)
            jepa_step = jepa_step + float(w_h) * jepa_h

            if h == 1:
                active = valid_k > 0.5
                if bool(active.any()) and active.sum() >= 2:
                    emb_var_step = embedding_variance(z_roll[active])
                else:
                    emb_var_step = z_roll.new_zeros(())
                var_step = F.relu(z_roll.new_tensor(var_floor) - emb_var_step)

            if use_ever:
                ever_logits = world.clinical_heads.binary_logits(z_roll)
                ever_probs_h = torch.sigmoid(ever_logits)
                ever_pred_masks.append(valid_k)
                for name, idx in ever_channels:
                    ever_prob_lists[name].append(ever_probs_h[:, idx])
                    ever_label_lists[name].append(batch.reward[:, k, idx])
                    ever_label_masks[name].append(
                        batch.reward_mask[:, k, idx] * valid_k
                    )

            apply_clinical = (h == 1) or mh_clinical
            if apply_clinical:
                clin_w = float(w_h) if h == 1 else float(w_h) * mh_clin_scale
                (
                    reward_binary_loss,
                    reward_cont_level_loss,
                    reward_cont_delta_loss,
                    reward_metrics,
                ) = world.clinical_heads.loss(
                    z_roll,
                    batch.reward[:, k],
                    batch.reward_mask[:, k],
                    valid_k,
                    binary_event_scales=event_scales,
                    current_levels=cur_levels,
                    current_levels_mask=cur_mask,
                )
                reward_cont_loss = reward_cont_level_loss + reward_cont_delta_loss
                reward_loss = reward_binary_loss + reward_cont_loss
                probe_binary_h = reward_metrics.pop(
                    "_probe_binary_loss", z_roll.new_zeros(())
                )
                reward_term_h = (
                    reward_binary_scale * reward_binary_loss
                    + reward_cont_scale * reward_cont_level_loss
                    + reward_delta_scale * reward_cont_delta_loss
                )
                probe_term_h = reward_binary_scale * probe_binary_h
                reward_term_step = reward_term_step + clin_w * reward_term_h
                probe_term_step = probe_term_step + clin_w * probe_term_h
                sums["clinical_mh_loss"] = sums["clinical_mh_loss"] + (
                    clin_w * (reward_binary_loss + reward_cont_loss)
                ).detach()

                if h == 1:
                    sums["reward_loss"] = sums["reward_loss"] + reward_loss.detach()
                    sums["clinical_binary_loss"] = (
                        sums["clinical_binary_loss"] + reward_binary_loss.detach()
                    )
                    sums["clinical_continuous_loss"] = (
                        sums["clinical_continuous_loss"] + reward_cont_loss.detach()
                    )
                    sums["clinical_continuous_level_loss"] = (
                        sums["clinical_continuous_level_loss"]
                        + reward_cont_level_loss.detach()
                    )
                    sums["clinical_continuous_delta_loss"] = (
                        sums["clinical_continuous_delta_loss"]
                        + reward_cont_delta_loss.detach()
                    )
                    sums["reward_binary_loss"] = (
                        sums["reward_binary_loss"] + reward_binary_loss.detach()
                    )
                    sums["reward_continuous_loss"] = (
                        sums["reward_continuous_loss"] + reward_cont_loss.detach()
                    )
                    sums["reward_continuous_level_loss"] = (
                        sums["reward_continuous_level_loss"]
                        + reward_cont_level_loss.detach()
                    )
                    sums["reward_continuous_delta_loss"] = (
                        sums["reward_continuous_delta_loss"]
                        + reward_cont_delta_loss.detach()
                    )
                    sums["prior_reward_binary_loss"] = (
                        sums["prior_reward_binary_loss"] + reward_binary_loss.detach()
                    )
                    sums["prior_reward_continuous_loss"] = (
                        sums["prior_reward_continuous_loss"] + reward_cont_loss.detach()
                    )
                    for name, value in reward_metrics.items():
                        sums[name] = sums[name] + value
                        sums[f"prior_aux_{name}"] = sums[f"prior_aux_{name}"] + value
                else:
                    sums["multi_horizon_clinical_loss"] = (
                        sums["multi_horizon_clinical_loss"]
                        + (clin_w * reward_term_h).detach()
                    )

                # Open-loop residual base for next horizon.
                if n_cont > 0 and h < multi_h and (t + h) < steps:
                    pred_dict = world.clinical_heads.mean_dict(
                        z_roll,
                        current_levels=cur_levels,
                        current_levels_mask=cur_mask,
                    )
                    cur_levels = torch.stack(
                        [pred_dict[name] for name in agent.spec.reward_continuous],
                        dim=-1,
                    )
                    if cur_mask is None:
                        cur_mask = cur_levels.new_ones(cur_levels.shape)
                    else:
                        cur_mask = cur_mask.new_ones(cur_mask.shape)

            sums[f"jepa_loss_h{h}"] = sums[f"jepa_loss_h{h}"] + jepa_h.detach()

        if use_ever and ever_pred_masks:
            pred_mask = torch.stack(ever_pred_masks, dim=-1)
            start_valid = batch.valid[:, t]
            for name, idx in ever_channels:
                probs = torch.stack(ever_prob_lists[name], dim=-1)
                labels = torch.stack(ever_label_lists[name], dim=-1)
                label_mask = torch.stack(ever_label_masks[name], dim=-1)
                y_ever = ((labels * label_mask).sum(dim=-1) > 0.5).to(dtype=probs.dtype)
                valid_ever = ((label_mask.sum(dim=-1) > 0.5) & (start_valid > 0.5)).to(
                    dtype=probs.dtype
                )
                p_ever = ever_event_aggregate_prob(
                    probs, pred_mask, aggregate=ever_aggregate
                )
                pw = _ever_event_pos_weight(
                    agent.spec, idx, max_weight=ever_pw_max
                )
                ever_h = ever_event_bce_loss(
                    p_ever, y_ever, valid_ever, pos_weight=pw
                )
                ever_term_step = ever_term_step + ever_h
                sums[f"ever_event_loss__{name}"] = (
                    sums[f"ever_event_loss__{name}"] + ever_h.detach()
                )

        ever_term = ever_scale * ever_term_step
        # Include ever in clinical_mh so early-stop / best_final tracks Fig 4O/4P.
        sums["clinical_mh_loss"] = sums["clinical_mh_loss"] + ever_term.detach()
        jepa_clinical = jepa_scale * jepa_step + reward_term_step + ever_term
        rep_step = jepa_clinical + var_scale * var_step
        step_loss = rep_step + probe_term_step
        total = total + step_loss

        sums["world_loss"] = sums["world_loss"] + rep_step.detach()
        sums["jepa_clinical_loss"] = sums["jepa_clinical_loss"] + jepa_clinical.detach()
        sums["jepa_loss"] = sums["jepa_loss"] + jepa_step.detach()
        sums["variance_loss"] = sums["variance_loss"] + var_step.detach()
        sums["embedding_variance"] = sums["embedding_variance"] + emb_var_step.detach()
        sums["prior_aux_loss"] = sums["prior_aux_loss"] + reward_term_step.detach()
        sums["probe_binary_loss"] = sums["probe_binary_loss"] + probe_term_step.detach()
        sums["ever_event_loss"] = sums["ever_event_loss"] + ever_term.detach()
        if multi_h > 1:
            sums["multi_horizon"] = sums["multi_horizon"] + z_roll.new_tensor(float(multi_h))

    denominator = max(steps, 1)
    total = total / denominator
    metrics = {name: value / denominator for name, value in sums.items()}
    return total, metrics


def resolve_clinical_finetune_loss_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``clinical_finetune`` overrides into a copy of ``world_loss``."""
    world_loss = dict(cfg.get("world_loss") or {})
    finetune = dict(cfg.get("clinical_finetune") or {})
    for key in (
        "binary_reward_pos_weight_max",
        "reward_binary_event_scales",
        "reward_binary_loss_scale",
        "multi_horizon",
        "multi_horizon_weights",
        "multi_horizon_clinical",
        "multi_horizon_clinical_scale",
    ):
        if key in finetune:
            world_loss[key] = finetune[key]
    return world_loss


def configure_clinical_finetune(
    agent: JEPAAgent,
    *,
    trainable_binary_groups: Sequence[str] = ("death",),
) -> list[torch.nn.Parameter]:
    """Freeze world backbone; train only selected ``ClinicalHeads`` binary groups."""
    world = agent.world
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    trainable: list[torch.nn.Parameter] = []
    heads = world.clinical_heads
    allowed = {str(name) for name in trainable_binary_groups}
    for group_name, module in heads.binary_groups.items():
        if group_name in allowed:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                trainable.append(parameter)
    return trainable


def apply_finetune_pos_weight_cap(
    agent: JEPAAgent,
    spec: ModelSpec,
    *,
    max_weight: float,
    reward_names: Sequence[str] = ("death_event",),
) -> None:
    """Cap selected binary ``pos_weight`` entries on the live ClinicalHeads buffer."""
    if not agent.world.clinical_heads.has_binary:
        return
    weights = agent.world.clinical_heads.binary_pos_weight.clone()
    name_to_idx = {name: idx for idx, name in enumerate(spec.reward_binary)}
    cap = float(max_weight)
    for name in reward_names:
        idx = name_to_idx.get(name)
        if idx is None or idx >= weights.numel():
            continue
        weights[idx] = min(float(weights[idx]), cap)
    agent.world.clinical_heads.binary_pos_weight.copy_(weights)


def _binary_group_clinical_loss(
    heads: Any,
    group_name: str,
    feature: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    *,
    event_scales: Sequence[float],
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    """BCE for one ClinicalHeads binary group (e.g. ``death``)."""
    names = heads.binary_group_names[group_name]
    name_to_idx = {name: idx for idx, name in enumerate(heads.spec.reward_binary)}
    idxs = [name_to_idx[name] for name in names]
    logits = heads.binary_groups[group_name](feature)
    group_target = torch.stack([target[..., idx] for idx in idxs], dim=-1)
    group_pw = None
    if pos_weight is not None and pos_weight.numel() == len(heads.spec.reward_binary):
        group_pw = pos_weight[idxs]
    loss_all = F.binary_cross_entropy_with_logits(
        logits,
        group_target,
        reduction="none",
        pos_weight=group_pw,
    )
    scales = torch.tensor(
        [float(event_scales[idx]) for idx in idxs],
        device=loss_all.device,
        dtype=loss_all.dtype,
    )
    while scales.ndim < loss_all.ndim:
        scales = scales.unsqueeze(0)
    loss_all = loss_all * scales
    group_mask = torch.stack([mask[..., idx] for idx in idxs], dim=-1) * valid[..., None]
    return masked_mean(loss_all, group_mask)


def clinical_finetune_loss(
    agent: JEPAAgent,
    batch: TrajectoryBatch,
    cfg: dict[str, Any],
    *,
    trainable_binary_groups: Sequence[str] = ("death",),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Open-loop clinical loss for frozen-backbone head fine-tuning (death-only default).

    No JEPA / variance / ever-event / EMA target update. Uses Predictor rollout +
    multi-horizon death BCE only.
    """
    world = agent.world
    heads = world.clinical_heads
    _, steps = batch.valid.shape
    total = batch.valid.new_zeros(())
    sums: dict[str, torch.Tensor] = defaultdict(lambda: batch.valid.new_zeros(()))

    event_scales = resolve_binary_event_scales(
        agent.spec.reward_binary,
        cfg.get("reward_binary_event_scales")
        or cfg.get("prior_reward_binary_event_scales"),
    )
    reward_binary_scale = float(
        cfg.get(
            "reward_binary_loss_scale",
            cfg.get("prior_reward_binary_loss_scale", 1.0),
        )
    )
    multi_h = max(int(cfg.get("multi_horizon", 1)), 1)
    mh_weights = _resolve_multi_horizon_weights(cfg, multi_h)
    mh_clinical = bool(cfg.get("multi_horizon_clinical", True))
    mh_clin_scale = float(cfg.get("multi_horizon_clinical_scale", 1.0))
    pos_weight = heads.binary_pos_weight if heads.has_binary else None
    groups = [str(g) for g in trainable_binary_groups if g in heads.binary_groups]

    z_seq = encode_batch_trajectory(world, batch)
    static_embed = encode_batch_static(world, batch, t=0)

    for t in range(steps):
        z_roll = z_seq[:, t]
        death_step = batch.valid.new_zeros(())

        for h, w_h in enumerate(mh_weights, start=1):
            k = t + h - 1
            if k >= steps:
                break
            valid_k = batch.valid[:, k]
            z_roll = world.predict_next(
                z_roll,
                batch.action_cont[:, k],
                batch.action_cont_mask[:, k],
                batch.action_cat[:, k],
                batch.delta_t_norm[:, k],
                static_embed=static_embed,
            )
            apply_clinical = (h == 1) or mh_clinical
            if not apply_clinical or not groups:
                continue
            clin_w = float(w_h) if h == 1 else float(w_h) * mh_clin_scale
            group_loss = batch.valid.new_zeros(())
            z_feat = z_roll.detach()
            for group_name in groups:
                group_loss = group_loss + _binary_group_clinical_loss(
                    heads,
                    group_name,
                    z_feat,
                    batch.reward[:, k],
                    batch.reward_mask[:, k],
                    valid_k,
                    event_scales=event_scales,
                    pos_weight=pos_weight,
                )
                sums[f"reward_binary_bce__{group_name}"] = (
                    sums[f"reward_binary_bce__{group_name}"] + group_loss.detach()
                )
            death_step = death_step + clin_w * group_loss
            if h == 1:
                sums["clinical_binary_loss"] = (
                    sums["clinical_binary_loss"] + group_loss.detach()
                )
            else:
                sums["multi_horizon_clinical_loss"] = (
                    sums["multi_horizon_clinical_loss"] + (clin_w * group_loss).detach()
                )

        total = total + reward_binary_scale * death_step
        sums["clinical_finetune_loss"] = (
            sums["clinical_finetune_loss"] + death_step.detach()
        )

    denominator = max(steps, 1)
    total = total / denominator
    metrics = {name: value / denominator for name, value in sums.items()}
    metrics["clinical_mh_loss"] = metrics.get("clinical_finetune_loss", total.detach())
    return total, metrics


def jepa_clinical_selection_weights(cfg: Mapping[str, Any]) -> tuple[float, float]:
    """``w_jepa``, ``w_clin`` for early-stop / best_final (explicit JEPA+clinical)."""
    world = cfg.get("world_loss") or {}
    training = cfg.get("training") or {}
    w_jepa = float(
        training.get(
            "early_stopping_jepa_weight",
            world.get("jepa_loss_scale", 1.5),
        )
    )
    default_clin = world.get(
        "reward_binary_loss_scale",
        world.get("reward_loss_scale", 0.7),
    )
    w_clin = float(training.get("early_stopping_clinical_weight", default_clin))
    return w_jepa, w_clin


DEFAULT_RARE_EVENT_REWARD_NAMES: tuple[str, ...] = ("death_event",)


def _accumulate_binary_confusion(
    confusion: dict[str, torch.Tensor],
    prefix: str,
    prob: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> None:
    active = mask > 0.5
    if not bool(active.any()):
        return
    pred_pos = prob[active] >= float(threshold)
    true_pos = target[active] >= 0.5
    zero = mask.new_zeros(())
    confusion[f"{prefix}__tp"] = confusion.get(f"{prefix}__tp", zero) + (
        pred_pos & true_pos
    ).sum()
    confusion[f"{prefix}__tn"] = confusion.get(f"{prefix}__tn", zero) + (
        (~pred_pos) & (~true_pos)
    ).sum()
    confusion[f"{prefix}__fp"] = confusion.get(f"{prefix}__fp", zero) + (
        pred_pos & (~true_pos)
    ).sum()
    confusion[f"{prefix}__fn"] = confusion.get(f"{prefix}__fn", zero) + (
        (~pred_pos) & true_pos
    ).sum()


def _binary_rates_from_counts(
    tp: float, tn: float, fp: float, fn: float
) -> dict[str, float]:
    total = tp + tn + fp + fn
    pos = tp + fn
    neg = tn + fp
    sensitivity = tp / pos if pos > 0 else 0.0
    specificity = tn / neg if neg > 0 else 0.0
    return {
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
    }


def finalize_binary_classification_metrics(
    confusion: dict[str, torch.Tensor],
    *,
    metric_prefix: str = "prior_reward",
) -> dict[str, float]:
    names = {
        key[: -len("__tp")]
        for key in confusion
        if key.endswith("__tp")
    }
    metrics: dict[str, float] = {}
    for name in sorted(names):
        tp = float(confusion.get(f"{name}__tp", 0.0))
        tn = float(confusion.get(f"{name}__tn", 0.0))
        fp = float(confusion.get(f"{name}__fp", 0.0))
        fn = float(confusion.get(f"{name}__fn", 0.0))
        rates = _binary_rates_from_counts(tp, tn, fp, fn)
        metrics[f"{metric_prefix}_accuracy__{name}"] = rates["accuracy"]
        metrics[f"{metric_prefix}_balanced_accuracy__{name}"] = rates[
            "balanced_accuracy"
        ]
        metrics[f"{metric_prefix}_sensitivity__{name}"] = rates["sensitivity"]
        metrics[f"{metric_prefix}_specificity__{name}"] = rates["specificity"]
    return metrics


def fit_balanced_accuracy_threshold(
    prob: np.ndarray,
    target: np.ndarray,
    *,
    grid_size: int = 99,
) -> tuple[float, float]:
    if prob.size == 0:
        return 0.5, 0.0
    y = target.astype(np.float64) >= 0.5
    p = prob.astype(np.float64)
    pos = int(y.sum())
    neg = int((~y).sum())
    if pos == 0 or neg == 0:
        return 0.5, 0.5

    grid = np.linspace(1.0 / (grid_size + 1), 1.0 - 1.0 / (grid_size + 1), grid_size)
    best_t = 0.5
    best_bal = -1.0
    best_dist = abs(best_t - 0.5)
    for threshold in grid:
        pred = p >= threshold
        tp = float(np.logical_and(pred, y).sum())
        tn = float(np.logical_and(~pred, ~y).sum())
        fp = float(np.logical_and(pred, ~y).sum())
        fn = float(np.logical_and(~pred, y).sum())
        bal = _binary_rates_from_counts(tp, tn, fp, fn)["balanced_accuracy"]
        dist = abs(float(threshold) - 0.5)
        if bal > best_bal + 1e-12 or (abs(bal - best_bal) <= 1e-12 and dist < best_dist):
            best_bal = bal
            best_t = float(threshold)
            best_dist = dist
    return best_t, float(best_bal)


def binary_classification_metrics_from_scores(
    prob: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    metric_prefix: str,
    name: str,
) -> dict[str, float]:
    y = target.astype(np.float64) >= 0.5
    pred = prob.astype(np.float64) >= float(threshold)
    tp = float(np.logical_and(pred, y).sum())
    tn = float(np.logical_and(~pred, ~y).sum())
    fp = float(np.logical_and(pred, ~y).sum())
    fn = float(np.logical_and(~pred, y).sum())
    rates = _binary_rates_from_counts(tp, tn, fp, fn)
    return {
        f"{metric_prefix}_accuracy__{name}": rates["accuracy"],
        f"{metric_prefix}_balanced_accuracy__{name}": rates["balanced_accuracy"],
        f"{metric_prefix}_sensitivity__{name}": rates["sensitivity"],
        f"{metric_prefix}_specificity__{name}": rates["specificity"],
    }


def safe_auroc(prob: np.ndarray, target: np.ndarray) -> float:
    y = target.astype(np.float64) >= 0.5
    p = prob.astype(np.float64)
    if y.size < 2 or np.unique(y).size < 2:
        return float("nan")
    pos = p[y]
    neg = p[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(p, dtype=np.float64)
    ranks[order] = np.arange(1, p.size + 1, dtype=np.float64)
    i = 0
    while i < p.size:
        j = i + 1
        while j < p.size and p[order[j]] == p[order[i]]:
            j += 1
        if j > i + 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j
    sum_pos_ranks = float(ranks[y].sum())
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    return (sum_pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def safe_auprc(prob: np.ndarray, target: np.ndarray) -> float:
    """Average precision (AUPRC); NaN if a class is missing."""
    y = target.astype(np.float64) >= 0.5
    p = prob.astype(np.float64)
    if y.size < 2 or np.unique(y).size < 2:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    n_pos = float(tp[-1])
    if n_pos <= 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1.0)
    # Sum precision at each positive label (sklearn average_precision style).
    return float(precision[y_sorted].sum() / n_pos)


def rare_event_reward_names(cfg: Mapping[str, Any] | None = None) -> list[str]:
    training = {}
    if cfg is not None:
        training = dict(cfg.get("training", {}))
    names = training.get("rare_event_reward_names")
    if names:
        return [str(x) for x in names]
    return list(DEFAULT_RARE_EVENT_REWARD_NAMES)


@torch.no_grad()
def prior_prediction_metrics(
    agent: JEPAAgent,
    batch: TrajectoryBatch,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
]:
    """Teacher-forced one-step metrics (JEPA v2 open-loop / prior_* keys).

    z_t from Temporal Transformer over Wave1…t; z_future = Predictor(z_t, a, Δt).
    """
    world = agent.world
    _, steps = batch.valid.shape
    sums: dict[str, torch.Tensor] = defaultdict(lambda: batch.valid.new_zeros(()))
    counts: dict[str, torch.Tensor] = defaultdict(lambda: batch.valid.new_zeros(()))
    confusion: dict[str, torch.Tensor] = {}
    binary_prob_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    binary_target_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)

    def add(name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        mask = mask.to(values.dtype)
        sums[name] = sums[name] + (values * mask).sum()
        counts[name] = counts[name] + mask.sum()

    z_seq = encode_batch_trajectory(world, batch)
    static_embed = encode_batch_static(world, batch, t=0)
    for t in range(steps):
        valid = batch.valid[:, t]
        pred = world.predict_next(
            z_seq[:, t],
            batch.action_cont[:, t],
            batch.action_cont_mask[:, t],
            batch.action_cat[:, t],
            batch.delta_t_norm[:, t],
            static_embed=static_embed,
        )

        reward_pred = world.clinical_heads.mean_dict(
            pred,
            state_cont=batch.state_cont[:, t],
            state_cont_mask=batch.state_cont_mask[:, t],
        )
        for index, name in enumerate(agent.spec.reward_binary):
            target = batch.reward[:, t, index]
            mask = batch.reward_mask[:, t, index] * valid
            prob = reward_pred[name].clamp(1e-6, 1 - 1e-6)
            bce = F.binary_cross_entropy(prob, target, reduction="none")
            add(f"prior_reward_bce__{name}", bce, mask)
            add(f"prior_reward_brier__{name}", (prob - target).square(), mask)
            _accumulate_binary_confusion(
                confusion, name, prob, target, mask, threshold=0.5
            )
            active = mask > 0.5
            if bool(active.any()):
                binary_prob_chunks[name].append(prob[active].detach().cpu())
                binary_target_chunks[name].append(target[active].detach().cpu())
        offset = len(agent.spec.reward_binary)
        for index, name in enumerate(agent.spec.reward_continuous):
            target = batch.reward[:, t, offset + index]
            mask = batch.reward_mask[:, t, offset + index] * valid
            add(f"prior_reward_mae__{name}", (reward_pred[name] - target).abs(), mask)

    metrics = {
        name: sums[name] / counts[name].clamp_min(1.0)
        for name in sums
    }
    binary_pairs = {
        name: (
            torch.cat(binary_prob_chunks[name], dim=0),
            torch.cat(binary_target_chunks[name], dim=0),
        )
        for name in binary_prob_chunks
        if binary_prob_chunks[name]
    }
    return metrics, confusion, binary_pairs


# ======================== Training controller ========================

class Trainer:
    """Train and evaluate the world model, and manage logs, optimizers, and checkpoints."""

    def __init__(
        self,
        agent: JEPAAgent,
        spec: ModelSpec,
        cfg: dict[str, Any],
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.agent = agent.to(device)
        self.spec = spec
        self.cfg = cfg
        self.binary_reward_thresholds: dict[str, float] = {}
        self.device = device
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        training = cfg["training"]
        world_params = [p for p in self.agent.world.parameters() if p.requires_grad]
        self.world_opt = torch.optim.AdamW(
            world_params,
            lr=float(training["world_lr"]),
            weight_decay=float(training["weight_decay"]),
            eps=1e-5,
        )
        self.history: list[dict[str, Any]] = []

    def _clip(self, parameters: Any) -> float:
        return float(
            nn.utils.clip_grad_norm_(
                parameters,
                max_norm=float(self.cfg["training"]["grad_clip"]),
            )
        )

    def train_world_epoch(self, loader: Any) -> dict[str, float]:
        self.agent.train()
        avg = MetricAverager()
        tau = float(self.cfg["training"].get("target_encoder_tau", 0.01))
        trainable = [p for p in self.agent.world.parameters() if p.requires_grad]
        for batch in loader:
            batch = batch.to(self.device)
            self.world_opt.zero_grad(set_to_none=True)
            loss, metrics = world_model_loss(
                self.agent,
                batch,
                self.cfg["world_loss"],
            )
            loss.backward()
            grad = self._clip(trainable)
            self.world_opt.step()
            self.agent.world.update_target_encoder(tau)
            metrics["world_grad_norm"] = grad
            avg.update(metrics)
        return avg.mean()

    @torch.no_grad()
    def evaluate_world(
        self,
        loader: Any,
        *,
        fit_thresholds: bool | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Evaluate world_model_loss (posterior_* prefix) plus prior_prediction_metrics."""
        training = self.cfg.get("training", {})
        calibrate = bool(training.get("calibrate_binary_thresholds", False))
        if fit_thresholds is None:
            fit_thresholds = calibrate
        rare_names = [
            name
            for name in rare_event_reward_names(self.cfg)
            if name in self.spec.reward_binary
        ]
        grid_size = int(training.get("binary_threshold_grid_size", 99))
        selection_weight = float(training.get("rare_event_selection_weight", 8.0))
        death_selection_weight = float(training.get("death_selection_weight", 0.0))

        self.agent.eval()
        avg = MetricAverager()
        confusion_at05: dict[str, torch.Tensor] = {}
        collected_prob: dict[str, list[torch.Tensor]] = defaultdict(list)
        collected_target: dict[str, list[torch.Tensor]] = defaultdict(list)

        for batch in loader:
            batch = batch.to(self.device)
            _, loss_metrics = world_model_loss(
                self.agent,
                batch,
                self.cfg["world_loss"],
            )
            metrics = {f"posterior_{k}": v for k, v in loss_metrics.items()}
            prior_metrics, batch_confusion, binary_pairs = prior_prediction_metrics(
                self.agent, batch
            )
            metrics.update(prior_metrics)
            avg.update(metrics)
            for key, value in batch_confusion.items():
                confusion_at05[key] = (
                    confusion_at05.get(key, value.new_zeros(())) + value
                )
            for name, (prob, target) in binary_pairs.items():
                collected_prob[name].append(prob)
                collected_target[name].append(target)

        out = avg.mean()
        metrics_at05 = finalize_binary_classification_metrics(
            confusion_at05, metric_prefix="prior_reward_at05"
        )
        out.update(metrics_at05)

        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, probs in collected_prob.items():
            if not probs:
                continue
            arrays[name] = (
                torch.cat(probs, dim=0).numpy(),
                torch.cat(collected_target[name], dim=0).numpy(),
            )
        for name, (prob, target) in arrays.items():
            auroc = safe_auroc(prob, target)
            if np.isfinite(auroc):
                out[f"prior_reward_auroc__{name}"] = float(auroc)
                out[f"prior_reward_at05_auroc__{name}"] = float(auroc)
            auprc = safe_auprc(prob, target)
            if np.isfinite(auprc):
                out[f"prior_reward_auprc__{name}"] = float(auprc)
                out[f"prior_reward_at05_auprc__{name}"] = float(auprc)

        used_thresholds: dict[str, float] = {}
        if fit_thresholds:
            for name in rare_names:
                if name not in arrays:
                    continue
                prob, target = arrays[name]
                threshold, _ = fit_balanced_accuracy_threshold(
                    prob, target, grid_size=grid_size
                )
                used_thresholds[name] = float(threshold)
            self.binary_reward_thresholds = dict(used_thresholds)
        elif thresholds is not None:
            used_thresholds = {
                name: float(thresholds[name])
                for name in rare_names
                if name in thresholds
            }
        elif calibrate and self.binary_reward_thresholds:
            used_thresholds = {
                name: float(self.binary_reward_thresholds[name])
                for name in rare_names
                if name in self.binary_reward_thresholds
            }
        else:
            self.binary_reward_thresholds = {}

        primary = finalize_binary_classification_metrics(
            confusion_at05, metric_prefix="prior_reward"
        )
        if used_thresholds:
            for name in rare_names:
                if name not in arrays:
                    continue
                prob, target = arrays[name]
                threshold = float(used_thresholds.get(name, 0.5))
                primary.update(
                    binary_classification_metrics_from_scores(
                        prob,
                        target,
                        threshold=threshold,
                        metric_prefix="prior_reward",
                        name=name,
                    )
                )
                out[f"prior_reward_threshold__{name}"] = threshold
        else:
            for name in rare_names:
                out[f"prior_reward_threshold__{name}"] = 0.5
        out.update(primary)

        latent_loss = float(out.get("posterior_jepa_loss", float("inf")))
        world_loss = float(out.get("posterior_world_loss", float("inf")))
        jepa_clinical_loss = float(
            out.get("posterior_jepa_clinical_loss", float("inf"))
        )
        out["latent_loss"] = latent_loss
        out["overall_loss"] = world_loss
        out["jepa_clinical_loss"] = jepa_clinical_loss
        w_jepa, w_clin = jepa_clinical_selection_weights(self.cfg)
        clin_mh = float(out.get("posterior_clinical_mh_loss", float("nan")))
        if math.isfinite(latent_loss) and math.isfinite(clin_mh):
            out["jepa_clinical_weighted"] = w_jepa * latent_loss + w_clin * clin_mh
        elif math.isfinite(jepa_clinical_loss):
            out["jepa_clinical_weighted"] = jepa_clinical_loss
        out["early_stopping_jepa_weight"] = w_jepa
        out["early_stopping_clinical_weight"] = w_clin
        death_auprc_key = "prior_reward_auprc__death_event"
        death_auprc = (
            float(out[death_auprc_key]) if death_auprc_key in out else float("nan")
        )
        if np.isfinite(death_auprc):
            out["death_event_auprc"] = death_auprc

        bal_keys = [
            f"prior_reward_balanced_accuracy__{name}"
            for name in rare_names
            if f"prior_reward_balanced_accuracy__{name}" in out
        ]
        if bal_keys:
            mean_bal = float(np.mean([out[key] for key in bal_keys]))
            out["rare_event_mean_balanced_accuracy"] = mean_bal
            death_key = "prior_reward_balanced_accuracy__death_event"
            death_bal = float(out[death_key]) if death_key in out else 0.0
            out["death_event_balanced_accuracy"] = death_bal
            out["rare_event_composite"] = (
                world_loss
                - selection_weight * mean_bal
                - death_selection_weight * death_bal
            )
        return out

    def record(self, phase: str, epoch: int, metrics: dict[str, float]) -> None:
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "epoch": epoch,
            **metrics,
        }
        self.history.append(row)
        printable = " ".join(
            f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, float)
        )
        print(f"[{phase} {epoch}] {printable}", flush=True)
        self.write_history()

    def write_history(self) -> None:
        if not self.history:
            return
        columns = sorted({key for row in self.history for key in row})
        path = self.output_dir / "training_history.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.history)

    def save_checkpoint(self, name: str, extra: dict[str, Any] | None = None) -> Path:
        path = self.output_dir / name
        payload = {
            "agent": self.agent.state_dict(),
            "world_optimizer": self.world_opt.state_dict(),
            "model_spec": self.spec.to_dict(),
            "config": self.cfg,
            "extra": extra or {},
        }
        torch.save(payload, path)
        return path

    def _checkpoint_extra(
        self,
        *,
        kind: str,
        epoch: int,
        metric_name: str,
        metric_value: float,
        val_metrics: dict[str, float],
        thresholds: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "checkpoint_kind": kind,
            "best_epoch": epoch,
            "best_metric_name": metric_name,
            "best_metric_value": metric_value,
            "binary_reward_thresholds": thresholds,
            "latent_loss": val_metrics.get("latent_loss"),
            "overall_loss": val_metrics.get("overall_loss"),
            "death_event_auprc": val_metrics.get("death_event_auprc"),
            "validation_metrics": {
                k: float(v)
                for k, v in val_metrics.items()
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            },
        }

    def fit(
        self,
        train_loader: Any,
        test_loader: Any,
        val_loader: Any | None = None,
    ) -> dict[str, float]:
        """Train the health world model, keep the 3 best validation checkpoints, then evaluate test with best_final."""
        training = self.cfg["training"]
        eval_every = max(int(training.get("world_eval_every", 1)), 1)
        patience = int(training.get("early_stopping_patience", 0))
        min_delta = float(training.get("early_stopping_min_delta", 0.0))
        # Early stop / best_final: w_jepa * L_JEPA + w_clin * L_clin
        # (no variance; probe-only BCE is excluded when probe_only_rewards is set).
        early_metric = str(
            training.get("early_stopping_metric", "jepa_clinical_weighted")
        )
        early_higher_better = bool(
            training.get(
                "early_stopping_higher_better",
                early_metric in {"death_event_auprc", "rare_event_mean_balanced_accuracy"},
            )
        )
        final_metric = str(
            training.get("best_final_metric", early_metric if not early_higher_better else "jepa_clinical_weighted")
        )
        sel_w_jepa, sel_w_clin = jepa_clinical_selection_weights(self.cfg)
        calibrate = bool(training.get("calibrate_binary_thresholds", False))
        selection_weight = float(training.get("rare_event_selection_weight", 8.0))
        death_selection_weight = float(training.get("death_selection_weight", 0.0))
        rare_names = rare_event_reward_names(self.cfg)
        world_epochs = int(training["world_epochs"])

        best_jepa = float("inf")
        best_jepa_epoch = 0
        best_clinical = float("-inf")
        best_clinical_epoch = 0
        best_final = float("inf")
        best_final_epoch = 0
        best_thresholds: dict[str, float] = {}
        early_best = float("-inf") if early_higher_better else float("inf")
        epochs_without_improve = 0
        early_stopped = False
        stopped_epoch = 0
        path_jepa = self.output_dir / "best_jepa.pt"
        path_clinical = self.output_dir / "best_clinical.pt"
        path_final = self.output_dir / "best_final.pt"

        if patience > 0 and val_loader is None:
            print(
                "Warning: early_stopping_patience>0 but no validation loader; "
                "early stopping is disabled for this run.",
                flush=True,
            )
        else:
            print(
                "Checkpoint selection: "
                "best_jepa.pt=min(latent_loss), "
                "best_clinical.pt=max(death_event_auprc), "
                f"best_final.pt=min({final_metric}); "
                f"early_stop={early_metric} "
                f"({'higher' if early_higher_better else 'lower'}-better), "
                f"patience={patience}, min_delta={min_delta}, "
                f"max_epochs={world_epochs}, rare_events={rare_names}",
                flush=True,
            )

        for epoch in range(1, world_epochs + 1):
            self.record("world", epoch, self.train_world_epoch(train_loader))

            if val_loader is None or epoch % eval_every != 0:
                continue

            val_metrics = self.evaluate_world(val_loader, fit_thresholds=calibrate)
            self.record("validation_world", epoch, val_metrics)

            if calibrate:
                thresholds = {
                    name: float(val_metrics[f"prior_reward_threshold__{name}"])
                    for name in rare_names
                    if f"prior_reward_threshold__{name}" in val_metrics
                }
            else:
                thresholds = {}
                self.binary_reward_thresholds = {}

            latent = float(val_metrics.get("latent_loss", float("inf")))
            death_auprc = float(val_metrics.get("death_event_auprc", float("nan")))
            overall = float(val_metrics.get("overall_loss", float("inf")))
            final_score = float(val_metrics.get(final_metric, float("inf")))
            jc_weighted = float(val_metrics.get("jepa_clinical_weighted", float("nan")))
            updated: list[str] = []

            if math.isfinite(latent) and latent < (best_jepa - min_delta):
                best_jepa = latent
                best_jepa_epoch = epoch
                self.save_checkpoint(
                    "best_jepa.pt",
                    extra=self._checkpoint_extra(
                        kind="best_jepa",
                        epoch=epoch,
                        metric_name="latent_loss",
                        metric_value=latent,
                        val_metrics=val_metrics,
                        thresholds=thresholds,
                    ),
                )
                updated.append(f"jepa={latent:.6f}")

            if math.isfinite(death_auprc) and death_auprc > (
                best_clinical + min_delta
            ):
                best_clinical = death_auprc
                best_clinical_epoch = epoch
                self.save_checkpoint(
                    "best_clinical.pt",
                    extra=self._checkpoint_extra(
                        kind="best_clinical",
                        epoch=epoch,
                        metric_name="death_event_auprc",
                        metric_value=death_auprc,
                        val_metrics=val_metrics,
                        thresholds=thresholds,
                    ),
                )
                updated.append(f"clinical_auprc={death_auprc:.6f}")

            if math.isfinite(final_score) and final_score < (best_final - min_delta):
                best_final = final_score
                best_final_epoch = epoch
                best_thresholds = dict(thresholds)
                if calibrate:
                    self.binary_reward_thresholds = dict(best_thresholds)
                self.save_checkpoint(
                    "best_final.pt",
                    extra=self._checkpoint_extra(
                        kind="best_final",
                        epoch=epoch,
                        metric_name=final_metric,
                        metric_value=final_score,
                        val_metrics=val_metrics,
                        thresholds=thresholds,
                    ),
                )
                updated.append(f"final={final_score:.6f}")

            early_score = float(val_metrics.get(early_metric, float("nan")))
            if early_higher_better:
                early_improved = math.isfinite(early_score) and early_score > (
                    early_best + min_delta
                )
            else:
                early_improved = math.isfinite(early_score) and early_score < (
                    early_best - min_delta
                )
            if early_improved:
                early_best = early_score
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            update_txt = ", ".join(updated) if updated else "no new best"
            death_txt = (
                f", death_auprc={death_auprc:.4f}"
                if math.isfinite(death_auprc)
                else ""
            )
            clinical_best_txt = (
                f"{best_clinical:.6f}" if math.isfinite(best_clinical) else "nan"
            )
            print(
                f"[validation {epoch}] latent={latent:.6f}"
                f"{death_txt}, overall_loss={overall:.6f}"
                + (
                    f", jepa_clinical_weighted={jc_weighted:.6f}"
                    if math.isfinite(jc_weighted)
                    else ""
                )
                + f" | best_jepa@{best_jepa_epoch}={best_jepa:.6f}, "
                f"best_clinical@{best_clinical_epoch}={clinical_best_txt}, "
                f"best_final@{best_final_epoch}={best_final:.6f} "
                f"| {update_txt}; "
                f"no_improve={epochs_without_improve}/{patience or 'off'}",
                flush=True,
            )
            if patience > 0 and epochs_without_improve >= patience:
                early_stopped = True
                stopped_epoch = epoch
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={patience}, best_final_epoch={best_final_epoch}, "
                    f"best_{final_metric}={best_final:.6f})",
                    flush=True,
                )
                break

        if val_loader is None:
            # No validation: dump last weights into all three names.
            self.save_checkpoint(
                "best_jepa.pt",
                extra={"checkpoint_kind": "best_jepa", "note": "no_validation"},
            )
            self.save_checkpoint(
                "best_clinical.pt",
                extra={"checkpoint_kind": "best_clinical", "note": "no_validation"},
            )
            self.save_checkpoint(
                "best_final.pt",
                extra={"checkpoint_kind": "best_final", "note": "no_validation"},
            )
        elif path_final.exists():
            checkpoint = torch.load(
                path_final, map_location=self.device, weights_only=False
            )
            self.agent.load_state_dict(checkpoint["agent"])
            if "world_optimizer" in checkpoint:
                self.world_opt.load_state_dict(checkpoint["world_optimizer"])
            extra = checkpoint.get("extra") or {}
            if calibrate:
                restored = extra.get("binary_reward_thresholds") or best_thresholds
                self.binary_reward_thresholds = {
                    str(k): float(v) for k, v in dict(restored).items()
                }
            else:
                self.binary_reward_thresholds = {}
            print(
                f"Restored best_final.pt from epoch {best_final_epoch} "
                f"({final_metric}={best_final:.6f})"
                + (" [early stopped]" if early_stopped else ""),
                flush=True,
            )
        else:
            print(
                "WARNING: best_final.pt missing after validation; "
                "evaluating last-epoch weights on test.",
                flush=True,
            )
            self.save_checkpoint(
                "best_final.pt",
                extra={
                    "checkpoint_kind": "best_final",
                    "note": "fallback_last_epoch",
                    "binary_reward_thresholds": self.binary_reward_thresholds,
                },
            )

        test_metrics = self.evaluate_world(
            test_loader,
            fit_thresholds=False,
            thresholds=(
                self.binary_reward_thresholds or None if calibrate else None
            ),
        )
        self.record("test_world_final", 1, test_metrics)
        # Refresh best_final payload with test metrics (weights unchanged).
        self.save_checkpoint(
            "best_final.pt",
            extra={
                "checkpoint_kind": "best_final",
                "best_epoch": best_final_epoch if val_loader is not None else None,
                "best_metric_name": final_metric,
                "best_metric_value": best_final if val_loader is not None else None,
                "best_jepa_epoch": best_jepa_epoch if val_loader is not None else None,
                "best_jepa_latent_loss": best_jepa if val_loader is not None else None,
                "best_clinical_epoch": (
                    best_clinical_epoch if val_loader is not None else None
                ),
                "best_clinical_death_auprc": (
                    best_clinical if val_loader is not None else None
                ),
                "binary_reward_thresholds": self.binary_reward_thresholds,
                "early_stopped": early_stopped,
                "stopped_epoch": stopped_epoch if early_stopped else world_epochs,
                "test_world_metrics": test_metrics,
            },
        )
        (self.output_dir / "test_world_metrics.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.binary_reward_thresholds:
            (self.output_dir / "binary_reward_thresholds.json").write_text(
                json.dumps(self.binary_reward_thresholds, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        selection = {
            "checkpoints": {
                "best_jepa.pt": {
                    "metric": "latent_loss",
                    "direction": "min",
                    "best_epoch": best_jepa_epoch,
                    "best_value": best_jepa if best_jepa_epoch else None,
                    "path": str(path_jepa) if path_jepa.exists() else None,
                },
                "best_clinical.pt": {
                    "metric": "death_event_auprc",
                    "direction": "max",
                    "best_epoch": best_clinical_epoch,
                    "best_value": (
                        best_clinical if best_clinical_epoch else None
                    ),
                    "path": str(path_clinical) if path_clinical.exists() else None,
                },
                "best_final.pt": {
                    "metric": final_metric,
                    "direction": "min",
                    "best_epoch": best_final_epoch,
                    "best_value": best_final if best_final_epoch else None,
                    "path": str(path_final) if path_final.exists() else None,
                },
            },
            "early_stopped": early_stopped,
            "stopped_epoch": stopped_epoch if early_stopped else world_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "early_stopping_metric": early_metric,
            "best_final_metric": final_metric,
            "early_stopping_jepa_weight": sel_w_jepa,
            "early_stopping_clinical_weight": sel_w_clin,
            "calibrate_binary_thresholds": calibrate,
            "rare_event_selection_weight": selection_weight,
            "death_selection_weight": death_selection_weight,
            "rare_event_reward_names": rare_names,
            "binary_reward_thresholds": self.binary_reward_thresholds,
            "jepa_loss": str(self.cfg["world_loss"].get("jepa_loss", "mse")),
            "jepa_loss_scale": float(
                self.cfg["world_loss"].get("jepa_loss_scale", 1.0)
            ),
        }
        (self.output_dir / "best_checkpoint_selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return test_metrics

    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.agent.load_state_dict(checkpoint["agent"])
        if "world_optimizer" in checkpoint:
            self.world_opt.load_state_dict(checkpoint["world_optimizer"])
        return checkpoint

    def train_clinical_finetune_epoch(
        self,
        loader: Any,
        *,
        finetune_loss_cfg: dict[str, Any],
        trainable_binary_groups: Sequence[str],
    ) -> dict[str, float]:
        self.agent.train()
        avg = MetricAverager()
        trainable = [p for p in self.agent.world.parameters() if p.requires_grad]
        for batch in loader:
            batch = batch.to(self.device)
            self.world_opt.zero_grad(set_to_none=True)
            loss, metrics = clinical_finetune_loss(
                self.agent,
                batch,
                finetune_loss_cfg,
                trainable_binary_groups=trainable_binary_groups,
            )
            loss.backward()
            grad = self._clip(trainable)
            self.world_opt.step()
            metrics["clinical_finetune_grad_norm"] = grad
            avg.update(metrics)
        return avg.mean()

    def fit_clinical_finetune(
        self,
        train_loader: Any,
        test_loader: Any,
        val_loader: Any | None = None,
    ) -> dict[str, float]:
        """Phase 2: frozen backbone, train selected clinical binary heads only."""
        cf_cfg = dict(self.cfg.get("clinical_finetune") or {})
        if not cf_cfg.get("enabled", False):
            raise RuntimeError("clinical_finetune.enabled is false")

        trainable_groups = [
            str(x) for x in (cf_cfg.get("trainable_binary_groups") or ["death"])
        ]
        finetune_loss_cfg = resolve_clinical_finetune_loss_cfg(self.cfg)
        epochs = int(cf_cfg.get("epochs", 15))
        lr = float(cf_cfg.get("lr", 1e-4))
        weight_decay = float(
            cf_cfg.get("weight_decay", self.cfg["training"].get("weight_decay", 0.0))
        )
        eval_every = max(int(cf_cfg.get("eval_every", 1)), 1)
        patience = int(cf_cfg.get("early_stopping_patience", 4))
        min_delta = float(cf_cfg.get("early_stopping_min_delta", 0.0))
        early_metric = str(cf_cfg.get("early_stopping_metric", "death_event_auprc"))
        early_higher_better = bool(
            cf_cfg.get(
                "early_stopping_higher_better",
                early_metric.endswith("_auprc")
                or early_metric.endswith("_auroc")
                or "accuracy" in early_metric,
            )
        )
        pw_max = float(cf_cfg.get("binary_reward_pos_weight_max", 0.0) or 0.0)
        if pw_max > 0:
            apply_finetune_pos_weight_cap(
                self.agent,
                self.spec,
                max_weight=pw_max,
                reward_names=tuple(
                    cf_cfg.get("pos_weight_reward_names") or ["death_event"]
                ),
            )

        trainable = configure_clinical_finetune(
            self.agent,
            trainable_binary_groups=trainable_groups,
        )
        if not trainable:
            raise RuntimeError(
                f"No trainable parameters for clinical finetune groups: {trainable_groups}"
            )
        self.world_opt = torch.optim.AdamW(
            trainable,
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-5,
        )

        path_finetune = self.output_dir / "best_death_finetune.pt"
        best_score = float("-inf") if early_higher_better else float("inf")
        best_epoch = 0
        epochs_without_improve = 0
        early_stopped = False
        stopped_epoch = 0

        print(
            "Clinical finetune (Phase 2): "
            f"groups={trainable_groups}, epochs={epochs}, lr={lr}, "
            f"early_stop={early_metric} "
            f"({'higher' if early_higher_better else 'lower'}-better), "
            f"patience={patience}, pos_weight_max={pw_max or 'unchanged'}",
            flush=True,
        )

        for epoch in range(1, epochs + 1):
            self.record(
                "clinical_finetune",
                epoch,
                self.train_clinical_finetune_epoch(
                    train_loader,
                    finetune_loss_cfg=finetune_loss_cfg,
                    trainable_binary_groups=trainable_groups,
                ),
            )

            if val_loader is None or epoch % eval_every != 0:
                continue

            val_metrics = self.evaluate_world(val_loader, fit_thresholds=False)
            self.record("clinical_finetune_val", epoch, val_metrics)
            early_score = float(val_metrics.get(early_metric, float("nan")))
            death_auprc = float(val_metrics.get("death_event_auprc", float("nan")))
            updated = False

            if early_higher_better:
                improved = math.isfinite(early_score) and early_score > (
                    best_score + min_delta
                )
            else:
                improved = math.isfinite(early_score) and early_score < (
                    best_score - min_delta
                )
            if improved:
                best_score = early_score
                best_epoch = epoch
                epochs_without_improve = 0
                updated = True
                self.save_checkpoint(
                    "best_death_finetune.pt",
                    extra={
                        "checkpoint_kind": "best_death_finetune",
                        "best_epoch": epoch,
                        "best_metric_name": early_metric,
                        "best_metric_value": early_score,
                        "death_event_auprc": death_auprc,
                        "validation_metrics": {
                            k: float(v)
                            for k, v in val_metrics.items()
                            if isinstance(v, (int, float)) and math.isfinite(float(v))
                        },
                        "clinical_finetune_groups": trainable_groups,
                    },
                )
            else:
                epochs_without_improve += 1

            death_txt = (
                f", death_auprc={death_auprc:.4f}"
                if math.isfinite(death_auprc)
                else ""
            )
            print(
                f"[clinical_finetune val {epoch}] {early_metric}={early_score:.6f}"
                f"{death_txt} | best@{best_epoch}={best_score:.6f} | "
                f"{'saved' if updated else 'no new best'}; "
                f"no_improve={epochs_without_improve}/{patience or 'off'}",
                flush=True,
            )
            if patience > 0 and epochs_without_improve >= patience:
                early_stopped = True
                stopped_epoch = epoch
                print(
                    f"Clinical finetune early stopping at epoch {epoch} "
                    f"(patience={patience}, best_epoch={best_epoch}, "
                    f"best_{early_metric}={best_score:.6f})",
                    flush=True,
                )
                break

        if val_loader is None:
            self.save_checkpoint(
                "best_death_finetune.pt",
                extra={
                    "checkpoint_kind": "best_death_finetune",
                    "note": "no_validation",
                    "clinical_finetune_groups": trainable_groups,
                },
            )
        elif path_finetune.exists():
            checkpoint = torch.load(
                path_finetune, map_location=self.device, weights_only=False
            )
            self.agent.load_state_dict(checkpoint["agent"])
            if "world_optimizer" in checkpoint:
                self.world_opt.load_state_dict(checkpoint["world_optimizer"])
            print(
                f"Restored best_death_finetune.pt from epoch {best_epoch} "
                f"({early_metric}={best_score:.6f})"
                + (" [early stopped]" if early_stopped else ""),
                flush=True,
            )
        else:
            print(
                "WARNING: best_death_finetune.pt missing; evaluating last-epoch weights.",
                flush=True,
            )
            self.save_checkpoint(
                "best_death_finetune.pt",
                extra={
                    "checkpoint_kind": "best_death_finetune",
                    "note": "fallback_last_epoch",
                    "clinical_finetune_groups": trainable_groups,
                },
            )

        test_metrics = self.evaluate_world(test_loader, fit_thresholds=False)
        self.record("test_clinical_finetune", 1, test_metrics)
        self.save_checkpoint(
            "best_death_finetune.pt",
            extra={
                "checkpoint_kind": "best_death_finetune",
                "best_epoch": best_epoch if val_loader is not None else None,
                "best_metric_name": early_metric,
                "best_metric_value": best_score if val_loader is not None else None,
                "death_event_auprc": test_metrics.get("death_event_auprc"),
                "early_stopped": early_stopped,
                "stopped_epoch": stopped_epoch if early_stopped else epochs,
                "clinical_finetune_groups": trainable_groups,
                "test_world_metrics": test_metrics,
            },
        )
        (self.output_dir / "test_clinical_finetune_metrics.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        selection = {
            "checkpoint": "best_death_finetune.pt",
            "metric": early_metric,
            "direction": "max" if early_higher_better else "min",
            "best_epoch": best_epoch,
            "best_value": best_score if best_epoch else None,
            "early_stopped": early_stopped,
            "stopped_epoch": stopped_epoch if early_stopped else epochs,
            "patience": patience,
            "min_delta": min_delta,
            "trainable_binary_groups": trainable_groups,
            "finetune_loss_cfg": finetune_loss_cfg,
        }
        (self.output_dir / "clinical_finetune_selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return test_metrics
