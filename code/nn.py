"""Health world model network.

Architecture::

    Dynamic s_1:t ──► StateEncoder ──► TemporalTransformer ──► z_t
    Static  c     ──► StaticEncoder ──► c_emb   (condition only)
    Predictor(z_t, c_emb, a_t, Δt) ──► z_future
    Target: EMA(StateEncoder+Temporal)(s_1:t+1) ──► target_z   (no static)

ClinicalHeads(z_future) → grouped binary heads + optional continuous trunk/heads.

Binary groups (independent MLP trunks; only names present in ModelSpec are built)::

    death      : death_event
    worsening  : adl_worsening, iadl_worsening

``ModelSpec.probe_only_rewards``: the group MLP still predicts them for eval,
but ``z`` is detached so the encoder and predictor are not trained on those labels.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .specs import (
    CONTINUOUS_REWARD_TO_STATE,
    CONTINUOUS_REWARD_TO_STATE_FALLBACKS,
    ModelSpec,
)

# Preferred membership for grouped binary ClinicalHeads (order within each group).
BINARY_HEAD_GROUP_MEMBERS: dict[str, tuple[str, ...]] = {
    "death": ("death_event",),
    "worsening": (
        "adl_worsening",
        "iadl_worsening",
    ),
}
BINARY_HEAD_GROUP_ORDER: tuple[str, ...] = (
    "death",
    "worsening",
    "other",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def cosine_distance(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1, eps=eps)
    target_n = F.normalize(target, dim=-1, eps=eps)
    return 1.0 - (pred_n * target_n).sum(dim=-1)


def embedding_variance(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if x.shape[0] < 2:
        return x.new_zeros(())
    return x.std(dim=0, unbiased=False).mean().clamp_min(eps)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        layers: int = 2,
        activation: type[nn.Module] = nn.SiLU,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        dim = input_dim
        for _ in range(layers):
            modules.append(nn.Linear(dim, hidden_dim))
            if layer_norm:
                modules.append(nn.LayerNorm(hidden_dim))
            modules.append(activation())
            dim = hidden_dim
        modules.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 1) Dynamic state encoder  —  s_t → w_t  (target latent pathway)
# ---------------------------------------------------------------------------


class StateEncoder(nn.Module):
    """Encode the dynamic state only (no static context). Shared by the online encoder and the EMA target."""

    def __init__(
        self,
        spec: ModelSpec,
        embed_dim: int,
        hidden_dim: int,
        cat_embed_dim: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(feature.classes + 1, cat_embed_dim)
                for feature in spec.state_categorical
            ]
        )
        input_dim = 2 * len(spec.state_continuous)
        input_dim += cat_embed_dim * len(spec.state_categorical)
        input_dim += len(spec.state_categorical)
        if input_dim == 0:
            raise ValueError("StateEncoder has no inputs")
        self.mlp = MLP(input_dim, embed_dim, hidden_dim, layers=2)

    def forward(
        self,
        cont: torch.Tensor,
        cont_mask: torch.Tensor,
        cat: torch.Tensor,
        cat_mask: torch.Tensor,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if cont.shape[-1]:
            parts.extend([cont * cont_mask, cont_mask])
        for index, embedding in enumerate(self.cat_embeddings):
            parts.append(embedding(cat[..., index].long()))
        if cat_mask.shape[-1]:
            parts.append(cat_mask)
        return self.mlp(torch.cat(parts, dim=-1))


# Alias: wave token = dynamic state embedding before temporal encoder.
WaveEncoder = StateEncoder


# ---------------------------------------------------------------------------
# 1b) Static condition encoder  —  c → c_emb  (not part of target latent)
# ---------------------------------------------------------------------------


class StaticEncoder(nn.Module):
    """Person-level static-context encoder. It enters the predictor only, not the target latent."""

    def __init__(
        self,
        spec: ModelSpec,
        embed_dim: int,
        hidden_dim: int,
        cat_embed_dim: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        if not spec.has_static_context:
            raise ValueError("StaticEncoder requires static_context variables")
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(feature.classes + 1, cat_embed_dim)
                for feature in spec.static_categorical
            ]
        )
        input_dim = 2 * len(spec.static_continuous)
        input_dim += cat_embed_dim * len(spec.static_categorical)
        input_dim += len(spec.static_categorical)
        if input_dim == 0:
            raise ValueError("StaticEncoder has no inputs")
        self.mlp = MLP(input_dim, embed_dim, hidden_dim, layers=2)

    def forward(
        self,
        cont: torch.Tensor,
        cont_mask: torch.Tensor,
        cat: torch.Tensor,
        cat_mask: torch.Tensor,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if cont.shape[-1]:
            parts.extend([cont * cont_mask, cont_mask])
        for index, embedding in enumerate(self.cat_embeddings):
            parts.append(embedding(cat[..., index].long()))
        if cat_mask.shape[-1]:
            parts.append(cat_mask)
        return self.mlp(torch.cat(parts, dim=-1))


# ---------------------------------------------------------------------------
# 2) Temporal Transformer  —  w_1…w_t → z_t
# ---------------------------------------------------------------------------


class TemporalTransformerEncoder(nn.Module):
    """Causal transformer: position t attends only to Wave 1 through Wave t."""

    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_dim: int | None = None,
        max_seq_len: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.max_seq_len = int(max_seq_len)
        ff = int(ff_dim if ff_dim is not None else 4 * embed_dim)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(n_heads),
            dim_feedforward=ff,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.out_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch_size, steps, _ = tokens.shape
        if steps > self.max_seq_len:
            raise ValueError(
                f"sequence length {steps} exceeds max_seq_len={self.max_seq_len}"
            )
        positions = torch.arange(steps, device=tokens.device)
        x = tokens + self.pos_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        causal = torch.triu(
            torch.ones(steps, steps, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        pad = valid <= 0.5
        all_pad = pad.all(dim=-1)
        if bool(all_pad.any()):
            pad = pad.clone()
            pad[all_pad, 0] = False
        encoded = self.encoder(x, mask=causal, src_key_padding_mask=pad)
        return self.out_norm(encoded)


# ---------------------------------------------------------------------------
# 3) Action encoder + Predictor  —  (z_t, c, a, Δt) → z_future
# ---------------------------------------------------------------------------


class ActionEncoder(nn.Module):
    """Embed mixed continuous and categorical actions."""

    def __init__(
        self,
        spec: ModelSpec,
        embed_dim: int,
        hidden_dim: int,
        cat_embed_dim: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(feature.classes + 1, cat_embed_dim)
                for feature in spec.action_categorical
            ]
        )
        input_dim = 2 * len(spec.action_continuous)
        input_dim += cat_embed_dim * len(spec.action_categorical)
        if input_dim == 0:
            raise ValueError("ActionEncoder has no inputs")
        self.mlp = MLP(input_dim, embed_dim, hidden_dim, layers=2)

    def forward(
        self,
        cont: torch.Tensor,
        cont_mask: torch.Tensor,
        cat: torch.Tensor,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if cont.shape[-1]:
            parts.extend([cont * cont_mask, cont_mask])
        for index, embedding in enumerate(self.cat_embeddings):
            parts.append(embedding(cat[..., index].long()))
        return self.mlp(torch.cat(parts, dim=-1))

    observed = forward


class ConditionedJEPAPredictor(nn.Module):
    """Predictor: (z_t, c_emb, a_emb, Δt) → z_future."""

    def __init__(
        self,
        embed_dim: int,
        static_embed_dim: int,
        action_embed_dim: int,
        hidden_dim: int,
        layers: int = 3,
    ) -> None:
        super().__init__()
        self.net = MLP(
            embed_dim + static_embed_dim + action_embed_dim + 1,
            embed_dim,
            hidden_dim,
            layers=layers,
        )

    def forward(
        self,
        z_t: torch.Tensor,
        static_embed: torch.Tensor,
        action_embed: torch.Tensor,
        delta_t_norm: torch.Tensor,
    ) -> torch.Tensor:
        if delta_t_norm.ndim == 1:
            delta_t_norm = delta_t_norm.unsqueeze(-1)
        return self.net(
            torch.cat([z_t, static_embed, action_embed, delta_t_norm], dim=-1)
        )


# Backward-compatible names.
ActionConditionedJEPAPredictor = ConditionedJEPAPredictor
JEPAPredictor = ConditionedJEPAPredictor


# ---------------------------------------------------------------------------
# 4) Clinical projection heads  —  z_future → events / levels
# ---------------------------------------------------------------------------


class ContinuousStateHeads(nn.Module):
    """Shared continuous trunk + per-variable scalar heads.

    ::

        z_future → Linear→LN→SiLU → Linear→LN→SiLU → h_state(64)
                 → {name: Linear(64 → 1|2)} for each continuous reward
    """

    def __init__(
        self,
        feature_dim: int,
        names: Sequence[str],
        *,
        trunk_dims: Sequence[int] = (128, 64),
        loss_kind: str = "huber",
        huber_delta: float = 1.0,
        log_var_clamp: tuple[float, float] = (-8.0, 8.0),
    ) -> None:
        super().__init__()
        self.names = [str(x) for x in names]
        self.loss_kind = str(loss_kind).lower().strip()
        if self.loss_kind not in {"huber", "gaussian_nll"}:
            raise ValueError(
                f"Unsupported continuous_loss={loss_kind!r}; "
                "expected 'huber' or 'gaussian_nll'"
            )
        self.huber_delta = float(huber_delta)
        self.log_var_min = float(log_var_clamp[0])
        self.log_var_max = float(log_var_clamp[1])
        dims = [int(feature_dim), *[int(x) for x in trunk_dims]]
        if len(dims) < 2:
            raise ValueError("continuous trunk_dims must yield at least one projection")
        modules: list[nn.Module] = []
        for din, dout in zip(dims[:-1], dims[1:]):
            modules.extend([nn.Linear(din, dout), nn.LayerNorm(dout), nn.SiLU()])
        self.trunk = nn.Sequential(*modules)
        self.state_dim = dims[-1]
        out_dim = 2 if self.loss_kind == "gaussian_nll" else 1
        self.heads = nn.ModuleDict(
            {name: nn.Linear(self.state_dim, out_dim) for name in self.names}
        )
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def encode(self, feature: torch.Tensor) -> torch.Tensor:
        return self.trunk(feature)

    def forward_raw(self, feature: torch.Tensor) -> torch.Tensor:
        """Stacked head outputs ``[..., n_cont, out_dim]`` in ``self.names`` order."""
        h = self.encode(feature)
        return torch.stack([self.heads[name](h) for name in self.names], dim=-2)

    def mean(self, feature: torch.Tensor) -> torch.Tensor:
        raw = self.forward_raw(feature)
        return raw[..., 0]

    def loss(
        self,
        feature: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        current: torch.Tensor | None = None,
        current_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(L_state, L_delta)`` with residual decode when ``current`` is set.

        Head output is ``delta``; ``s_hat = s_t + delta`` when current levels exist::

            L_state = Huber(s_hat, s_{t+1})
            L_delta = Huber(delta, s_{t+1} - s_t)

        Under Huber these match on the residual mask; residual zero-init (persistence)
        is what reduces overly flat absolute-mean predictions.
        """
        raw = self.forward_raw(feature)
        delta = raw[..., 0]
        if current is None:
            pred_level = delta
            has_cur = None
        else:
            has_cur = (
                mask.new_ones(mask.shape, dtype=torch.bool)
                if current_mask is None
                else (current_mask > 0.5)
            )
            pred_level = torch.where(has_cur, current + delta, delta)
        if self.loss_kind == "huber":
            per = F.smooth_l1_loss(
                pred_level, target, reduction="none", beta=self.huber_delta
            )
        else:
            log_var = raw[..., 1].clamp(self.log_var_min, self.log_var_max)
            per = 0.5 * (
                log_var + (target - pred_level).square() * torch.exp(-log_var)
            )
        level_loss = masked_mean(per, mask)
        if current is None or has_cur is None:
            return level_loss, feature.new_zeros(())
        cmask = mask * has_cur.float()
        per_delta = F.smooth_l1_loss(
            delta,
            target - current,
            reduction="none",
            beta=self.huber_delta,
        )
        delta_loss = masked_mean(per_delta, cmask)
        return level_loss, delta_loss

    def decode_level(
        self,
        feature: torch.Tensor,
        current: torch.Tensor | None = None,
        current_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode next-wave levels; residual ``s_t + delta`` when current given."""
        delta = self.mean(feature)
        if current is None:
            return delta
        has_cur = (
            current.new_ones(current.shape, dtype=torch.bool)
            if current_mask is None
            else (current_mask > 0.5)
        )
        return torch.where(has_cur, current + delta, delta)



def _partition_binary_reward_groups(
    reward_binary: Sequence[str],
) -> dict[str, list[str]]:
    """Split ``reward_binary`` into death / worsening / other."""
    assigned: set[str] = set()
    groups: dict[str, list[str]] = {name: [] for name in BINARY_HEAD_GROUP_ORDER}
    present = list(reward_binary)
    for group_name in ("death", "worsening"):
        for name in BINARY_HEAD_GROUP_MEMBERS[group_name]:
            if name in present and name not in assigned:
                groups[group_name].append(name)
                assigned.add(name)
    for name in present:
        if name not in assigned:
            groups["other"].append(name)
            assigned.add(name)
    return {k: v for k, v in groups.items() if v}


class ClinicalHeads(nn.Module):
    """Clinical projection heads: grouped binary MLPs plus an optional continuous residual head.

    Binary groups (independent 2-layer MLPs; only names in ``ModelSpec``)::

        death / worsening / other

    Continuous: shared trunk + per-variable Linear when ``reward_continuous``.
    """

    def __init__(
        self,
        spec: ModelSpec,
        feature_dim: int,
        hidden_dim: int,
        *,
        continuous_trunk_dims: Sequence[int] = (128, 64),
        continuous_loss: str = "huber",
        continuous_huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.probe_only_rewards = set(spec.probe_only_rewards)
        self.binary_group_names = _partition_binary_reward_groups(spec.reward_binary)
        self.binary_groups = nn.ModuleDict()
        for group_name, names in self.binary_group_names.items():
            mlp = MLP(feature_dim, len(names), hidden_dim, layers=2)
            final = mlp.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
            self.binary_groups[group_name] = mlp
        # Legacy: callers used ``if self.binary is not None``. Prefer ``has_binary``.
        self.binary = True if self.binary_groups else None
        if self.has_binary:
            weights = list(spec.reward_binary_pos_weight)
            if len(weights) != len(spec.reward_binary):
                weights = [1.0] * len(spec.reward_binary)
            self.register_buffer(
                "binary_pos_weight",
                torch.tensor(weights, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.register_buffer(
                "binary_pos_weight",
                torch.empty(0, dtype=torch.float32),
                persistent=True,
            )
        self.continuous = (
            ContinuousStateHeads(
                feature_dim,
                spec.reward_continuous,
                trunk_dims=continuous_trunk_dims,
                loss_kind=continuous_loss,
                huber_delta=continuous_huber_delta,
            )
            if spec.reward_continuous
            else None
        )

    @property
    def has_binary(self) -> bool:
        return len(self.binary_groups) > 0

    def _group_is_probe(self, names: Sequence[str]) -> bool:
        return bool(self.probe_only_rewards) and any(
            name in self.probe_only_rewards for name in names
        )

    def binary_logits(self, feature: torch.Tensor) -> torch.Tensor:
        """Concatenate group logits in ``spec.reward_binary`` column order.

        Probe-only groups receive ``feature.detach()`` so those BCE terms cannot
        train the encoder / predictor / JEPA representation.
        """
        if not self.has_binary:
            raise RuntimeError("ClinicalHeads has no binary reward groups")
        name_to_logit: dict[str, torch.Tensor] = {}
        for group_name, names in self.binary_group_names.items():
            feat = feature.detach() if self._group_is_probe(names) else feature
            logits = self.binary_groups[group_name](feat)
            for i, name in enumerate(names):
                name_to_logit[name] = logits[..., i]
        return torch.stack(
            [name_to_logit[name] for name in self.spec.reward_binary],
            dim=-1,
        )

    def _current_levels_for_rewards(
        self,
        state_cont: torch.Tensor,
        state_cont_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map ``state_cont`` → current levels aligned with ``reward_continuous``."""
        n = len(self.spec.reward_continuous)
        current = state_cont.new_zeros(state_cont.shape[0], n)
        cmask = state_cont.new_zeros(state_cont.shape[0], n)
        name_to_idx = {name: i for i, name in enumerate(self.spec.state_continuous)}
        for j, reward_name in enumerate(self.spec.reward_continuous):
            state_name = CONTINUOUS_REWARD_TO_STATE.get(reward_name)
            if state_name not in name_to_idx:
                for alt in CONTINUOUS_REWARD_TO_STATE_FALLBACKS.get(reward_name, ()):
                    if alt in name_to_idx:
                        state_name = alt
                        break
            if state_name not in name_to_idx:
                continue
            i = name_to_idx[state_name]
            current[:, j] = state_cont[:, i]
            cmask[:, j] = state_cont_mask[:, i]
        return current, cmask

    def loss(
        self,
        feature: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        valid: torch.Tensor,
        *,
        binary_event_scales: Sequence[float] | torch.Tensor | None = None,
        state_cont: torch.Tensor | None = None,
        state_cont_mask: torch.Tensor | None = None,
        current_levels: torch.Tensor | None = None,
        current_levels_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(binary, continuous_level, continuous_delta, metrics)``."""
        binary_loss = feature.new_zeros(())
        continuous_level_loss = feature.new_zeros(())
        continuous_delta_loss = feature.new_zeros(())
        metrics: dict[str, torch.Tensor] = {}
        valid = valid[..., None]
        b = len(self.spec.reward_binary)
        if self.has_binary:
            logits = self.binary_logits(feature)
            pos_weight = self.binary_pos_weight
            if pos_weight.numel() != b:
                pos_weight = None
            loss_all = F.binary_cross_entropy_with_logits(
                logits,
                target[..., :b],
                reduction="none",
                pos_weight=pos_weight,
            )
            if binary_event_scales is not None:
                if torch.is_tensor(binary_event_scales):
                    scales = binary_event_scales.to(
                        device=loss_all.device, dtype=loss_all.dtype
                    )
                else:
                    scales = torch.tensor(
                        list(binary_event_scales),
                        device=loss_all.device,
                        dtype=loss_all.dtype,
                    )
                if scales.numel() != b:
                    raise ValueError(
                        f"binary_event_scales length {scales.numel()} != n_binary {b}"
                    )
                while scales.ndim < loss_all.ndim:
                    scales = scales.unsqueeze(0)
                loss_all = loss_all * scales
            binary_mask = mask[..., :b] * valid
            probe_col = torch.zeros(b, device=loss_all.device, dtype=loss_all.dtype)
            if self.probe_only_rewards:
                for i, name in enumerate(self.spec.reward_binary):
                    if name in self.probe_only_rewards:
                        probe_col[i] = 1.0
            while probe_col.ndim < loss_all.ndim:
                probe_col = probe_col.unsqueeze(0)
            repr_mask = binary_mask * (1.0 - probe_col)
            probe_mask = binary_mask * probe_col
            binary_repr_loss = masked_mean(loss_all, repr_mask)
            binary_probe_loss = masked_mean(loss_all, probe_mask)
            # Representation BCE only; probe BCE is returned in metrics for a
            # separate backward term that does not enter early-stopping world_loss.
            binary_loss = binary_repr_loss
            metrics["reward_binary_bce"] = binary_repr_loss.detach()
            metrics["reward_binary_bce_probe"] = binary_probe_loss.detach()
            metrics["_probe_binary_loss"] = binary_probe_loss
            # Per-group BCE (same pos_weight / scales, masked within group columns).
            name_to_idx = {n: i for i, n in enumerate(self.spec.reward_binary)}
            for group_name, names in self.binary_group_names.items():
                idxs = [name_to_idx[n] for n in names]
                g_loss = loss_all[..., idxs]
                g_mask = binary_mask[..., idxs]
                metrics[f"reward_binary_bce__{group_name}"] = masked_mean(
                    g_loss, g_mask
                ).detach()
        if self.continuous is not None:
            current = current_mask = None
            if current_levels is not None:
                current = current_levels
                if current_levels_mask is None:
                    current_mask = current.new_ones(current.shape)
                else:
                    current_mask = current_levels_mask
                current_mask = current_mask * valid
            elif state_cont is not None and state_cont_mask is not None:
                current, current_mask = self._current_levels_for_rewards(
                    state_cont, state_cont_mask
                )
                current_mask = current_mask * valid
            continuous_level_loss, continuous_delta_loss = self.continuous.loss(
                feature,
                target[..., b:],
                mask[..., b:] * valid,
                current=current,
                current_mask=current_mask,
            )
            continuous_loss = continuous_level_loss + continuous_delta_loss
            metrics["reward_cont_level"] = continuous_level_loss.detach()
            metrics["reward_cont_delta"] = continuous_delta_loss.detach()
            metrics["reward_cont_regression"] = continuous_loss.detach()
            metrics[f"reward_cont_{self.continuous.loss_kind}"] = continuous_level_loss.detach()
        return binary_loss, continuous_level_loss, continuous_delta_loss, metrics

    def mean_dict(
        self,
        feature: torch.Tensor,
        *,
        state_cont: torch.Tensor | None = None,
        state_cont_mask: torch.Tensor | None = None,
        current_levels: torch.Tensor | None = None,
        current_levels_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode clinical means.

        Continuous heads predict residual ``delta``; pass ``state_cont`` (or
        pre-aligned ``current_levels``) so levels are ``s_t + delta``.
        """
        result: dict[str, torch.Tensor] = {}
        if self.has_binary:
            values = torch.sigmoid(self.binary_logits(feature))
            for idx, name in enumerate(self.spec.reward_binary):
                result[name] = values[..., idx]
        if self.continuous is not None:
            current = current_mask = None
            if current_levels is not None:
                current = current_levels
                current_mask = current_levels_mask
            elif state_cont is not None and state_cont_mask is not None:
                current, current_mask = self._current_levels_for_rewards(
                    state_cont, state_cont_mask
                )
            values = self.continuous.decode_level(
                feature, current=current, current_mask=current_mask
            )
            for idx, name in enumerate(self.spec.reward_continuous):
                result[name] = values[..., idx]
        return result



# Aliases for older trainer / eval naming.
ClinicalProjectionHeads = ClinicalHeads
RewardHead = ClinicalHeads


# ---------------------------------------------------------------------------
# World model + agent
# ---------------------------------------------------------------------------


class WorldModel(nn.Module):
    """JEPA: z_t = Temporal(s_1:t); target_z = EMA Temporal(s_1:t+1); static context is a condition only."""

    def __init__(self, spec: ModelSpec, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.spec = spec
        embed_dim = int(cfg["obs_embed_dim"])
        action_embed_dim = int(cfg["action_embed_dim"])
        static_embed_dim = int(cfg.get("static_embed_dim", embed_dim))
        hidden = int(cfg["hidden_dim"])
        cat_embed = int(cfg["categorical_embed_dim"])
        self.embed_dim = embed_dim
        self.static_embed_dim = static_embed_dim
        self.feature_dim = embed_dim
        self.has_static = bool(spec.has_static_context)
        temporal_kwargs = dict(
            embed_dim=embed_dim,
            n_layers=int(cfg.get("transformer_layers", 4)),
            n_heads=int(cfg.get("transformer_heads", 8)),
            ff_dim=int(cfg.get("transformer_ff_dim", 4 * embed_dim)),
            max_seq_len=int(cfg.get("max_seq_len", 16)),
            dropout=float(cfg.get("transformer_dropout", 0.1)),
        )

        # Online dynamic tower.
        self.state_encoder = StateEncoder(spec, embed_dim, hidden, cat_embed)
        self.temporal_encoder = TemporalTransformerEncoder(**temporal_kwargs)
        self.wave_encoder = self.state_encoder

        # EMA target tower (same dynamic pathway; no static).
        self.target_state_encoder = StateEncoder(spec, embed_dim, hidden, cat_embed)
        self.target_temporal_encoder = TemporalTransformerEncoder(**temporal_kwargs)
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())
        self.target_temporal_encoder.load_state_dict(self.temporal_encoder.state_dict())
        for module in (self.target_state_encoder, self.target_temporal_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.target_wave_encoder = self.target_state_encoder

        self.static_encoder = (
            StaticEncoder(spec, static_embed_dim, hidden, cat_embed)
            if self.has_static
            else None
        )
        self.action_encoder = ActionEncoder(spec, action_embed_dim, hidden, cat_embed)
        self.predictor = ConditionedJEPAPredictor(
            embed_dim=embed_dim,
            static_embed_dim=static_embed_dim if self.has_static else 0,
            action_embed_dim=action_embed_dim,
            hidden_dim=hidden,
            layers=int(cfg.get("predictor_layers", 3)),
        )
        trunk_dims = cfg.get("continuous_trunk_dims", [128, 64])
        self.clinical_heads = ClinicalHeads(
            spec,
            self.feature_dim,
            hidden,
            continuous_trunk_dims=[int(x) for x in trunk_dims],
            continuous_loss=str(cfg.get("continuous_loss", "huber")),
            continuous_huber_delta=float(cfg.get("continuous_huber_delta", 1.0)),
        )
        self.reward_head = self.clinical_heads

    def encode_wave(
        self,
        cont: torch.Tensor,
        cont_mask: torch.Tensor,
        cat: torch.Tensor,
        cat_mask: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        _ = args, kwargs
        return self.state_encoder(cont, cont_mask, cat, cat_mask)

    def encode_observation(
        self,
        cont: torch.Tensor,
        cont_mask: torch.Tensor,
        cat: torch.Tensor,
        cat_mask: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.encode_wave(cont, cont_mask, cat, cat_mask, *args, **kwargs)

    def encode_static(
        self,
        static_cont: torch.Tensor,
        static_cont_mask: torch.Tensor,
        static_cat: torch.Tensor,
        static_cat_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.static_encoder is None:
            return static_cont.new_zeros(static_cont.shape[0], 0)
        return self.static_encoder(
            static_cont, static_cont_mask, static_cat, static_cat_mask
        )

    def _encode_dynamic_trajectory(
        self,
        state_encoder: StateEncoder,
        temporal_encoder: TemporalTransformerEncoder,
        state_cont: torch.Tensor,
        state_cont_mask: torch.Tensor,
        state_cat: torch.Tensor,
        state_cat_mask: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        steps = state_cont.shape[1]
        tokens = [
            state_encoder(
                state_cont[:, t],
                state_cont_mask[:, t],
                state_cat[:, t],
                state_cat_mask[:, t],
            )
            for t in range(steps)
        ]
        return temporal_encoder(torch.stack(tokens, dim=1), valid)

    def encode_trajectory_waves(
        self,
        state_cont: torch.Tensor,
        state_cont_mask: torch.Tensor,
        state_cat: torch.Tensor,
        state_cat_mask: torch.Tensor,
        valid: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Online path: s_1:T → Temporal → z_1:T."""
        _ = args, kwargs
        return self._encode_dynamic_trajectory(
            self.state_encoder,
            self.temporal_encoder,
            state_cont,
            state_cont_mask,
            state_cat,
            state_cat_mask,
            valid,
        )

    @torch.no_grad()
    def encode_target_trajectory(
        self,
        state_cont: torch.Tensor,
        state_cont_mask: torch.Tensor,
        state_cat: torch.Tensor,
        state_cat_mask: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """EMA path: s_1:L → Temporal_EMA → z_1:L. L is usually t+1, and the last frame is target_z."""
        return self._encode_dynamic_trajectory(
            self.target_state_encoder,
            self.target_temporal_encoder,
            state_cont,
            state_cont_mask,
            state_cat,
            state_cat_mask,
            valid,
        )

    @torch.no_grad()
    def encode_target(
        self,
        state_cont: torch.Tensor,
        state_cont_mask: torch.Tensor,
        state_cat: torch.Tensor,
        state_cat_mask: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Last frame of EMA Temporal(s_1:t+1), used as target_z."""
        z_seq = self.encode_target_trajectory(
            state_cont,
            state_cont_mask,
            state_cat,
            state_cat_mask,
            valid,
        )
        return z_seq[:, -1]

    def predict_next(
        self,
        z_t: torch.Tensor,
        action_cont: torch.Tensor,
        action_cont_mask: torch.Tensor,
        action_cat: torch.Tensor,
        delta_t_norm: torch.Tensor,
        static_cont: torch.Tensor | None = None,
        static_cont_mask: torch.Tensor | None = None,
        static_cat: torch.Tensor | None = None,
        static_cat_mask: torch.Tensor | None = None,
        static_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if static_embed is None:
            if self.has_static:
                if (
                    static_cont is None
                    or static_cont_mask is None
                    or static_cat is None
                    or static_cat_mask is None
                ):
                    raise ValueError("static tensors required for conditioned predictor")
                static_embed = self.encode_static(
                    static_cont, static_cont_mask, static_cat, static_cat_mask
                )
            else:
                static_embed = z_t.new_zeros(z_t.shape[0], 0)
        action_embed = self.action_encoder(action_cont, action_cont_mask, action_cat)
        return self.predictor(z_t, static_embed, action_embed, delta_t_norm)

    @torch.no_grad()
    def update_target_encoder(self, tau: float) -> None:
        """EMA update of the dynamic tower: StateEncoder and TemporalEncoder, not the static encoder."""
        pairs = (
            (self.target_state_encoder, self.state_encoder),
            (self.target_temporal_encoder, self.temporal_encoder),
        )
        for target_module, source_module in pairs:
            for target, source in zip(
                target_module.parameters(), source_module.parameters(), strict=True
            ):
                target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)


class JEPAAgent(nn.Module):
    """Training and evaluation entry point. Contains the health world model only."""

    def __init__(self, spec: ModelSpec, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.spec = spec
        self.world = WorldModel(spec, model_cfg)
