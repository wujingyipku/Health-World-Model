"""Collectors for the health world model evaluation report.

Plotting / CLI live in ``evaluate_and_plot_hrs_jepa.py``. This module focuses on
latent metrics, MLP baseline fit/cache, probing, rollouts, and clinical tables.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from code.data import TrajectoryBatch, build_loader
from code.evaluation import (
    CLINICAL_EVENT_REWARDS,
    CONTINUOUS_REWARDS,
    DEFAULT_BINARY_POS_WEIGHT_MAX,
    EVENT_LABELS,
    LEVEL_OUTCOMES,
    WORSENING_REWARDS,
    baseline_cache_matches_features,
    baseline_cache_matches_pos_weight,
    baseline_feature_columns,
    expand_binary_by_pos_weight,
    brier,
    inverse_standardized,
    load_or_prepare_baselines,
    read_table,
    safe_auprc,
    binary_rates_at_threshold,
    sensitivity_at_min_specificity,
)
from code.nn import JEPAAgent, MLP
from code.specs import (
    ModelSpec,
    continuous_reward_standardize_info,
    materialize_pooled_binary_rewards,
    resolve_pooled_binary_rewards,
)
from code.trainer import (
    encode_batch_static,
    encode_batch_target,
    encode_batch_trajectory,
    safe_auroc,
)

# Reward (next-wave level) → current-state column for persistence.
LEVEL_TO_STATE = {
    "adl_worsening": "adl_total_score",
    "iadl_worsening": "iadl_total_score",
}

# Fig 2: shared PCA of z, colored by these Bernoulli labels (0/1), panel order.
FIG2_PCA_REWARDS: tuple[str, ...] = (
    "death_event",
    "adl_worsening",
    "iadl_worsening",
)
FIG2_CONTINUOUS_REWARDS = list(WORSENING_REWARDS)
# Positive worsening_delta = health got worse (state-score sign).
FIG2_WORSEN_SIGN = {
    "adl_worsening": 1.0,
    "iadl_worsening": 1.0,
}

# Frozen logistic probe z_t → next-wave binary rewards (preferred order).
PROBE_BINARY_REWARDS = list(CLINICAL_EVENT_REWARDS)
PRIMARY_LEVELS = list(WORSENING_REWARDS)


# ---------------------------------------------------------------------------
# MLP baseline (legacy world-model; Fig 3/4 now use tabular MLP — see fit_mlp_*)
# ---------------------------------------------------------------------------


class MLPWorldBaseline(nn.Module):
    """Deprecated latent world-model MLP (kept for optional legacy scripts).

    Evaluation report MLP bars now use sklearn MLPRegressor/Classifier on the
    same state__/action__ features as Linear/Logistic (no encoder latent).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_binary: int,
        n_continuous: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = MLP(input_dim, latent_dim, hidden_dim, layers=2)
        self.predictor = MLP(latent_dim + 1, latent_dim, hidden_dim, layers=2)  # +dt
        self.binary = (
            nn.Linear(latent_dim, n_binary) if n_binary > 0 else None
        )
        self.continuous = (
            nn.Linear(latent_dim, n_continuous) if n_continuous > 0 else None
        )
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def predict_latent(self, z: torch.Tensor, delta_t_norm: torch.Tensor) -> torch.Tensor:
        if delta_t_norm.ndim == 1:
            delta_t_norm = delta_t_norm[:, None]
        return self.predictor(torch.cat([z, delta_t_norm], dim=-1))

    def clinical(self, z: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        logits = self.binary(z) if self.binary is not None else None
        cont = self.continuous(z) if self.continuous is not None else None
        return logits, cont


def flat_features(batch: TrajectoryBatch, t: int) -> torch.Tensor:
    """Concatenate current state / static / action / masks at step t."""
    parts = [
        batch.state_cont[:, t],
        batch.state_cont_mask[:, t],
        batch.state_cat[:, t].float(),
        batch.state_cat_mask[:, t],
        batch.static_cont[:, t],
        batch.static_cont_mask[:, t],
        batch.static_cat[:, t].float(),
        batch.static_cat_mask[:, t],
        batch.action_cont[:, t],
        batch.action_cont_mask[:, t],
        batch.action_cat[:, t].float(),
        batch.action_cat_mask[:, t],
    ]
    return torch.cat(parts, dim=-1)


# Backward-compatible alias.
_flat_features = flat_features


def _infer_input_dim(batch: TrajectoryBatch) -> int:
    return int(flat_features(batch, 0).shape[-1])


def mlp_cache_paths(baseline_dir: Path) -> dict[str, Path]:
    baseline_dir = Path(baseline_dir)
    return {
        "weights": baseline_dir / "mlp_baseline.pt",
        "manifest": baseline_dir / "mlp_baseline_manifest.json",
    }


def train_or_load_mlp_baseline(
    *,
    train_data: Path,
    test_data: Path,
    spec: ModelSpec,
    device: torch.device,
    baseline_dir: Path,
    latent_dim: int,
    seed: int = 2026,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
    force: bool = False,
) -> MLPWorldBaseline:
    """Fit MLP on train transitions; cache under ``baseline_dir``."""
    paths = mlp_cache_paths(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    n_bin = len(spec.reward_binary)
    n_cont = len(spec.reward_continuous)

    train_loader = build_loader(
        table_path=train_data,
        spec=spec,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        seed=seed,
    )
    probe = next(iter(train_loader)).to(device)
    input_dim = _infer_input_dim(probe)

    model = MLPWorldBaseline(input_dim, latent_dim, n_bin, n_cont).to(device)
    if paths["weights"].exists() and not force:
        payload = torch.load(paths["weights"], map_location=device, weights_only=False)
        if int(payload.get("input_dim", -1)) == input_dim:
            model.load_state_dict(payload["state_dict"])
            model.eval()
            print(f"Reusing cached MLP baseline: {paths['weights']}", flush=True)
            return model
        print("MLP cache input_dim mismatch; refitting...", flush=True)

    print(f"Fitting MLP baseline ({epochs} epochs) -> {paths['weights']}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    rng = np.random.default_rng(seed)
    _ = rng  # reserved for future subsample
    for epoch in range(epochs):
        losses = []
        for raw in train_loader:
            batch = raw.to(device)
            _, steps = batch.valid.shape
            total = batch.valid.new_zeros(())
            count = 0
            for t in range(steps):
                valid = batch.valid[:, t] > 0.5
                if not valid.any():
                    continue
                x = _flat_features(batch, t)
                z = model.encode(x)
                # Target latent: encode next-state flat features (same encoder).
                x_next = torch.cat(
                    [
                        batch.next_state_cont[:, t],
                        batch.next_state_cont_mask[:, t],
                        batch.next_state_cat[:, t].float(),
                        batch.next_state_cat_mask[:, t],
                        batch.static_cont[:, t],
                        batch.static_cont_mask[:, t],
                        batch.static_cat[:, t].float(),
                        batch.static_cat_mask[:, t],
                        batch.action_cont[:, t],
                        batch.action_cont_mask[:, t],
                        batch.action_cat[:, t].float(),
                        batch.action_cat_mask[:, t],
                    ],
                    dim=-1,
                )
                with torch.no_grad():
                    z_tgt = model.encode(x_next)
                z_pred = model.predict_latent(z, batch.delta_t_norm[:, t])
                latent_loss = F.mse_loss(z_pred[valid], z_tgt[valid])
                logits, cont = model.clinical(z_pred)
                clin = latent_loss.new_zeros(())
                if logits is not None and n_bin:
                    m = batch.reward_mask[:, t, :n_bin] * valid[:, None].float()
                    if m.sum() > 0:
                        bce = F.binary_cross_entropy_with_logits(
                            logits,
                            batch.reward[:, t, :n_bin],
                            reduction="none",
                        )
                        clin = clin + (bce * m).sum() / m.sum().clamp_min(1.0)
                if cont is not None and n_cont:
                    m = batch.reward_mask[:, t, n_bin:] * valid[:, None].float()
                    if m.sum() > 0:
                        mse = (cont - batch.reward[:, t, n_bin:]).square()
                        clin = clin + (mse * m).sum() / m.sum().clamp_min(1.0)
                loss = latent_loss + clin
                total = total + loss
                count += 1
            if count == 0:
                continue
            opt.zero_grad(set_to_none=True)
            (total / count).backward()
            opt.step()
            losses.append(float((total / count).detach().cpu()))
        print(
            f"  MLP epoch {epoch + 1}/{epochs} loss={np.mean(losses) if losses else float('nan'):.4f}",
            flush=True,
        )

    model.eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "n_binary": n_bin,
            "n_continuous": n_cont,
            "seed": seed,
            "epochs": epochs,
        },
        paths["weights"],
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "train_data": str(train_data),
                "test_data": str(test_data),
                "weights": str(paths["weights"]),
                "input_dim": input_dim,
                "latent_dim": latent_dim,
                "epochs": epochs,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model


# ---------------------------------------------------------------------------
# Target helpers
# ---------------------------------------------------------------------------


@torch.inference_mode()
def encode_target_at_horizon(
    world: Any,
    batch: TrajectoryBatch,
    t0: int,
    horizon: int,
) -> torch.Tensor:
    """EMA Temporal(s_1:t0+horizon) last frame when observed waves exist."""
    # Prefix through current wave t0, then append next_state for each step.
    cont_parts = [batch.state_cont[:, : t0 + 1]]
    cont_mask_parts = [batch.state_cont_mask[:, : t0 + 1]]
    cat_parts = [batch.state_cat[:, : t0 + 1]]
    cat_mask_parts = [batch.state_cat_mask[:, : t0 + 1]]
    valid_parts = [batch.valid[:, : t0 + 1]]
    for h in range(horizon):
        t = t0 + h
        cont_parts.append(batch.next_state_cont[:, t : t + 1])
        cont_mask_parts.append(batch.next_state_cont_mask[:, t : t + 1])
        cat_parts.append(batch.next_state_cat[:, t : t + 1])
        cat_mask_parts.append(batch.next_state_cat_mask[:, t : t + 1])
        # Validity of appended wave follows the transition that produced it.
        valid_parts.append(batch.valid[:, t : t + 1])
    cont = torch.cat(cont_parts, dim=1)
    cont_mask = torch.cat(cont_mask_parts, dim=1)
    cat = torch.cat(cat_parts, dim=1)
    cat_mask = torch.cat(cat_mask_parts, dim=1)
    valid = torch.cat(valid_parts, dim=1)
    return world.encode_target(cont, cont_mask, cat, cat_mask, valid)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_n = F.normalize(a, dim=-1, eps=eps)
    b_n = F.normalize(b, dim=-1, eps=eps)
    return (a_n * b_n).sum(dim=-1)


# ---------------------------------------------------------------------------
# Fig 1 collectors
# ---------------------------------------------------------------------------


@torch.inference_mode()
def collect_jepa_latent_prediction(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    horizons: Sequence[int] = (1, 2, 3, 5),
) -> pd.DataFrame:
    """One-step and multi-horizon latent cosine / MSE vs EMA targets.

    Also records a *latent persistence* reference (predict z_{t0+h} = z_t0).
    It shares the predictor's embedding space and target, so the two are
    directly comparable; cross-model latent numbers are not.
    """
    world = agent.world
    rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        _, steps = batch.valid.shape
        z_seq = encode_batch_trajectory(world, batch)
        static_embed = encode_batch_static(world, batch, t=0)
        for t0 in range(steps):
            valid0 = batch.valid[:, t0] > 0.5
            if not valid0.any():
                continue
            z = z_seq[:, t0]
            for h in horizons:
                if t0 + h - 1 >= steps:
                    break
                # All intermediate transitions must be valid for open-loop compare.
                active = valid0.clone()
                z_roll = z
                for k in range(h):
                    t = t0 + k
                    step_valid = batch.valid[:, t] > 0.5
                    active = active & step_valid
                    z_roll = world.predict_next(
                        z_roll,
                        batch.action_cont[:, t],
                        batch.action_cont_mask[:, t],
                        batch.action_cat[:, t],
                        batch.delta_t_norm[:, t],
                        static_embed=static_embed,
                    )
                if not active.any():
                    continue
                if h == 1:
                    target = encode_batch_target(world, batch, t0)
                else:
                    target = encode_target_at_horizon(world, batch, t0, h)
                cos = _cosine_sim(z_roll, target)
                mse = (z_roll - target).square().mean(dim=-1)
                # Same-space reference: no latent motion at all.
                cos_pers = _cosine_sim(z, target)
                mse_pers = (z - target).square().mean(dim=-1)
                idx = active.nonzero(as_tuple=False).view(-1).cpu().numpy()
                pids = np.asarray(batch.person_id, dtype=object)
                waves = batch.wave[:, t0].detach().cpu().numpy().astype(int)
                for i in idx:
                    rows.append(
                        {
                            "model": "JEPA",
                            "person_id": str(pids[i]),
                            "wave": int(waves[i]),
                            "horizon": int(h),
                            "cosine_sim": float(cos[i].cpu()),
                            "latent_mse": float(mse[i].cpu()),
                            "persistence_cosine": float(cos_pers[i].cpu()),
                            "persistence_mse": float(mse_pers[i].cpu()),
                        }
                    )
    return pd.DataFrame(rows)


@torch.inference_mode()
def collect_mlp_latent_prediction(
    mlp: MLPWorldBaseline,
    loader: DataLoader,
    device: torch.device,
    horizons: Sequence[int] = (1, 2, 3, 5),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        _, steps = batch.valid.shape
        for t0 in range(steps):
            valid0 = batch.valid[:, t0] > 0.5
            if not valid0.any():
                continue
            z = mlp.encode(_flat_features(batch, t0))
            for h in horizons:
                if t0 + h - 1 >= steps:
                    break
                active = valid0.clone()
                z_roll = z
                for k in range(h):
                    t = t0 + k
                    active = active & (batch.valid[:, t] > 0.5)
                    z_roll = mlp.predict_latent(z_roll, batch.delta_t_norm[:, t])
                if not active.any():
                    continue
                # Target = encode next-state features at last horizon step.
                t_last = t0 + h - 1
                x_tgt = torch.cat(
                    [
                        batch.next_state_cont[:, t_last],
                        batch.next_state_cont_mask[:, t_last],
                        batch.next_state_cat[:, t_last].float(),
                        batch.next_state_cat_mask[:, t_last],
                        batch.static_cont[:, t_last],
                        batch.static_cont_mask[:, t_last],
                        batch.static_cat[:, t_last].float(),
                        batch.static_cat_mask[:, t_last],
                        batch.action_cont[:, t_last],
                        batch.action_cont_mask[:, t_last],
                        batch.action_cat[:, t_last].float(),
                        batch.action_cat_mask[:, t_last],
                    ],
                    dim=-1,
                )
                z_tgt = mlp.encode(x_tgt)
                cos = _cosine_sim(z_roll, z_tgt)
                mse = (z_roll - z_tgt).square().mean(dim=-1)
                idx = active.nonzero(as_tuple=False).view(-1).cpu().numpy()
                pids = np.asarray(batch.person_id, dtype=object)
                waves = batch.wave[:, t0].detach().cpu().numpy().astype(int)
                for i in idx:
                    rows.append(
                        {
                            "model": "MLP",
                            "person_id": str(pids[i]),
                            "wave": int(waves[i]),
                            "horizon": int(h),
                            "cosine_sim": float(cos[i].cpu()),
                            "latent_mse": float(mse[i].cpu()),
                        }
                    )
    return pd.DataFrame(rows)


@torch.inference_mode()
def collect_latent_variance(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    max_rows: int = 50_000,
) -> pd.DataFrame:
    world = agent.world
    zs: list[np.ndarray] = []
    n = 0
    for raw in loader:
        batch = raw.to(device)
        z_seq = encode_batch_trajectory(world, batch)
        valid = batch.valid > 0.5
        z = z_seq[valid].detach().cpu().numpy()
        if len(z) == 0:
            continue
        zs.append(z)
        n += len(z)
        if n >= max_rows:
            break
    if not zs:
        return pd.DataFrame(columns=["dim", "std"])
    mat = np.concatenate(zs, axis=0)[:max_rows]
    std = mat.std(axis=0)
    return pd.DataFrame({"dim": np.arange(len(std)), "std": std})


def _best_final_epoch(run_dir: Path) -> int | None:
    """Resolve checkpoint epoch for annotating test ``world_loss``."""
    ckpt_path = Path(run_dir) / "best_final.pt"
    if not ckpt_path.exists():
        return None
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    extra = payload.get("extra") or {}
    epoch = extra.get("best_epoch")
    if epoch is None:
        return None
    try:
        return int(epoch)
    except (TypeError, ValueError):
        return None


def load_training_history_curves(run_dir: Path) -> pd.DataFrame:
    """Load train / val / test ``world_loss`` curves for Fig 1C.

    Train rows (``phase=world``): ``world_loss`` → ``train_world_loss``.

    Val rows (``phase=validation_world``): ``posterior_world_loss`` →
    ``val_world_loss``.

    Test row (``phase=test_world_final``): scalar ``posterior_world_loss`` stored
    in ``DataFrame.attrs['test_world_loss']`` (single eval on ``best_final.pt``).
    """
    path = Path(run_dir) / "training_history.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()

    train = frame
    val = pd.DataFrame()
    test = pd.DataFrame()
    if "phase" in frame.columns:
        phase = frame["phase"].astype(str)
        train = frame.loc[phase.isin(["world", "train_world"])].copy()
        val = frame.loc[phase.eq("validation_world")].copy()
        test = frame.loc[phase.eq("test_world_final")].copy()
        if train.empty and val.empty:
            train = frame.copy()

    out = pd.DataFrame()
    if not train.empty and "epoch" in train.columns and "world_loss" in train.columns:
        out = train[["epoch", "world_loss"]].copy()
        out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce")
        out = out.rename(columns={"world_loss": "train_world_loss"})
        out = out.dropna(subset=["epoch"]).sort_values("epoch", kind="stable")
        out = out.drop_duplicates("epoch", keep="last")

    if not val.empty and "epoch" in val.columns:
        v = val.copy()
        v["epoch"] = pd.to_numeric(v["epoch"], errors="coerce")
        val_col = (
            "posterior_world_loss"
            if "posterior_world_loss" in v.columns
            else "world_loss"
            if "world_loss" in v.columns
            else None
        )
        if val_col is not None:
            v = v[["epoch", val_col]].rename(columns={val_col: "val_world_loss"})
            v = v.dropna(subset=["epoch"]).sort_values("epoch", kind="stable")
            v = v.drop_duplicates("epoch", keep="last")
            out = (
                v
                if out.empty
                else out.merge(v, on="epoch", how="outer").sort_values(
                    "epoch", kind="stable"
                )
            )

    test_world_loss = float("nan")
    if not test.empty:
        if "posterior_world_loss" in test.columns:
            test_world_loss = float(test["posterior_world_loss"].iloc[0])
        elif "world_loss" in test.columns:
            test_world_loss = float(test["world_loss"].iloc[0])

    useful = [
        c
        for c in out.columns
        if c != "epoch" and out[c].notna().any()
    ]
    if not useful and not np.isfinite(test_world_loss):
        return pd.DataFrame()

    out.attrs["test_world_loss"] = test_world_loss
    out.attrs["test_epoch"] = _best_final_epoch(run_dir)
    return out


def fig1c_export_table(history: pd.DataFrame) -> pd.DataFrame:
    """Serialize Fig 1C curves; append one row for scalar test ``world_loss``."""
    if history.empty:
        return history
    out = history.copy()
    test_wl = float(history.attrs.get("test_world_loss", float("nan")))
    if np.isfinite(test_wl):
        out["test_world_loss"] = np.nan
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "epoch": history.attrs.get("test_epoch"),
                            "train_world_loss": np.nan,
                            "val_world_loss": np.nan,
                            "test_world_loss": test_wl,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return out


# ---------------------------------------------------------------------------
# Fig 2 collectors
# ---------------------------------------------------------------------------


@torch.inference_mode()
def collect_latents_and_health(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    max_rows: int = 20_000,
) -> pd.DataFrame:
    """Per-transition online z_t plus next-wave reward labels."""
    world = agent.world
    reward_meta = preprocessing.get("reward_continuous", {})
    rows: list[dict[str, Any]] = []
    reward_binary = list(agent.spec.reward_binary)
    reward_continuous = list(agent.spec.reward_continuous)
    n_bin = len(reward_binary)
    for raw in loader:
        batch = raw.to(device)
        _, steps = batch.valid.shape
        z_seq = encode_batch_trajectory(world, batch)
        for t in range(steps):
            active = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
            if not active.any():
                continue
            z = z_seq[:, t].detach().cpu().numpy()
            pids = np.asarray(batch.person_id, dtype=object)
            waves = batch.wave[:, t].detach().cpu().numpy().astype(int)
            for i in np.where(active)[0]:
                rec: dict[str, Any] = {
                    "person_id": str(pids[i]),
                    "wave": int(waves[i]),
                    "z": z[i],
                }
                # Next-wave binary rewards (Fig 2B); invalid mask → NaN
                for j, name in enumerate(reward_binary):
                    if float(batch.reward_mask[i, t, j].cpu()) <= 0.5:
                        rec[name] = float("nan")
                    else:
                        rec[name] = float(batch.reward[i, t, j].cpu())
                # Next-wave continuous rewards (Fig 2C coloring); invalid → NaN
                for j, name in enumerate(reward_continuous):
                    rj = n_bin + j
                    if float(batch.reward_mask[i, t, rj].cpu()) <= 0.5:
                        rec[name] = float("nan")
                        continue
                    std_val = float(batch.reward[i, t, rj].cpu())
                    info = reward_meta.get(name, {})
                    if info:
                        rec[name] = float(
                            inverse_standardized(np.array([std_val]), info)[0]
                        )
                    else:
                        rec[name] = std_val
                rows.append(rec)
                if len(rows) >= max_rows:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def fit_binary_probe_metrics(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    model_name: str,
    outcome_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Frozen logistic probe: z_t → next-wave binary rewards (AUROC / AUPRC)."""
    from sklearn.linear_model import LogisticRegression

    if train_frame.empty or test_frame.empty:
        return pd.DataFrame()
    z_train_all = np.stack(train_frame["z"].to_numpy())
    z_test_all = np.stack(test_frame["z"].to_numpy())
    if outcome_names is None:
        preferred = [n for n in PROBE_BINARY_REWARDS if n in train_frame.columns]
        extras = [
            c
            for c in train_frame.columns
            if c in test_frame.columns
            and c not in preferred
            and c not in {"person_id", "wave", "z"}
            and pd.api.types.is_numeric_dtype(train_frame[c])
        ]
        # Prefer known event order; only keep columns that look like 0/1 rewards.
        outcome_names = preferred + [
            c
            for c in extras
            if c.endswith("_event")
            or c.endswith("_worsening")
        ]
    rows = []
    for name in outcome_names:
        if name in CONTINUOUS_REWARDS:
            continue
        if name not in train_frame or name not in test_frame:
            continue
        y_tr_raw = pd.to_numeric(train_frame[name], errors="coerce").to_numpy(float)
        y_te_raw = pd.to_numeric(test_frame[name], errors="coerce").to_numpy(float)
        m_tr = np.isfinite(y_tr_raw)
        m_te = np.isfinite(y_te_raw)
        if m_tr.sum() < 50 or m_te.sum() < 20:
            continue
        y_tr = (y_tr_raw[m_tr] > 0.5).astype(int)
        y_te = (y_te_raw[m_te] > 0.5).astype(int)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(z_train_all[m_tr], y_tr)
        prob = clf.predict_proba(z_test_all[m_te])[:, 1]
        rows.append(
            {
                "model": model_name,
                "outcome": name,
                "label": EVENT_LABELS.get(name, name),
                "auroc": safe_auroc(prob, y_te),
                "auprc": safe_auprc(y_te, prob),
                "n": int(m_te.sum()),
                "prevalence": float(y_te.mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fig 2 cache (latent health frames + probes + Δz)
# ---------------------------------------------------------------------------

FIG2_CACHE_VERSION = "3"
FIG2_TRAIN_MAX_ROWS = 30_000
FIG2_TEST_MAX_ROWS = 20_000


def _file_fingerprint(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    p = Path(path)
    if not p.exists():
        return {"path": str(p.resolve()) if p.parent.exists() else str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def fig2_cache_paths(cache_dir: Path) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    return {
        "dir": cache_dir,
        "manifest": cache_dir / "manifest.json",
        "jepa_train_meta": cache_dir / "jepa_train_meta.parquet",
        "jepa_train_z": cache_dir / "jepa_train_z.npy",
        "jepa_test_meta": cache_dir / "jepa_test_meta.parquet",
        "jepa_test_z": cache_dir / "jepa_test_z.npy",
        "mlp_train_meta": cache_dir / "mlp_train_meta.parquet",
        "mlp_train_z": cache_dir / "mlp_train_z.npy",
        "mlp_test_meta": cache_dir / "mlp_test_meta.parquet",
        "mlp_test_z": cache_dir / "mlp_test_z.npy",
        "probe": cache_dir / "fig2_linear_probe.csv",
        "binary_probe": cache_dir / "fig2_binary_probe.csv",
        "change": cache_dir / "fig2_latent_change.csv",
        "latent_2d": cache_dir / "fig2_latent_2d.csv",
        "pca_rewards": cache_dir / "fig2_latent_pca_by_reward.csv",
        "method_2d": cache_dir / "fig2_method_2d.txt",
    }


def build_fig2_cache_manifest(
    *,
    checkpoint: Path,
    train_data: Path,
    test_data: Path,
    preprocessing_path: Path,
    seed: int,
    train_max_rows: int,
    test_max_rows: int,
    skip_mlp: bool,
    mlp_weights: Path | None,
) -> dict[str, Any]:
    return {
        "cache_version": FIG2_CACHE_VERSION,
        "checkpoint": _file_fingerprint(checkpoint),
        "train_data": _file_fingerprint(train_data),
        "test_data": _file_fingerprint(test_data),
        "preprocessing": _file_fingerprint(preprocessing_path),
        "seed": int(seed),
        "train_max_rows": int(train_max_rows),
        "test_max_rows": int(test_max_rows),
        "skip_mlp": bool(skip_mlp),
        "mlp_weights": _file_fingerprint(mlp_weights),
        "pca_reward_names": list(FIG2_PCA_REWARDS),
    }


def _fig2_manifest_matches(cached: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    keys = [
        "cache_version",
        "checkpoint",
        "train_data",
        "test_data",
        "preprocessing",
        "seed",
        "train_max_rows",
        "test_max_rows",
        # skip_mlp / mlp_weights ignored: Fig 2A/B no longer store MLP probes;
        # callers filter probe CSVs to JEPA-only.
    ]
    for key in keys:
        if cached.get(key) != expected.get(key):
            return False
    return True


def save_health_frame(frame: pd.DataFrame, meta_path: Path, z_path: Path) -> None:
    """Persist health frame with object ``z`` as meta parquet + float32 npy."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        pd.DataFrame().to_parquet(meta_path, index=False)
        np.save(z_path, np.zeros((0, 0), dtype=np.float32))
        return
    if "z" not in frame.columns:
        raise ValueError("health frame missing z column")
    z = np.stack(frame["z"].to_numpy()).astype(np.float32, copy=False)
    meta = frame.drop(columns=["z"]).copy()
    meta.to_parquet(meta_path, index=False)
    np.save(z_path, z)


def load_health_frame(meta_path: Path, z_path: Path) -> pd.DataFrame:
    if not meta_path.exists() or not z_path.exists():
        return pd.DataFrame()
    meta = pd.read_parquet(meta_path)
    z = np.load(z_path)
    if meta.empty:
        return pd.DataFrame()
    if len(meta) != len(z):
        raise ValueError(
            f"Fig2 cache length mismatch: meta={len(meta)} z={len(z)} ({meta_path.name})"
        )
    out = meta.copy()
    out["z"] = list(z)
    return out


@torch.inference_mode()
def collect_mlp_latents_and_health(
    mlp: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_frame: pd.DataFrame,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Encode MLP ``z`` and attach labels from a matching JEPA health frame."""
    rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        _, steps = batch.valid.shape
        for t in range(steps):
            active = (batch.valid[:, t] > 0.5).cpu().numpy()
            if not active.any():
                continue
            z = mlp.encode(flat_features(batch, t)).detach().cpu().numpy()
            pids = np.asarray(batch.person_id, dtype=object)
            waves = batch.wave[:, t].cpu().numpy().astype(int)
            for i in np.where(active)[0]:
                rows.append(
                    {
                        "person_id": str(pids[i]),
                        "wave": int(waves[i]),
                        "z": z[i],
                    }
                )
                if max_rows is not None and len(rows) >= max_rows:
                    break
            if max_rows is not None and len(rows) >= max_rows:
                break
        if max_rows is not None and len(rows) >= max_rows:
            break
    frame = pd.DataFrame(rows)
    if frame.empty or label_frame.empty:
        return pd.DataFrame()
    cols = [c for c in label_frame.columns if c not in {"z"}]
    return frame.merge(label_frame[cols], on=["person_id", "wave"], how="inner")


def try_load_fig2_cache(
    cache_dir: Path,
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load Fig 2 artifacts when manifest matches; else ``None``."""
    paths = fig2_cache_paths(cache_dir)
    if not paths["manifest"].exists():
        return None
    try:
        cached_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not _fig2_manifest_matches(cached_manifest, expected_manifest):
        return None

    required = [
        paths["jepa_train_meta"],
        paths["jepa_train_z"],
        paths["jepa_test_meta"],
        paths["jepa_test_z"],
        paths["probe"],
        paths["binary_probe"],
        paths["change"],
        paths["latent_2d"],
        paths["pca_rewards"],
        paths["method_2d"],
    ]
    skip_mlp = bool(expected_manifest.get("skip_mlp"))
    if not skip_mlp:
        required.extend(
            [
                paths["mlp_train_meta"],
                paths["mlp_train_z"],
                paths["mlp_test_meta"],
                paths["mlp_test_z"],
            ]
        )
    if not all(p.exists() for p in required):
        return None

    try:
        jepa_train = load_health_frame(paths["jepa_train_meta"], paths["jepa_train_z"])
        jepa_test = load_health_frame(paths["jepa_test_meta"], paths["jepa_test_z"])
        probe = pd.read_csv(paths["probe"])
        binary_probe = pd.read_csv(paths["binary_probe"])
        change = pd.read_csv(paths["change"])
        latent_2d = pd.read_csv(paths["latent_2d"])
        pca_rewards = pd.read_csv(paths["pca_rewards"])
        method_2d = paths["method_2d"].read_text(encoding="utf-8").strip() or "pca"
        mlp_train = mlp_test = pd.DataFrame()
        if not skip_mlp:
            mlp_train = load_health_frame(paths["mlp_train_meta"], paths["mlp_train_z"])
            mlp_test = load_health_frame(paths["mlp_test_meta"], paths["mlp_test_z"])
    except Exception as exc:  # noqa: BLE001
        print(f"Fig 2 cache unreadable ({exc}); recomputing...", flush=True)
        return None

    print(f"Reusing Fig 2 cache: {paths['dir']}", flush=True)
    return {
        "jepa_train": jepa_train,
        "jepa_test": jepa_test,
        "mlp_train": mlp_train,
        "mlp_test": mlp_test,
        "probe": probe,
        "binary_probe": binary_probe,
        "change": change,
        "latent_2d": latent_2d,
        "pca_rewards": pca_rewards,
        "method_2d": method_2d,
        "manifest": cached_manifest,
    }


def write_fig2_cache(
    cache_dir: Path,
    *,
    manifest: Mapping[str, Any],
    jepa_train: pd.DataFrame,
    jepa_test: pd.DataFrame,
    mlp_train: pd.DataFrame,
    mlp_test: pd.DataFrame,
    probe: pd.DataFrame,
    binary_probe: pd.DataFrame,
    change: pd.DataFrame,
    latent_2d: pd.DataFrame,
    pca_rewards: pd.DataFrame,
    method_2d: str,
) -> None:
    """Write Fig 2 health / probe / geometry cache."""
    paths = fig2_cache_paths(cache_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    save_health_frame(jepa_train, paths["jepa_train_meta"], paths["jepa_train_z"])
    save_health_frame(jepa_test, paths["jepa_test_meta"], paths["jepa_test_z"])
    skip_mlp = bool(manifest.get("skip_mlp"))
    if not skip_mlp:
        save_health_frame(mlp_train, paths["mlp_train_meta"], paths["mlp_train_z"])
        save_health_frame(mlp_test, paths["mlp_test_meta"], paths["mlp_test_z"])
    probe.to_csv(paths["probe"], index=False)
    binary_probe.to_csv(paths["binary_probe"], index=False)
    change.to_csv(paths["change"], index=False)
    latent_2d.to_csv(paths["latent_2d"], index=False)
    pca_rewards.to_csv(paths["pca_rewards"], index=False)
    paths["method_2d"].write_text(str(method_2d), encoding="utf-8")
    paths["manifest"].write_text(
        json.dumps(dict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote Fig 2 cache: {paths['dir']}", flush=True)


# ---------------------------------------------------------------------------
# Fig 5/6 cache (always-on + delayed-switch action simulations)
# ---------------------------------------------------------------------------

FIG56_CACHE_VERSION = "3"
FIG5_ALWAYS_ON_N_PERSONS = 256
FIG5_DELAYED_N_PERSONS = 512
FIG5_SWITCH_HORIZON = 3


def fig5_cache_paths(cache_dir: Path) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    return {
        "dir": cache_dir,
        "manifest": cache_dir / "manifest.json",
        "always_on": cache_dir / "always_on.parquet",
        "delayed_switch": cache_dir / "delayed_switch.parquet",
    }


def build_fig5_cache_manifest(
    *,
    checkpoint: Path,
    test_data: Path,
    seed: int,
    actions: Sequence[str],
    outcomes: Sequence[str],
    always_on_horizons: int,
    delayed_horizons: int,
    switch_horizon: int,
    n_persons_always_on: int,
    n_persons_delayed: int,
) -> dict[str, Any]:
    return {
        "cache_version": FIG56_CACHE_VERSION,
        "checkpoint": _file_fingerprint(checkpoint),
        "test_data": _file_fingerprint(test_data),
        "seed": int(seed),
        "actions": list(actions),
        "outcomes": list(outcomes),
        "always_on_horizons": int(always_on_horizons),
        "delayed_horizons": int(delayed_horizons),
        "switch_horizon": int(switch_horizon),
        "n_persons_always_on": int(n_persons_always_on),
        "n_persons_delayed": int(n_persons_delayed),
    }


def _fig5_manifest_matches(cached: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    keys = [
        "cache_version",
        "checkpoint",
        "test_data",
        "seed",
        "actions",
        "outcomes",
        "always_on_horizons",
        "delayed_horizons",
        "switch_horizon",
        "n_persons_always_on",
        "n_persons_delayed",
    ]
    for key in keys:
        if cached.get(key) != expected.get(key):
            return False
    return True


def try_load_fig5_cache(
    cache_dir: Path,
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load Fig 5/6 simulation tables when manifest matches; else ``None``."""
    paths = fig5_cache_paths(cache_dir)
    if not paths["manifest"].exists():
        return None
    try:
        cached_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not _fig5_manifest_matches(cached_manifest, expected_manifest):
        return None
    required = [paths["always_on"], paths["delayed_switch"]]
    if not all(p.exists() for p in required):
        return None
    try:
        always_on = _read_sim_table(paths["always_on"])
        delayed_switch = _read_sim_table(paths["delayed_switch"])
    except Exception as exc:  # noqa: BLE001
        print(f"Fig 5/6 cache unreadable ({exc}); recomputing...", flush=True)
        return None
    print(f"Reusing Fig 5/6 cache: {paths['dir']}", flush=True)
    return {
        "always_on": always_on,
        "delayed_switch": delayed_switch,
        "manifest": cached_manifest,
    }


_EMPTY_SIM_CACHE_COL = "_empty_cache"


def _write_sim_table(frame: pd.DataFrame, path: Path) -> None:
    out = frame
    if out.empty and len(out.columns) == 0:
        out = pd.DataFrame({_EMPTY_SIM_CACHE_COL: pd.Series(dtype="int8")})
    out.to_parquet(path, index=False)


def _read_sim_table(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if list(frame.columns) == [_EMPTY_SIM_CACHE_COL]:
        return pd.DataFrame()
    if _EMPTY_SIM_CACHE_COL in frame.columns:
        return frame.drop(columns=[_EMPTY_SIM_CACHE_COL])
    return frame


def write_fig5_cache(
    cache_dir: Path,
    *,
    manifest: Mapping[str, Any],
    always_on: pd.DataFrame,
    delayed_switch: pd.DataFrame,
) -> None:
    """Write Fig 5/6 always-on + delayed-switch simulation cache."""
    paths = fig5_cache_paths(cache_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    _write_sim_table(always_on, paths["always_on"])
    _write_sim_table(delayed_switch, paths["delayed_switch"])
    paths["manifest"].write_text(
        json.dumps(dict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote Fig 5/6 cache: {paths['dir']}", flush=True)


FIG56_VALIDATION_CACHE_VERSION = "15"
FIG4_DEFAULT_ACTION = "vigorous_activity_frequency"
FIG5_VALIDATION_ACTION = FIG4_DEFAULT_ACTION  # backward-compatible alias
FIG5_VALIDATION_OUTCOMES: tuple[str, ...] = (
    "adl_worsening",
    "iadl_worsening",
    "death_event",
)
FIG5_VALIDATION_OBS_STRATA: tuple[str, ...] = ("low", "mid", "high")
FIG5_VALIDATION_SIM_STRATA: tuple[str, ...] = ("low", "mid", "high")
FIG5_H1_CF_SWITCH_HORIZON = 2
FIG5_H1_CF_OBS_STRATA: tuple[str, ...] = ("stay_low", "switch_mid", "switch_high")
FIG5_H1_CF_SIM_STRATA: tuple[str, ...] = ("low", "mid", "high")
PERSISTENT_ACTION_MAX_HORIZON = 5
PERSISTENT_SMOKING_MAX_HORIZON = PERSISTENT_ACTION_MAX_HORIZON
PERSISTENT_ACTION_LOW_HORIZONS = 2
PERSISTENT_ACTION_MID_HIGH_HORIZONS = 2
PERSISTENT_SMOKING_LOW_HORIZONS = PERSISTENT_ACTION_LOW_HORIZONS
PERSISTENT_SMOKING_MID_HIGH_HORIZONS = PERSISTENT_ACTION_MID_HIGH_HORIZONS
PERSISTENT_SMOKING_MID_MAX_CIGS = 20.0
PERSISTENT_SMOKING_HIGH_MIN_CIGS = 20.0
PERSISTENT_SMOKING_ZERO_EPS = 0.05
FIG5_H1_CF_MID_RAW_CIGS = 10.0
FIG5_H1_CF_HIGH_RAW_CIGS = 20.0
AGE_STD_LABELS: tuple[str, ...] = ("<60", "60-69", "70-79", "80+")


@dataclass(frozen=True)
class Fig4ActionSpec:
    """Persistent low/mid/high + h1-low counterfactual levels for one action."""

    name: str
    kind: str  # "cont" | "cat"
    title: str
    binary: bool
    low_max: float
    high_min: float
    cf_low_raw: float
    cf_mid_raw: float | None
    cf_high_raw: float
    obs_low_label: str
    obs_exposed_label: str
    sim_low_label: str
    sim_mid_label: str | None
    sim_high_label: str
    sim_legend_suffix: str = "observed"


FIG4_ACTION_SPECS: dict[str, Fig4ActionSpec] = {
    "drinks_per_drinking_day": Fig4ActionSpec(
        name="drinks_per_drinking_day",
        kind="cont",
        title="Persistent drinking",
        binary=False,
        low_max=0.05,
        high_min=3.0,
        cf_low_raw=0.0,
        cf_mid_raw=2.0,
        cf_high_raw=4.0,
        obs_low_label="low (2×0 drinks)",
        obs_exposed_label="mid+high (2×>0 drinks)",
        sim_low_label="low (2×0 drinks)",
        sim_mid_label="mid (2×0–3 drinks)",
        sim_high_label="high (2×≥3 drinks)",
        sim_legend_suffix="observed drinks",
    ),
    "alcohol_days_per_week": Fig4ActionSpec(
        name="alcohol_days_per_week",
        kind="cont",
        title="Alcohol days per week",
        binary=False,
        low_max=0.05,
        high_min=4.0,
        cf_low_raw=0.0,
        cf_mid_raw=2.0,
        cf_high_raw=7.0,
        obs_low_label="low (2×0 days)",
        obs_exposed_label="mid+high (2×>0 days)",
        sim_low_label="low (2×0 days)",
        sim_mid_label="mid (2×1–3 days)",
        sim_high_label="high (2×≥4 days)",
        sim_legend_suffix="observed days",
    ),
    "cigarettes_per_day": Fig4ActionSpec(
        name="cigarettes_per_day",
        kind="cont",
        title="Persistent smoking",
        binary=False,
        low_max=PERSISTENT_SMOKING_ZERO_EPS,
        high_min=PERSISTENT_SMOKING_HIGH_MIN_CIGS,
        cf_low_raw=0.0,
        cf_mid_raw=FIG5_H1_CF_MID_RAW_CIGS,
        cf_high_raw=FIG5_H1_CF_HIGH_RAW_CIGS,
        obs_low_label="low (2×0 cigs)",
        obs_exposed_label="mid+high (2×>0 cigs)",
        sim_low_label="low (2×0 cigs)",
        sim_mid_label="mid (2×0–20 cigs)",
        sim_high_label="high (2×≥20 cigs)",
        sim_legend_suffix="observed cigs",
    ),
    "vigorous_activity_frequency": Fig4ActionSpec(
        name="vigorous_activity_frequency",
        kind="cat",
        title="Frequency of vigorous physical activity",
        binary=False,
        low_max=0.0,
        high_min=3.0,
        cf_low_raw=0.0,
        cf_mid_raw=2.0,
        cf_high_raw=4.0,
        obs_low_label="never (2×0)",
        obs_exposed_label="any (2×>0)",
        sim_low_label="never (2×0)",
        sim_mid_label="mid (2×1–2)",
        sim_high_label="high (2×3–4)",
        sim_legend_suffix="observed activity",
    ),
    "moderate_activity_frequency": Fig4ActionSpec(
        name="moderate_activity_frequency",
        kind="cat",
        title="Persistent moderate activity",
        binary=False,
        low_max=0.0,
        high_min=3.0,
        cf_low_raw=0.0,
        cf_mid_raw=2.0,
        cf_high_raw=4.0,
        obs_low_label="never (2×0)",
        obs_exposed_label="any (2×>0)",
        sim_low_label="never (2×0)",
        sim_mid_label="mid (2×1–2)",
        sim_high_label="high (2×3–4)",
        sim_legend_suffix="observed activity",
    ),
    "light_activity_frequency": Fig4ActionSpec(
        name="light_activity_frequency",
        kind="cat",
        title="Persistent light activity",
        binary=False,
        low_max=0.0,
        high_min=3.0,
        cf_low_raw=0.0,
        cf_mid_raw=2.0,
        cf_high_raw=4.0,
        obs_low_label="never (2×0)",
        obs_exposed_label="any (2×>0)",
        sim_low_label="never (2×0)",
        sim_mid_label="mid (2×1–2)",
        sim_high_label="high (2×3–4)",
        sim_legend_suffix="observed activity",
    ),
    "hypertension_treatment": Fig4ActionSpec(
        name="hypertension_treatment",
        kind="cat",
        title="Persistent antihypertensive",
        binary=True,
        low_max=0.0,
        high_min=1.0,
        cf_low_raw=0.0,
        cf_mid_raw=None,
        cf_high_raw=1.0,
        obs_low_label="off (2×0)",
        obs_exposed_label="on (2×1)",
        sim_low_label="off (2×0)",
        sim_mid_label=None,
        sim_high_label="on (2×1)",
        sim_legend_suffix="observed Rx",
    ),
    "diabetes_oral_medication": Fig4ActionSpec(
        name="diabetes_oral_medication",
        kind="cat",
        title="Persistent diabetes oral med",
        binary=True,
        low_max=0.0,
        high_min=1.0,
        cf_low_raw=0.0,
        cf_mid_raw=None,
        cf_high_raw=1.0,
        obs_low_label="off (2×0)",
        obs_exposed_label="on (2×1)",
        sim_low_label="off (2×0)",
        sim_mid_label=None,
        sim_high_label="on (2×1)",
        sim_legend_suffix="observed Rx",
    ),
}
FIG4_ACTION_ORDER: tuple[str, ...] = (
    "vigorous_activity_frequency",
    "drinks_per_drinking_day",
    "alcohol_days_per_week",
    "cigarettes_per_day",
    "moderate_activity_frequency",
    "light_activity_frequency",
    "hypertension_treatment",
    "diabetes_oral_medication",
)


def fig5_validation_cache_paths(cache_dir: Path) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    return {
        "dir": cache_dir,
        "manifest": cache_dir / "manifest.json",
        "observed_strata": cache_dir / "observed_strata.parquet",
        "sim_observed_action": cache_dir / "sim_observed_action.parquet",
        "observed_h1_counterfactual": cache_dir / "observed_h1_counterfactual.parquet",
        "sim_h1_counterfactual": cache_dir / "sim_h1_counterfactual.parquet",
    }


def build_fig5_validation_cache_manifest(
    *,
    test_data: Path,
    seed: int,
    action: str = FIG4_DEFAULT_ACTION,
    max_horizon: int = PERSISTENT_ACTION_MAX_HORIZON,
    outcomes: Sequence[str] = FIG5_VALIDATION_OUTCOMES,
    preprocessing_path: Path | None = None,
) -> dict[str, Any]:
    spec = FIG4_ACTION_SPECS.get(action)
    manifest: dict[str, Any] = {
        "cache_version": FIG56_VALIDATION_CACHE_VERSION,
        "validation_mode": (
            "persistent_action_first2_waves_obs_low_mid_high_sim_age_std"
            "_h1_low_sim_on_obs_groups_cf_h1_obs_by_stratum"
        ),
        "obs_strata": list(FIG5_VALIDATION_OBS_STRATA),
        "sim_strata": list(FIG5_VALIDATION_SIM_STRATA),
        "counterfactual_h1_low": True,
        "cf_switch_horizon": int(FIG5_H1_CF_SWITCH_HORIZON),
        "cf_obs_strata": list(FIG5_H1_CF_OBS_STRATA),
        "cf_sim_strata": list(FIG5_H1_CF_OBS_STRATA),
        "cf_sim_matched_obs_groups": True,
        "cf_h1_obs_pooled": False,
        "age_std": True,
        "age_std_labels": list(AGE_STD_LABELS),
        "test_data": _file_fingerprint(test_data),
        "seed": int(seed),
        "action": str(action),
        "max_horizon": int(max_horizon),
        "outcomes": list(outcomes),
        "low_horizons": int(PERSISTENT_ACTION_LOW_HORIZONS),
        "mid_high_horizons": int(PERSISTENT_ACTION_MID_HIGH_HORIZONS),
        "binary": bool(spec.binary) if spec is not None else False,
        "low_max": float(spec.low_max) if spec is not None else 0.0,
        "high_min": float(spec.high_min) if spec is not None else 0.0,
        "cf_low_raw": float(spec.cf_low_raw) if spec is not None else 0.0,
        "cf_mid_raw": (
            None
            if spec is None or spec.cf_mid_raw is None
            else float(spec.cf_mid_raw)
        ),
        "cf_high_raw": float(spec.cf_high_raw) if spec is not None else 0.0,
    }
    if preprocessing_path is not None:
        manifest["preprocessing"] = _file_fingerprint(preprocessing_path)
    return manifest


def _fig5_validation_manifest_matches(
    cached: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    keys = [
        "cache_version",
        "validation_mode",
        "test_data",
        "seed",
        "action",
        "max_horizon",
        "outcomes",
        "low_horizons",
        "mid_high_horizons",
        "binary",
        "low_max",
        "high_min",
        "age_std",
        "age_std_labels",
        "counterfactual_h1_low",
        "cf_switch_horizon",
        "cf_low_raw",
        "cf_mid_raw",
        "cf_high_raw",
        "cf_obs_strata",
        "cf_sim_strata",
        "cf_sim_matched_obs_groups",
        "cf_h1_obs_pooled",
    ]
    if not all(cached.get(k) == expected.get(k) for k in keys):
        return False
    if "preprocessing" in expected:
        return cached.get("preprocessing") == expected.get("preprocessing")
    return True


def try_load_fig5_validation_cache(
    cache_dir: Path,
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    paths = fig5_validation_cache_paths(cache_dir)
    if not paths["manifest"].exists():
        return None
    try:
        cached_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not _fig5_validation_manifest_matches(cached_manifest, expected_manifest):
        return None
    required = [
        paths["observed_strata"],
        paths["sim_observed_action"],
        paths["observed_h1_counterfactual"],
        paths["sim_h1_counterfactual"],
    ]
    if not all(p.exists() for p in required):
        return None
    try:
        observed_strata = _read_sim_table(paths["observed_strata"])
        sim_observed_action = _read_sim_table(paths["sim_observed_action"])
        observed_h1_counterfactual = _read_sim_table(paths["observed_h1_counterfactual"])
        sim_h1_counterfactual = _read_sim_table(paths["sim_h1_counterfactual"])
    except Exception as exc:  # noqa: BLE001
        print(f"Fig 5 validation cache unreadable ({exc}); recomputing...", flush=True)
        return None
    print(f"Reusing Fig 5 validation cache: {paths['dir']}", flush=True)
    return {
        "observed_strata": observed_strata,
        "sim_observed_action": sim_observed_action,
        "observed_h1_counterfactual": observed_h1_counterfactual,
        "sim_h1_counterfactual": sim_h1_counterfactual,
        "manifest": cached_manifest,
    }


def write_fig5_validation_cache(
    cache_dir: Path,
    *,
    manifest: Mapping[str, Any],
    observed_strata: pd.DataFrame,
    sim_observed_action: pd.DataFrame,
    observed_h1_counterfactual: pd.DataFrame,
    sim_h1_counterfactual: pd.DataFrame,
) -> None:
    paths = fig5_validation_cache_paths(cache_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    _write_sim_table(observed_strata, paths["observed_strata"])
    _write_sim_table(sim_observed_action, paths["sim_observed_action"])
    _write_sim_table(observed_h1_counterfactual, paths["observed_h1_counterfactual"])
    _write_sim_table(sim_h1_counterfactual, paths["sim_h1_counterfactual"])
    paths["manifest"].write_text(
        json.dumps(dict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote Fig 5 validation cache: {paths['dir']}", flush=True)


def compute_latent_2d(
    frame: pd.DataFrame,
    color_col: str = "adl_total_score",
    max_points: int = 4000,
    seed: int = 2026,
    *,
    force_pca: bool = False,
) -> tuple[pd.DataFrame, str]:
    """UMAP if available else PCA; returns points + method name."""
    if frame.empty or "z" not in frame.columns:
        return pd.DataFrame(), "none"
    use = frame.copy()
    if color_col in use.columns:
        use = use.dropna(subset=[color_col])
    if use.empty:
        return pd.DataFrame(), "none"
    if len(use) > max_points:
        use = use.sample(n=max_points, random_state=seed)
    z = np.stack(use["z"].to_numpy())
    method = "pca"
    if not force_pca:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2, random_state=seed, n_neighbors=30, min_dist=0.1
            )
            xy = reducer.fit_transform(z)
            method = "umap"
        except Exception:
            from sklearn.decomposition import PCA

            xy = PCA(n_components=2, random_state=seed).fit_transform(z)
            method = "pca"
    else:
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2, random_state=seed).fit_transform(z)
        method = "pca"
    out = use[["person_id", "wave"]].copy()
    out["x"] = xy[:, 0]
    out["y"] = xy[:, 1]
    if color_col in use.columns:
        out["color"] = use[color_col].to_numpy()
        out["color_name"] = color_col
    return out, method


def compute_latent_pca_by_rewards(
    frame: pd.DataFrame,
    reward_names: Sequence[str] | None = None,
    max_points: int = 4000,
    seed: int = 2026,
) -> pd.DataFrame:
    """One shared PCA of z; long table with one 0/1 color channel per binary label."""
    if frame.empty or "z" not in frame.columns:
        return pd.DataFrame()
    names = list(reward_names) if reward_names is not None else list(FIG2_PCA_REWARDS)
    names = [n for n in names if n in frame.columns]
    if not names:
        # Fall back to any continuous-reward-like columns present.
        names = [
            c
            for c in frame.columns
            if c not in CONTINUOUS_REWARDS and c.endswith("_worsening")
        ]
    if not names:
        return pd.DataFrame()
    use = frame.copy()
    if len(use) > max_points:
        use = use.sample(n=max_points, random_state=seed)
    from sklearn.decomposition import PCA

    z = np.stack(use["z"].to_numpy())
    xy = PCA(n_components=2, random_state=seed).fit_transform(z)
    rows = []
    for name in names:
        vals = pd.to_numeric(use[name], errors="coerce").to_numpy(float)
        m = np.isfinite(vals)
        if m.sum() < 20:
            continue
        for i in np.where(m)[0]:
            rows.append(
                {
                    "person_id": str(use["person_id"].iloc[i]),
                    "wave": int(use["wave"].iloc[i]),
                    "outcome": name,
                    "label": LEVEL_OUTCOMES.get(name, name),
                    "x": float(xy[i, 0]),
                    "y": float(xy[i, 1]),
                    "color": float(vals[i]),
                }
            )
    return pd.DataFrame(rows)


@torch.inference_mode()
def collect_latent_change_vs_decline(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    reward_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Correlate ||Δz|| with worsening of each continuous-reward mapped state."""
    world = agent.world
    state_meta = preprocessing.get("state_continuous", {})
    cont_names = list(agent.spec.state_continuous)
    names = list(reward_names) if reward_names is not None else list(FIG2_CONTINUOUS_REWARDS)
    specs: list[tuple[str, int, float]] = []
    for reward_name in names:
        state_name = LEVEL_TO_STATE.get(reward_name)
        if state_name is None or state_name not in cont_names:
            continue
        specs.append(
            (
                reward_name,
                cont_names.index(state_name),
                float(FIG2_WORSEN_SIGN.get(reward_name, 1.0)),
            )
        )
    if not specs:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        _, steps = batch.valid.shape
        z_seq = encode_batch_trajectory(world, batch)
        for t in range(steps):
            active = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
            if not active.any():
                continue
            z_t = z_seq[:, t]
            z_tp1 = encode_batch_target(world, batch, t)
            delta = (z_tp1 - z_t).norm(dim=-1).detach().cpu().numpy()
            pids = np.asarray(batch.person_id, dtype=object)
            for reward_name, state_idx, sign in specs:
                now = batch.state_cont[:, t, state_idx].detach().cpu().numpy()
                nxt = batch.next_state_cont[:, t, state_idx].detach().cpu().numpy()
                state_name = LEVEL_TO_STATE[reward_name]
                info = state_meta.get(state_name, {})
                if info:
                    now = inverse_standardized(now, info)
                    nxt = inverse_standardized(nxt, info)
                worsening = sign * (nxt - now)
                for i in np.where(active)[0]:
                    rows.append(
                        {
                            "person_id": str(pids[i]),
                            "outcome": reward_name,
                            "label": LEVEL_OUTCOMES.get(reward_name, reward_name),
                            "latent_change": float(delta[i]),
                            "worsening_delta": float(worsening[i]),
                            # Backward-compatible alias used by older ADL-only plot.
                            "adl_decline": float(worsening[i])
                            if reward_name == "adl_worsening"
                            else float("nan"),
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fig 3 collectors
# ---------------------------------------------------------------------------


def _persistence_raw_from_table(
    test: pd.DataFrame,
    preprocessing: Mapping[str, Any],
    level_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    keys = ["person_id", "wave", "next_wave"]
    out = test[keys].copy()
    out["person_id"] = out["person_id"].astype(str)
    names = list(level_names) if level_names else list(LEVEL_TO_STATE)
    for reward_name in names:
        state, info = continuous_reward_standardize_info(reward_name, preprocessing)
        if state is None:
            state = LEVEL_TO_STATE.get(reward_name)
        if not state:
            continue
        col = f"state__{state}"
        if col not in test.columns:
            continue
        std = pd.to_numeric(test[col], errors="coerce").to_numpy(float)
        if info:
            raw = inverse_standardized(np.nan_to_num(std, nan=0.0), info)
        else:
            raw = std
        out[f"persistence_raw__{reward_name}"] = raw.astype(np.float32)
    return out


def _level_supervision(
    name: str,
    preprocessing: Mapping[str, Any],
    columns: pd.Index | Sequence[str],
) -> tuple[str, str, dict[str, Any]] | None:
    """Columns + standardize info for a continuous reward on the transition table."""
    cols = set(columns)
    state, info = continuous_reward_standardize_info(name, preprocessing)
    if state and f"next_state__{state}" in cols:
        value_col = f"next_state__{state}"
        mask_col = (
            f"next_state_mask__{state}"
            if f"next_state_mask__{state}" in cols
            else f"state_mask__{state}"
        )
        return value_col, mask_col, info
    value_col = f"reward__{name}"
    mask_col = f"reward_mask__{name}"
    if value_col in cols and info:
        return value_col, mask_col, info
    return None


def fit_linear_level_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    preprocessing: Mapping[str, Any],
    seed: int,
    level_names: Sequence[str] | None = None,
    allowed_feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Ridge regression: state/action features → next-wave level (raw)."""
    from sklearn.linear_model import Ridge

    _ = seed
    keys = ["person_id", "wave", "next_wave"]
    result = test[keys].copy()
    result["person_id"] = result["person_id"].astype(str)
    features = baseline_feature_columns(
        train.columns, allowed_names=allowed_feature_names
    )
    if not features:
        return result
    x_train = np.nan_to_num(train[features].to_numpy(np.float32), nan=0.0)
    x_test = np.nan_to_num(test[features].to_numpy(np.float32), nan=0.0)
    names = list(level_names) if level_names else list(LEVEL_OUTCOMES)
    for name in names:
        spec = _level_supervision(name, preprocessing, train.columns)
        if spec is None:
            continue
        value_col, mask_col, info = spec
        y_std = pd.to_numeric(train[value_col], errors="coerce").fillna(0).to_numpy()
        y = inverse_standardized(y_std, info) if info else y_std
        mask = (
            pd.to_numeric(train[mask_col], errors="coerce").fillna(0).to_numpy() > 0.5
            if mask_col in train.columns
            else np.isfinite(y)
        )
        if int(mask.sum()) < 50:
            continue
        reg = Ridge(alpha=1.0)
        reg.fit(x_train[mask], y[mask])
        result[f"linear_raw__{name}"] = reg.predict(x_test).astype(np.float32)
    return result


def mlp_tabular_cache_paths(baseline_dir: Path) -> dict[str, Path]:
    baseline_dir = Path(baseline_dir)
    return {
        "predictions": baseline_dir / "mlp_tabular_predictions.parquet",
        "manifest": baseline_dir / "mlp_tabular_manifest.json",
    }


def fit_mlp_level_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    preprocessing: Mapping[str, Any],
    seed: int,
    level_names: Sequence[str] | None = None,
    allowed_feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """sklearn MLPRegressor: same state/action features as Linear → next-wave level (raw)."""
    from sklearn.neural_network import MLPRegressor

    keys = ["person_id", "wave", "next_wave"]
    result = test[keys].copy()
    result["person_id"] = result["person_id"].astype(str)
    features = baseline_feature_columns(
        train.columns, allowed_names=allowed_feature_names
    )
    if not features:
        return result
    x_train = np.nan_to_num(train[features].to_numpy(np.float32), nan=0.0)
    x_test = np.nan_to_num(test[features].to_numpy(np.float32), nan=0.0)
    names = list(level_names) if level_names else list(LEVEL_OUTCOMES)
    for name in names:
        spec = _level_supervision(name, preprocessing, train.columns)
        if spec is None:
            continue
        value_col, mask_col, info = spec
        y_std = pd.to_numeric(train[value_col], errors="coerce").fillna(0).to_numpy()
        y = inverse_standardized(y_std, info) if info else y_std
        mask = (
            pd.to_numeric(train[mask_col], errors="coerce").fillna(0).to_numpy() > 0.5
            if mask_col in train.columns
            else np.isfinite(y)
        )
        if int(mask.sum()) < 50:
            continue
        reg = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=seed,
        )
        reg.fit(x_train[mask], y[mask])
        result[f"mlp_raw__{name}"] = reg.predict(x_test).astype(np.float32)
    return result


def fit_mlp_event_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    event_names: Sequence[str],
    *,
    binary_pos_weight_max: float = DEFAULT_BINARY_POS_WEIGHT_MAX,
    allowed_feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """sklearn MLPClassifier: same state/action features as Logistic → event probs.

    Older sklearn MLP has neither ``class_weight`` nor ``sample_weight``. Positive
    rows are duplicated ``round(clip(n_neg/n_pos, 1, max))`` times so the class
    ratio matches JEPA / Logistic ``pos_weight``.
    """
    from sklearn.neural_network import MLPClassifier

    keys = ["person_id", "wave", "next_wave"]
    result = test[keys].copy()
    result["person_id"] = result["person_id"].astype(str)
    features = baseline_feature_columns(
        train.columns, allowed_names=allowed_feature_names
    )
    if not features:
        return result
    x_train = np.nan_to_num(
        train[features].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    x_test = np.nan_to_num(
        test[features].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    pooled = resolve_pooled_binary_rewards(event_names)
    train = materialize_pooled_binary_rewards(train, pooled)
    test = materialize_pooled_binary_rewards(test, pooled)
    weight_notes: list[str] = []
    for name in event_names:
        if name in CONTINUOUS_REWARDS:
            continue
        value_col = f"reward__{name}"
        mask_col = f"reward_mask__{name}"
        if value_col not in train or mask_col not in train:
            continue
        y_raw = pd.to_numeric(train[value_col], errors="coerce").fillna(0).to_numpy()
        mask = pd.to_numeric(train[mask_col], errors="coerce").fillna(0).to_numpy() > 0.5
        active = mask & np.isfinite(y_raw)
        masked = y_raw[active]
        if masked.size == 0:
            continue
        uniq = np.unique(np.round(masked, 6))
        if not np.all(np.isin(uniq, [0.0, 1.0])):
            continue
        y = (masked > 0.5).astype(int)
        prevalence = float(y.mean()) if len(y) else 0.0
        if len(np.unique(y)) < 2 or int(np.bincount(y).min()) < 10:
            result[f"mlp_prob__{name}"] = np.full(len(test), prevalence, dtype=np.float32)
            continue
        x_fit, y_fit, pos_w = expand_binary_by_pos_weight(
            x_train[active],
            y,
            max_weight=binary_pos_weight_max,
            seed=seed,
        )
        if pos_w is not None:
            weight_notes.append(f"{name}={pos_w:.3f}x{max(int(round(pos_w)), 1)}")
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=seed,
        )
        clf.fit(x_fit, y_fit)
        if hasattr(clf, "predict_proba"):
            result[f"mlp_prob__{name}"] = clf.predict_proba(x_test)[:, 1].astype(np.float32)
        else:
            result[f"mlp_prob__{name}"] = clf.predict(x_test).astype(np.float32)
    if weight_notes:
        print(
            "MLP pos oversample "
            f"(max={binary_pos_weight_max}): " + ", ".join(weight_notes),
            flush=True,
        )
    elif float(binary_pos_weight_max) <= 0:
        print("MLP pos oversample disabled (binary_pos_weight_max<=0)", flush=True)
    return result


def load_or_fit_mlp_tabular(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    preprocessing: Mapping[str, Any],
    baseline_dir: Path,
    seed: int,
    event_names: Sequence[str],
    refit: bool,
    skip: bool,
    level_names: Sequence[str] | None = None,
    binary_pos_weight_max: float = DEFAULT_BINARY_POS_WEIGHT_MAX,
    allowed_feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fit/cache tabular MLP levels + events (Linear-style features, no latent)."""
    if skip:
        return pd.DataFrame()
    paths = mlp_tabular_cache_paths(baseline_dir)
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    pos_max = float(binary_pos_weight_max)
    needed_levels = [f"mlp_raw__{n}" for n in (level_names or ())]
    needed_events = [f"mlp_prob__{n}" for n in event_names]
    if not refit and paths["predictions"].exists():
        cached = pd.read_parquet(paths["predictions"])
        has_level = any(c.startswith("mlp_raw__") for c in cached.columns)
        has_event = any(c.startswith("mlp_prob__") for c in cached.columns)
        levels_ok = all(c in cached.columns for c in needed_levels)
        events_ok = all(c in cached.columns for c in needed_events)
        manifest: dict[str, Any] = {}
        if paths["manifest"].exists():
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        pos_ok = baseline_cache_matches_pos_weight(manifest, pos_max)
        features_ok = baseline_cache_matches_features(manifest, allowed_feature_names)
        if (
            (has_level or has_event)
            and levels_ok
            and events_ok
            and pos_ok
            and features_ok
        ):
            print(f"Reusing cached tabular MLP: {paths['predictions']}", flush=True)
            return cached
        if not pos_ok:
            print(
                f"Cached MLP outdated (binary_pos_weight_max={pos_max}); refitting...",
                flush=True,
            )
        elif not features_ok:
            print("Cached MLP outdated (ModelSpec feature restriction); refitting...", flush=True)
        else:
            print("Cached MLP tabular file incomplete; refitting...", flush=True)

    print(f"Fitting tabular MLP baselines -> {paths['predictions']}", flush=True)
    levels = fit_mlp_level_baselines(
        train,
        test,
        preprocessing,
        seed,
        level_names=level_names,
        allowed_feature_names=allowed_feature_names,
    )
    events = fit_mlp_event_baselines(
        train,
        test,
        seed,
        event_names,
        binary_pos_weight_max=pos_max,
        allowed_feature_names=allowed_feature_names,
    )
    keys = ["person_id", "wave", "next_wave"]
    if levels.empty and events.empty:
        return pd.DataFrame()
    if levels.empty:
        out = events.copy()
    elif events.empty:
        out = levels.copy()
    else:
        out = levels.merge(events, on=keys, how="outer")
    out.to_parquet(paths["predictions"], index=False)
    paths["manifest"].write_text(
        json.dumps(
            {
                "kind": "tabular_mlp",
                "seed": seed,
                "event_names": list(event_names),
                "level_columns": sorted(c for c in out.columns if c.startswith("mlp_raw__")),
                "prob_columns": sorted(c for c in out.columns if c.startswith("mlp_prob__")),
                "binary_pos_weight_max": pos_max,
                "class_weight_scheme": "neg_over_pos_capped_row_expand",
                "n_rows": int(len(out)),
                "framework": "sklearn MLP on state__/action__ features (same as Linear/Logistic)",
                "allowed_feature_names": (
                    list(allowed_feature_names)
                    if allowed_feature_names is not None
                    else None
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out


def mlp_horizon_star_cache_paths(baseline_dir: Path) -> dict[str, Path]:
    baseline_dir = Path(baseline_dir)
    return {
        "predictions": baseline_dir / "mlp_horizon_star_predictions.parquet",
        "manifest": baseline_dir / "mlp_horizon_star_manifest.json",
    }


def build_mlp_horizon_star_rows(
    table: pd.DataFrame,
    event_names: Sequence[str],
    *,
    max_horizon: int = 5,
    allowed_feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Origin features at t0 and binary outcome label at open-loop horizon h."""
    if table.empty:
        return pd.DataFrame()
    features = baseline_feature_columns(
        table.columns, allowed_names=allowed_feature_names
    )
    if not features:
        return pd.DataFrame()
    pooled = resolve_pooled_binary_rewards(event_names)
    wide = materialize_pooled_binary_rewards(table.copy(), pooled)
    wide["person_id"] = wide["person_id"].astype(str)
    wide = wide.sort_values(["person_id", "wave"], kind="stable")
    rows: list[dict[str, Any]] = []
    for pid, grp in wide.groupby("person_id", sort=False):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        for origin_idx in range(n):
            origin = grp.iloc[origin_idx]
            base = {
                "person_id": str(pid),
                "wave": int(origin["wave"]),
                "next_wave": int(origin["next_wave"]),
            }
            for feat in features:
                base[feat] = origin[feat]
            for h in range(1, int(max_horizon) + 1):
                target_idx = origin_idx + h - 1
                if target_idx >= n:
                    break
                target = grp.iloc[target_idx]
                rec = dict(base)
                rec["horizon"] = int(h)
                for name in event_names:
                    val_col = f"reward__{name}"
                    mask_col = f"reward_mask__{name}"
                    if val_col not in target.index:
                        rec[f"y__{name}"] = np.nan
                        continue
                    mask = (
                        float(pd.to_numeric(target.get(mask_col, 1), errors="coerce"))
                        > 0.5
                    )
                    val = pd.to_numeric(target[val_col], errors="coerce")
                    if mask and np.isfinite(val):
                        rec[f"y__{name}"] = float(val > 0.5)
                    else:
                        rec[f"y__{name}"] = np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def fit_mlp_horizon_star_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
    event_names: Sequence[str],
    max_horizon: int = 5,
    binary_pos_weight_max: float = DEFAULT_BINARY_POS_WEIGHT_MAX,
    allowed_feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Tabular MLP per (horizon, event): P(Y at t+h | features at origin t)."""
    from sklearn.neural_network import MLPClassifier

    train_rows = build_mlp_horizon_star_rows(
        train,
        event_names,
        max_horizon=max_horizon,
        allowed_feature_names=allowed_feature_names,
    )
    test_rows = build_mlp_horizon_star_rows(
        test,
        event_names,
        max_horizon=max_horizon,
        allowed_feature_names=allowed_feature_names,
    )
    if train_rows.empty or test_rows.empty:
        return pd.DataFrame()
    features = baseline_feature_columns(
        train.columns, allowed_names=allowed_feature_names
    )
    if not features:
        return pd.DataFrame()
    x_test_all = np.nan_to_num(
        test_rows[features].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    out = test_rows[["person_id", "wave", "next_wave", "horizon"]].copy()
    out["person_id"] = out["person_id"].astype(str)
    for name in event_names:
        out[f"mlp_hstar_prob__{name}"] = np.nan
    for h in range(1, int(max_horizon) + 1):
        tr_h = train_rows[train_rows["horizon"].eq(h)]
        te_mask = test_rows["horizon"].eq(h).to_numpy()
        if not te_mask.any():
            continue
        x_test = x_test_all[te_mask]
        for name in event_names:
            y_col = f"y__{name}"
            if y_col not in tr_h.columns:
                continue
            y_raw = pd.to_numeric(tr_h[y_col], errors="coerce")
            active = y_raw.notna().to_numpy()
            if int(active.sum()) < 50:
                continue
            y = (y_raw[active].to_numpy(float) > 0.5).astype(int)
            if len(np.unique(y)) < 2 or int(np.bincount(y).min()) < 10:
                prev = float(y.mean()) if len(y) else 0.0
                out.loc[te_mask, f"mlp_hstar_prob__{name}"] = prev
                continue
            x_train = np.nan_to_num(
                tr_h.loc[active, features].to_numpy(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            x_fit, y_fit, _ = expand_binary_by_pos_weight(
                x_train,
                y,
                max_weight=binary_pos_weight_max,
                seed=seed + h * 17 + sum(ord(c) for c in name) % 997,
            )
            clf = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=80,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8,
                random_state=seed + h * 31 + sum(ord(c) for c in name) % 997,
            )
            clf.fit(x_fit, y_fit)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(x_test)[:, 1].astype(np.float32)
            else:
                probs = clf.predict(x_test).astype(np.float32)
            out.loc[te_mask, f"mlp_hstar_prob__{name}"] = probs
    prob_cols = [c for c in out.columns if c.startswith("mlp_hstar_prob__")]
    if not prob_cols:
        return pd.DataFrame()
    keep = out[prob_cols].notna().any(axis=1)
    return out.loc[keep].reset_index(drop=True)


def load_or_fit_mlp_horizon_star(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline_dir: Path,
    seed: int,
    event_names: Sequence[str],
    max_horizon: int = 5,
    refit: bool,
    skip: bool,
    binary_pos_weight_max: float = DEFAULT_BINARY_POS_WEIGHT_MAX,
    allowed_feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fit/cache MLP-h* (direct t0→t+h) event probabilities on test origins."""
    if skip:
        return pd.DataFrame()
    paths = mlp_horizon_star_cache_paths(baseline_dir)
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    pos_max = float(binary_pos_weight_max)
    needed = [f"mlp_hstar_prob__{n}" for n in event_names]
    if not refit and paths["predictions"].exists():
        cached = pd.read_parquet(paths["predictions"])
        manifest: dict[str, Any] = {}
        if paths["manifest"].exists():
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        events_ok = all(c in cached.columns for c in needed)
        pos_ok = baseline_cache_matches_pos_weight(manifest, pos_max)
        features_ok = baseline_cache_matches_features(manifest, allowed_feature_names)
        horizon_ok = int(manifest.get("max_horizon", 0)) >= int(max_horizon)
        if events_ok and pos_ok and features_ok and horizon_ok:
            print(f"Reusing cached MLP-h*: {paths['predictions']}", flush=True)
            return cached
    print(f"Fitting MLP-h* baselines -> {paths['predictions']}", flush=True)
    out = fit_mlp_horizon_star_baselines(
        train,
        test,
        seed=seed,
        event_names=event_names,
        max_horizon=max_horizon,
        binary_pos_weight_max=pos_max,
        allowed_feature_names=allowed_feature_names,
    )
    if out.empty:
        return out
    out.to_parquet(paths["predictions"], index=False)
    paths["manifest"].write_text(
        json.dumps(
            {
                "kind": "mlp_horizon_star",
                "seed": seed,
                "event_names": list(event_names),
                "max_horizon": int(max_horizon),
                "prob_columns": sorted(
                    c for c in out.columns if c.startswith("mlp_hstar_prob__")
                ),
                "binary_pos_weight_max": pos_max,
                "n_rows": int(len(out)),
                "note": "One MLP per (horizon, event): P(event at t+h | state/action at origin t)",
                "allowed_feature_names": (
                    list(allowed_feature_names)
                    if allowed_feature_names is not None
                    else None
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out


def attach_mlp_horizon_star_probs(
    rollout: pd.DataFrame,
    predictions: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add ``mlp_hstar_prob`` to open-loop rollout rows."""
    if rollout.empty:
        return rollout
    out = rollout.copy()
    out["mlp_hstar_prob"] = np.nan
    if predictions is None or predictions.empty:
        return out
    pred = predictions.copy()
    pred["person_id"] = pred["person_id"].astype(str)
    lookup: dict[tuple[str, int, int, int], dict[str, float]] = {}
    prob_cols = [c for c in pred.columns if c.startswith("mlp_hstar_prob__")]
    for _, row in pred.iterrows():
        key = (
            str(row["person_id"]),
            int(row["wave"]),
            int(row["next_wave"]),
            int(row["horizon"]),
        )
        lookup[key] = {c.replace("mlp_hstar_prob__", ""): float(row[c]) for c in prob_cols}
    probs: list[float] = []
    for _, row in out.iterrows():
        key = (
            str(row["person_id"]),
            int(row["start_wave"]),
            int(row["start_next_wave"]),
            int(row["horizon"]),
        )
        ev = str(row["event"])
        probs.append(float(lookup.get(key, {}).get(ev, np.nan)))
    out["mlp_hstar_prob"] = probs
    return out


def mlp_events_long_from_wide(
    mlp_wide: pd.DataFrame,
    events_jepa: pd.DataFrame,
) -> pd.DataFrame:
    """Wide ``mlp_prob__*`` → long frame for ``clinical_compare_table``."""
    if mlp_wide.empty or events_jepa.empty:
        return pd.DataFrame()
    keys = ["person_id", "wave", "next_wave"]
    chunks: list[pd.DataFrame] = []
    wide = mlp_wide.copy()
    wide["person_id"] = wide["person_id"].astype(str)
    for event, group in events_jepa.groupby("event", sort=False):
        col = f"mlp_prob__{event}"
        if col not in wide.columns:
            continue
        use = wide[keys + [col]].copy()
        merged = group[keys].assign(event=event).merge(use, on=keys, how="left")
        merged = merged.rename(columns={col: "mlp_prob"})
        chunks.append(merged)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


@torch.inference_mode()
def collect_mlp_level_event_predictions(
    mlp: MLPWorldBaseline,
    loader: DataLoader,
    device: torch.device,
    spec: ModelSpec,
    preprocessing: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reward_meta = preprocessing.get("reward_continuous", {})
    n_bin = len(spec.reward_binary)
    level_rows: list[pd.DataFrame] = []
    event_rows: list[pd.DataFrame] = []
    for raw in loader:
        batch = raw.to(device)
        _, steps = batch.valid.shape
        for t in range(steps):
            valid = batch.valid[:, t] > 0.5
            if not valid.any():
                continue
            x = _flat_features(batch, t)
            z = mlp.encode(x)
            z_next = mlp.predict_latent(z, batch.delta_t_norm[:, t])
            logits, cont = mlp.clinical(z_next)
            active_np = valid.detach().cpu().numpy()
            keys = pd.DataFrame(
                {
                    "person_id": np.asarray(batch.person_id, dtype=object)[active_np].astype(str),
                    "wave": batch.wave[:, t].detach().cpu().numpy()[active_np].astype(int),
                    "next_wave": batch.next_wave[:, t].detach().cpu().numpy()[active_np].astype(int),
                }
            )
            if cont is not None:
                for idx, name in enumerate(spec.reward_continuous):
                    if name not in LEVEL_OUTCOMES or name not in reward_meta:
                        continue
                    ridx = n_bin + idx
                    m = (batch.reward_mask[:, t, ridx] > 0.5) & valid
                    m_np = m.detach().cpu().numpy()
                    if not m_np.any():
                        continue
                    obs_std = batch.reward[:, t, ridx].detach().cpu().numpy()[m_np]
                    pred_std = cont[:, idx].detach().cpu().numpy()[m_np]
                    info = reward_meta[name]
                    chunk = pd.DataFrame(
                        {
                            "person_id": np.asarray(batch.person_id, dtype=object)[m_np].astype(str),
                            "wave": batch.wave[:, t].detach().cpu().numpy()[m_np].astype(int),
                            "next_wave": batch.next_wave[:, t]
                            .detach()
                            .cpu()
                            .numpy()[m_np]
                            .astype(int),
                            "outcome": name,
                            "label": LEVEL_OUTCOMES[name],
                            "observed_raw": inverse_standardized(obs_std, info),
                            "mlp_raw": inverse_standardized(pred_std, info),
                        }
                    )
                    level_rows.append(chunk)
            if logits is not None:
                probs = torch.sigmoid(logits)
                for idx, name in enumerate(spec.reward_binary):
                    ridx = idx
                    m = (batch.reward_mask[:, t, ridx] > 0.5) & valid
                    m_np = m.detach().cpu().numpy()
                    if not m_np.any():
                        continue
                    event_rows.append(
                        pd.DataFrame(
                            {
                                "person_id": np.asarray(batch.person_id, dtype=object)[
                                    m_np
                                ].astype(str),
                                "wave": batch.wave[:, t]
                                .detach()
                                .cpu()
                                .numpy()[m_np]
                                .astype(int),
                                "next_wave": batch.next_wave[:, t]
                                .detach()
                                .cpu()
                                .numpy()[m_np]
                                .astype(int),
                                "event": name,
                                "label": EVENT_LABELS.get(name, name),
                                "observed": batch.reward[:, t, ridx]
                                .detach()
                                .cpu()
                                .numpy()[m_np],
                                "mlp_prob": probs[:, idx].detach().cpu().numpy()[m_np],
                            }
                        )
                    )
            _ = keys
    levels = pd.concat(level_rows, ignore_index=True) if level_rows else pd.DataFrame()
    events = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    return levels, events


@torch.inference_mode()
def collect_jepa_level_rollouts(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    max_horizon: int = 5,
) -> pd.DataFrame:
    """Open-loop residual levels vs frozen origin (Persistence) over h=1..H.

    Only scores steps that are actually observed (``valid[t]`` and reward mask);
    no padded short trajectories. Persistence copies the origin mapped state.
    """
    world = agent.world
    names = list(agent.spec.reward_continuous)
    if not names:
        return pd.DataFrame()
    n_bin = len(agent.spec.reward_binary)
    cont_lookup = {name: n_bin + idx for idx, name in enumerate(names)}
    cont_names = list(agent.spec.state_continuous)
    std_info: dict[str, dict[str, Any]] = {}
    state_map: dict[str, str | None] = {}
    for name in names:
        state, info = continuous_reward_standardize_info(
            name, preprocessing, agent.spec.state_continuous
        )
        std_info[name] = dict(info or {})
        state_map[name] = state
    rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        batch_size, steps = batch.valid.shape
        use_steps = min(steps, max_horizon)
        z_seq = encode_batch_trajectory(world, batch)
        z = z_seq[:, 0]
        static_embed = encode_batch_static(world, batch, t=0)
        elapsed = torch.zeros(batch_size, device=device)
        cur_levels, cur_mask = world.clinical_heads._current_levels_for_rewards(
            batch.state_cont[:, 0], batch.state_cont_mask[:, 0]
        )
        origin_state = batch.state_cont[:, 0].detach().cpu().numpy()
        origin_mask = batch.state_cont_mask[:, 0].detach().cpu().numpy()
        for t in range(use_steps):
            valid = batch.valid[:, t] > 0.5
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
                current_levels=cur_levels,
                current_levels_mask=cur_mask,
            )
            if world.clinical_heads.continuous is not None:
                stacked = torch.stack(
                    [reward_pred[n] for n in names], dim=-1
                )
                cur_levels = stacked
                cur_mask = cur_mask.new_ones(cur_mask.shape)
            elapsed = elapsed + batch.delta_t_raw[:, t, 0]
            z = feature
            active = valid.detach().cpu().numpy()
            if not active.any():
                continue
            pids = np.asarray(batch.person_id, dtype=object)
            for name in names:
                if name not in reward_pred:
                    continue
                idx = cont_lookup[name]
                m = active & (batch.reward_mask[:, t, idx].detach().cpu().numpy() > 0.5)
                state_name = state_map[name]
                pers_std = None
                if state_name and state_name in cont_names:
                    j = cont_names.index(state_name)
                    m = m & (origin_mask[:, j] > 0.5)
                    pers_std = origin_state[:, j]
                if not m.any():
                    continue
                info = std_info[name]
                obs_std = batch.reward[:, t, idx].detach().cpu().numpy()[m]
                pred_std = reward_pred[name].detach().cpu().numpy()[m]
                if info:
                    obs_raw = inverse_standardized(obs_std, info)
                    pred_raw = inverse_standardized(pred_std, info)
                    pers_raw = (
                        inverse_standardized(pers_std[m], info)
                        if pers_std is not None
                        else np.full(m.sum(), np.nan)
                    )
                else:
                    obs_raw = obs_std
                    pred_raw = pred_std
                    pers_raw = pers_std[m] if pers_std is not None else np.full(m.sum(), np.nan)
                for i_local, i in enumerate(np.where(m)[0]):
                    rows.append(
                        {
                            "person_id": str(pids[i]),
                            "horizon": t + 1,
                            "elapsed_years": float(elapsed[i].cpu()),
                            "outcome": name,
                            "label": LEVEL_OUTCOMES.get(name, name),
                            "observed_raw": float(obs_raw[i_local]),
                            "jepa_raw": float(pred_raw[i_local]),
                            "persistence_raw": float(pers_raw[i_local]),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_level_openloop_mse(rollout: pd.DataFrame) -> pd.DataFrame:
    """Per-outcome open-loop MSE vs Persistence; ``relative_mse`` < 1 beats copy-last."""
    if rollout.empty:
        return pd.DataFrame()
    need = {"outcome", "horizon", "observed_raw", "jepa_raw", "persistence_raw"}
    if not need.issubset(rollout.columns):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (outcome, horizon), group in rollout.groupby(["outcome", "horizon"], sort=True):
        obs = pd.to_numeric(group["observed_raw"], errors="coerce").to_numpy(float)
        jepa = pd.to_numeric(group["jepa_raw"], errors="coerce").to_numpy(float)
        pers = pd.to_numeric(group["persistence_raw"], errors="coerce").to_numpy(float)
        mask = np.isfinite(obs) & np.isfinite(jepa) & np.isfinite(pers)
        if int(mask.sum()) < 20:
            continue
        jepa_mse = float(np.mean((jepa[mask] - obs[mask]) ** 2))
        pers_mse = float(np.mean((pers[mask] - obs[mask]) ** 2))
        label = (
            str(group["label"].iloc[0])
            if "label" in group.columns
            else LEVEL_OUTCOMES.get(str(outcome), str(outcome))
        )
        rows.append(
            {
                "outcome": str(outcome),
                "label": label,
                "horizon": int(horizon),
                "jepa_mse": jepa_mse,
                "persistence_mse": pers_mse,
                "relative_mse": (
                    jepa_mse / pers_mse if pers_mse > 0 else float("nan")
                ),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


@torch.inference_mode()
def collect_latent_rollout_mse(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    max_horizon: int = 5,
) -> pd.DataFrame:
    world = agent.world
    rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        _, steps = batch.valid.shape
        use_steps = min(steps, max_horizon)
        z_seq = encode_batch_trajectory(world, batch)
        z = z_seq[:, 0]
        static_embed = encode_batch_static(world, batch, t=0)
        for t in range(use_steps):
            valid = batch.valid[:, t] > 0.5
            feature = world.predict_next(
                z,
                batch.action_cont[:, t],
                batch.action_cont_mask[:, t],
                batch.action_cat[:, t],
                batch.delta_t_norm[:, t],
                static_embed=static_embed,
            )
            target = encode_batch_target(world, batch, t) if t == 0 else encode_target_at_horizon(
                world, batch, 0, t + 1
            )
            mse = (feature - target).square().mean(dim=-1)
            active = valid.detach().cpu().numpy()
            for i in np.where(active)[0]:
                rows.append({"horizon": t + 1, "latent_mse": float(mse[i].cpu())})
            z = feature
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .groupby("horizon", as_index=False)
        .agg(latent_mse=("latent_mse", "mean"), n=("latent_mse", "size"))
    )


# ---------------------------------------------------------------------------
# Fig 6 · single-person case study (world-model application)
# ---------------------------------------------------------------------------

FIG6_CASE_STUDY_OUTCOMES: tuple[str, ...] = ("death_event",)

FIG6_CASE_STUDY_MAX_HORIZON: int = 5

# Counterfactual actions match observed until this horizon, then switch once.
FIG6_CASE_STUDY_SWITCH_HORIZON: int = 3

FIG6_CASE_STUDY_AGE_MIN: float = 50.0
FIG6_CASE_STUDY_AGE_MAX: float = 65.0

# Fig 4 mid tier for light activity (1–2 days/week in HRS coding).
FIG6_CASE_STUDY_MODERATE_LIGHT_ACTIVITY_RAW: float = 2.0

FIG6_CASE_STUDY_SCENARIOS: tuple[tuple[str, str, dict[str, float]], ...] = (
    ("observed", "Observed actions", {}),
    ("reduce_smoking", "Reduced smoking (0 cigs/day)", {"cigarettes_per_day": 0.0}),
    (
        "moderate_activity",
        "Moderate light activity",
        {"light_activity_frequency": FIG6_CASE_STUDY_MODERATE_LIGHT_ACTIVITY_RAW},
    ),
    (
        "start_antihypertensive",
        "Antihypertensive started",
        {"hypertension_treatment": 1.0},
    ),
)

FIG6_CASE_STUDY_SCENARIO_LABELS: dict[str, str] = {
    key: label for key, label, _overrides in FIG6_CASE_STUDY_SCENARIOS
}

FIG6_CASE_STUDY_SCENARIO_ORDER: tuple[str, ...] = tuple(
    key for key, _label, _overrides in FIG6_CASE_STUDY_SCENARIOS
)

# Soft target for a "typical" mid-risk death profile at origin (h=1).
FIG6_CASE_STUDY_H1_TARGETS: dict[str, float] = {
    "death_event": 0.12,
}


def _cat_embed_for_raw(
    spec: ModelSpec,
    feature_name: str,
    raw_value: float,
    *,
    action: bool = True,
) -> int | None:
    """Map a raw categorical code to the model embedding index (1..K)."""
    feats = list(spec.action_categorical if action else spec.state_categorical)
    names = [f.name for f in feats]
    if feature_name not in names:
        return None
    feat = feats[names.index(feature_name)]
    mapping = {float(v): idx + 1 for idx, v in enumerate(feat.values)}
    return mapping.get(float(raw_value))


def _cat_raw_from_embed(
    spec: ModelSpec,
    feature_name: str,
    embed: int,
    *,
    action: bool = True,
) -> float | None:
    feats = list(spec.action_categorical if action else spec.state_categorical)
    names = [f.name for f in feats]
    if feature_name not in names:
        return None
    feat = feats[names.index(feature_name)]
    idx = int(embed)
    if idx <= 0 or idx > len(feat.values):
        return None
    return float(feat.values[idx - 1])


def _case_study_static_age_raw(
    batch: TrajectoryBatch,
    pi: int,
    spec: ModelSpec,
    preprocessing: Mapping[str, Any] | None,
) -> float | None:
    static_names = list(spec.static_continuous)
    if "age_years" not in static_names:
        return None
    j = static_names.index("age_years")
    if float(batch.static_cont_mask[pi, 0, j].detach().cpu()) <= 0.5:
        return None
    z = float(batch.static_cont[pi, 0, j].detach().cpu())
    if preprocessing:
        info = preprocessing.get("continuous", {}).get("age_years")
        if info:
            return float(inverse_standardized(np.array([z]), info)[0])
    return z


def _case_study_action_cont_raw(
    batch: TrajectoryBatch,
    pi: int,
    t: int,
    spec: ModelSpec,
    action_name: str,
    preprocessing: Mapping[str, Any] | None,
) -> float | None:
    resolved = _resolve_sim_action(spec, action_name)
    if resolved is None or resolved[1] != "cont":
        return None
    _name, _kind, a_idx, _cl, _ch = resolved
    if float(batch.action_cont_mask[pi, t, a_idx].detach().cpu()) <= 0.5:
        return None
    z = float(batch.action_cont[pi, t, a_idx].detach().cpu())
    if preprocessing:
        mean, std = _continuous_preprocessing_stats(preprocessing, action_name)
        return _cigs_z_to_raw(z, mean, std)
    return z


def _case_study_action_cat_raw(
    batch: TrajectoryBatch,
    pi: int,
    t: int,
    spec: ModelSpec,
    action_name: str,
) -> float | None:
    resolved = _resolve_sim_action(spec, action_name)
    if resolved is None or resolved[1] != "cat":
        return None
    _name, _kind, a_idx, _cl, _ch = resolved
    if float(batch.action_cat_mask[pi, t, a_idx].detach().cpu()) <= 0.5:
        return None
    embed = int(batch.action_cat[pi, t, a_idx].detach().cpu())
    return _cat_raw_from_embed(spec, action_name, embed, action=True)


def _case_study_state_cat_raw(
    batch: TrajectoryBatch,
    pi: int,
    t: int,
    spec: ModelSpec,
    state_name: str,
) -> float | None:
    names = [f.name for f in spec.state_categorical]
    if state_name not in names:
        return None
    idx = names.index(state_name)
    if float(batch.state_cat_mask[pi, t, idx].detach().cpu()) <= 0.5:
        return None
    embed = int(batch.state_cat[pi, t, idx].detach().cpu())
    return _cat_raw_from_embed(spec, state_name, embed, action=False)


def _case_study_baseline_profile(
    batch: TrajectoryBatch,
    pi: int,
    spec: ModelSpec,
    preprocessing: Mapping[str, Any] | None = None,
    *,
    t: int = 0,
) -> dict[str, float | None]:
    return {
        "baseline_age_years": _case_study_static_age_raw(batch, pi, spec, preprocessing),
        "baseline_hypertension_dx": _case_study_state_cat_raw(
            batch, pi, t, spec, "hypertension_dx"
        ),
        "baseline_hypertension_treatment": _case_study_action_cat_raw(
            batch, pi, t, spec, "hypertension_treatment"
        ),
        "baseline_cigarettes_per_day": _case_study_action_cont_raw(
            batch, pi, t, spec, "cigarettes_per_day", preprocessing
        ),
        "baseline_light_activity_frequency": _case_study_action_cat_raw(
            batch, pi, t, spec, "light_activity_frequency"
        ),
    }


def _case_study_valid_steps_from(
    batch: TrajectoryBatch,
    pi: int,
    t0: int,
) -> int:
    valid = batch.valid[pi].detach().cpu().numpy() > 0.5
    return int(valid[t0:].sum())


def _case_study_matches_target_profile(
    batch: TrajectoryBatch,
    pi: int,
    spec: ModelSpec,
    preprocessing: Mapping[str, Any] | None = None,
    *,
    t: int = 0,
) -> bool:
    profile = _case_study_baseline_profile(batch, pi, spec, preprocessing, t=t)
    age = profile.get("baseline_age_years")
    if age is None or not (FIG6_CASE_STUDY_AGE_MIN <= float(age) <= FIG6_CASE_STUDY_AGE_MAX):
        return False
    if profile.get("baseline_hypertension_dx") != 1.0:
        return False
    if profile.get("baseline_hypertension_treatment") != 0.0:
        return False
    cigs = profile.get("baseline_cigarettes_per_day")
    if cigs is None or float(cigs) <= PERSISTENT_SMOKING_ZERO_EPS:
        return False
    if profile.get("baseline_light_activity_frequency") != 0.0:
        return False
    return True


def _apply_case_study_intervention(
    batch: TrajectoryBatch,
    pi: int,
    t: int,
    overrides: Mapping[str, float],
    spec: ModelSpec,
    preprocessing: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return 1-person actions with optional single-action raw overrides."""
    action_cont = batch.action_cont[pi : pi + 1, t].clone()
    action_cat = batch.action_cat[pi : pi + 1, t].clone()
    action_cont_mask = batch.action_cont_mask[pi : pi + 1, t]
    action_cat_mask = batch.action_cat_mask[pi : pi + 1, t]
    if not overrides:
        return action_cont, action_cont_mask, action_cat, action_cat_mask
    for action_name, raw_target in overrides.items():
        resolved = _resolve_sim_action(spec, action_name)
        if resolved is None:
            continue
        _name, kind, a_idx, _cat_low, _cat_high = resolved
        if kind == "cont":
            mask = action_cont_mask[:, a_idx]
            if float(mask.sum().detach().cpu()) <= 0.0:
                continue
            if preprocessing:
                mean, std = _continuous_preprocessing_stats(preprocessing, action_name)
                target_z = _cigs_raw_to_z(float(raw_target), mean, std)
            else:
                target_z = float(raw_target)
            action_cont[:, a_idx] = torch.where(
                mask > 0.5,
                torch.full_like(action_cont[:, a_idx], target_z),
                action_cont[:, a_idx],
            )
        else:
            embed = _cat_embed_for_raw(spec, action_name, float(raw_target), action=True)
            if embed is None:
                continue
            mask = action_cat_mask[:, a_idx]
            tgt = torch.full_like(action_cat[:, a_idx], int(embed))
            action_cat[:, a_idx] = torch.where(mask > 0.5, tgt, action_cat[:, a_idx])
    return action_cont, action_cont_mask, action_cat, action_cat_mask


def _denorm_case_study_meta(
    meta: dict[str, Any],
    preprocessing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not preprocessing:
        return meta
    cont = preprocessing.get("continuous", {})
    if "age_years" in meta and "age_years" in cont:
        meta = dict(meta)
        meta["age_years"] = float(
            inverse_standardized(np.array([meta["age_years"]]), cont["age_years"])[0]
        )
    return meta


def _case_study_h1_probs(
    agent: JEPAAgent,
    batch: TrajectoryBatch,
    pi: int,
    outcomes: Sequence[str],
    *,
    t: int = 0,
) -> dict[str, float]:
    world = agent.world
    z_seq = encode_batch_trajectory(world, batch)
    static_embed = encode_batch_static(world, batch, t=t)
    feature = world.predict_next(
        z_seq[pi : pi + 1, t],
        batch.action_cont[pi : pi + 1, t],
        batch.action_cont_mask[pi : pi + 1, t],
        batch.action_cat[pi : pi + 1, t],
        batch.delta_t_norm[pi : pi + 1, t],
        static_embed=static_embed[pi : pi + 1],
    )
    pred = world.clinical_heads.mean_dict(
        feature,
        state_cont=batch.state_cont[pi : pi + 1, t],
        state_cont_mask=batch.state_cont_mask[pi : pi + 1, t],
    )
    lookup = {n: i for i, n in enumerate(agent.spec.reward_binary)}
    out: dict[str, float] = {}
    for name in outcomes:
        if name in pred:
            out[name] = float(pred[name][0].detach().cpu())
        elif name in lookup:
            out[name] = float("nan")
    return out


def _case_study_person_step_count(batch: TrajectoryBatch, pi: int) -> int:
    valid = batch.valid[pi].detach().cpu().numpy() > 0.5
    return int(valid.sum())


def _case_study_action_coverage(
    batch: TrajectoryBatch,
    pi: int,
    spec: ModelSpec,
    *,
    t: int = 0,
) -> int:
    action_names = (
        "light_activity_frequency",
        "cigarettes_per_day",
        "hypertension_treatment",
    )
    covered = 0
    for action_name in action_names:
        resolved = _resolve_sim_action(spec, action_name)
        if resolved is None:
            continue
        _name, kind, a_idx, _cl, _ch = resolved
        if kind == "cont":
            if float(batch.action_cont_mask[pi, t, a_idx].detach().cpu()) > 0.5:
                covered += 1
        elif float(batch.action_cat_mask[pi, t, a_idx].detach().cpu()) > 0.5:
            covered += 1
    return covered


def _case_study_typicality_score(h1: dict[str, float]) -> float:
    return -sum(
        (float(h1.get(name, 0.5)) - target) ** 2
        for name, target in FIG6_CASE_STUDY_H1_TARGETS.items()
    )


def _case_study_person_meta(
    batch: TrajectoryBatch,
    pi: int,
    spec: ModelSpec,
    preprocessing: Mapping[str, Any] | None = None,
    *,
    origin_step: int = 0,
) -> dict[str, Any]:
    t = int(origin_step)
    meta: dict[str, Any] = {
        "person_id": str(batch.person_id[pi]),
        "origin_step": t,
        "start_wave": int(batch.wave[pi, t].detach().cpu()),
        "start_next_wave": int(batch.next_wave[pi, t].detach().cpu()),
        "n_steps": _case_study_person_step_count(batch, pi),
        "n_steps_from_origin": _case_study_valid_steps_from(batch, pi, t),
    }
    static_names = list(spec.static_continuous)
    if static_names and batch.static_cont.shape[-1] >= len(static_names):
        for j, name in enumerate(static_names):
            if float(batch.static_cont_mask[pi, 0, j].detach().cpu()) > 0.5:
                meta[name] = float(batch.static_cont[pi, 0, j].detach().cpu())
    for idx, feat in enumerate(spec.static_categorical):
        if float(batch.static_cat_mask[pi, 0, idx].detach().cpu()) > 0.5:
            meta[feat.name] = float(batch.static_cat[pi, 0, idx].detach().cpu())
    meta.update(_case_study_baseline_profile(batch, pi, spec, preprocessing, t=t))
    return _denorm_case_study_meta(meta, preprocessing)


def _case_study_rollout_spread_score(
    rollout: pd.DataFrame,
    *,
    switch_horizon: int = FIG6_CASE_STUDY_SWITCH_HORIZON,
) -> float:
    """Higher = clearer open-loop separation after the switch horizon."""
    if rollout.empty or "risk_prob" not in rollout.columns:
        return 0.0
    frame = rollout.copy()
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    frame["risk_prob"] = pd.to_numeric(frame["risk_prob"], errors="coerce")
    post = frame[frame["horizon"] >= int(switch_horizon)].dropna(subset=["risk_prob"])
    if post.empty:
        return 0.0

    parts: list[float] = []
    for _h, grp in post.groupby("horizon"):
        vals = grp["risk_prob"].to_numpy(dtype=float)
        if vals.size >= 2:
            parts.append(float(vals.max() - vals.min()))

    obs = post[post["scenario"].astype(str).eq("observed")][["horizon", "risk_prob"]]
    obs_map = {
        int(h): float(p) for h, p in zip(obs["horizon"], obs["risk_prob"], strict=False)
    }
    for scenario in (
        "reduce_smoking",
        "moderate_activity",
        "start_antihypertensive",
    ):
        sub = post[post["scenario"].astype(str).eq(scenario)]
        for row in sub.itertuples(index=False):
            h = int(getattr(row, "horizon"))
            p = float(getattr(row, "risk_prob"))
            if h in obs_map:
                parts.append(abs(p - obs_map[h]))

    if not parts:
        return 0.0
    return float(np.mean(parts))


@torch.inference_mode()
def pick_case_study_person(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    *,
    outcomes: Sequence[str] | None = None,
    min_steps: int = FIG6_CASE_STUDY_SWITCH_HORIZON,
    person_id: str | None = None,
    exclude_person_id: str | None = None,
    seed: int = 2026,
    preprocessing: Mapping[str, Any] | None = None,
) -> tuple[TrajectoryBatch | None, int | None, pd.DataFrame, dict[str, Any]]:
    """Pick one HRS test person for Fig 6 with maximal post-switch counterfactual spread."""
    outcome_names = [
        n
        for n in (list(outcomes) if outcomes is not None else list(FIG6_CASE_STUDY_OUTCOMES))
        if n in agent.spec.reward_binary
    ]
    if not outcome_names:
        return None, None, pd.DataFrame(), {}

    if person_id is not None:
        target = str(person_id)
        for raw in loader:
            batch = raw.to(device)
            for pi, pid in enumerate(batch.person_id):
                if str(pid) != target:
                    continue
                n_steps = _case_study_person_step_count(batch, pi)
                best: tuple[float, int, dict[str, Any], TrajectoryBatch] | None = None
                for t0 in range(n_steps):
                    if float(batch.valid[pi, t0].detach().cpu()) <= 0.5:
                        continue
                    if _case_study_valid_steps_from(batch, pi, t0) < min_steps:
                        continue
                    meta = _case_study_person_meta(
                        batch, pi, agent.spec, preprocessing, origin_step=t0
                    )
                    meta["selection"] = "manual"
                    rollout = collect_case_study_rollout(
                        agent,
                        batch,
                        pi,
                        device,
                        origin_step=t0,
                        preprocessing=preprocessing,
                    )
                    spread = _case_study_rollout_spread_score(rollout)
                    meta["counterfactual_spread"] = spread
                    if best is None or spread > best[0]:
                        best = (spread, t0, meta, batch)
                if best is not None:
                    _spread, _t0, meta, batch = best
                    return batch, pi, pd.DataFrame([meta]), meta
        return None, None, pd.DataFrame(), {}

    lookup = {n: i for i, n in enumerate(agent.spec.reward_binary)}
    profile_hits: list[tuple[int, int, int, TrajectoryBatch, dict[str, Any]]] = []
    for bi, raw in enumerate(loader):
        batch = raw.to(device)
        batch_size = len(batch.person_id)
        for pi in range(batch_size):
            n_steps = _case_study_person_step_count(batch, pi)
            for t0 in range(n_steps):
                if float(batch.valid[pi, t0].detach().cpu()) <= 0.5:
                    continue
                if _case_study_valid_steps_from(batch, pi, t0) < min_steps:
                    continue
                covered_outcomes = 0
                for name in outcome_names:
                    ridx = lookup[name]
                    mask = (
                        batch.reward_mask[pi, t0:n_steps, ridx].detach().cpu().numpy() > 0.5
                    )
                    if mask.any():
                        covered_outcomes += 1
                if covered_outcomes < len(outcome_names):
                    continue
                if _case_study_action_coverage(batch, pi, agent.spec, t=t0) < 3:
                    continue
                if not _case_study_matches_target_profile(
                    batch, pi, agent.spec, preprocessing, t=t0
                ):
                    continue
                pid = str(batch.person_id[pi])
                if exclude_person_id is not None and pid == str(exclude_person_id):
                    continue
                h1 = _case_study_h1_probs(agent, batch, pi, outcome_names, t=t0)
                meta = _case_study_person_meta(
                    batch, pi, agent.spec, preprocessing, origin_step=t0
                )
                meta.update({f"h1_{k}": v for k, v in h1.items()})
                meta["selection"] = "auto"
                profile_hits.append((bi, pi, t0, batch, meta))

    if not profile_hits:
        return None, None, pd.DataFrame(), {}

    scored: list[tuple[float, int, int, TrajectoryBatch, dict[str, Any]]] = []
    for _bi, pi, t0, batch, meta in profile_hits:
        rollout = collect_case_study_rollout(
            agent,
            batch,
            pi,
            device,
            origin_step=t0,
            preprocessing=preprocessing,
        )
        spread = _case_study_rollout_spread_score(rollout)
        meta = dict(meta)
        meta["counterfactual_spread"] = spread
        scored.append((spread, pi, t0, batch, meta))

    scored.sort(
        key=lambda x: (
            -x[0],
            -_case_study_typicality_score(
                {k.removeprefix("h1_"): v for k, v in x[4].items() if str(k).startswith("h1_")}
            ),
            str(x[4].get("person_id", "")),
        )
    )
    _spread, pi, _t0, batch, meta = scored[0]
    return batch, pi, pd.DataFrame([meta]), meta


@torch.inference_mode()
def collect_case_study_rollout(
    agent: JEPAAgent,
    batch: TrajectoryBatch,
    person_index: int,
    device: torch.device,
    *,
    max_horizon: int = FIG6_CASE_STUDY_MAX_HORIZON,
    switch_horizon: int = FIG6_CASE_STUDY_SWITCH_HORIZON,
    origin_step: int = 0,
    outcomes: Sequence[str] | None = None,
    preprocessing: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Open-loop death risks using baseline (origin) actions/state for every horizon.

    Observed keeps baseline actions throughout. Counterfactuals match baseline until
    ``switch_horizon``, then override a single action while other inputs stay at baseline.
    Latent ``z`` is rolled forward open-loop via ``predict_next``; only the clinical head
    reads baseline ``state_cont`` at the origin step.
    """
    _ = device
    world = agent.world
    pi = int(person_index)
    t0 = int(origin_step)
    switch_horizon = max(int(switch_horizon), 1)
    outcome_names = [
        n
        for n in (list(outcomes) if outcomes is not None else list(FIG6_CASE_STUDY_OUTCOMES))
        if n in agent.spec.reward_binary
    ]
    if not outcome_names:
        return pd.DataFrame()

    use_steps = int(max_horizon)
    if use_steps <= 0:
        return pd.DataFrame()
    if _case_study_valid_steps_from(batch, pi, t0) < switch_horizon:
        return pd.DataFrame()

    z_seq = encode_batch_trajectory(world, batch)
    static_embed = encode_batch_static(world, batch, t=t0)
    pid = str(batch.person_id[pi])
    wave0 = int(batch.wave[pi, t0].detach().cpu())
    next0 = int(batch.next_wave[pi, t0].detach().cpu())
    wave_step = max(int(next0 - wave0), 1)

    rows: list[dict[str, Any]] = []
    for scenario, scenario_label, overrides in FIG6_CASE_STUDY_SCENARIOS:
        z = z_seq[pi : pi + 1, t0]
        static = static_embed[pi : pi + 1]
        for h in range(use_steps):
            horizon = h + 1
            active_overrides = overrides if horizon >= switch_horizon else {}
            action_cont, action_cont_mask, action_cat, action_cat_mask = (
                _apply_case_study_intervention(
                    batch,
                    pi,
                    t0,
                    active_overrides,
                    agent.spec,
                    preprocessing,
                )
            )
            feature = world.predict_next(
                z,
                action_cont,
                action_cont_mask,
                action_cat,
                batch.delta_t_norm[pi : pi + 1, t0],
                static_embed=static,
            )
            pred = world.clinical_heads.mean_dict(
                feature,
                state_cont=batch.state_cont[pi : pi + 1, t0],
                state_cont_mask=batch.state_cont_mask[pi : pi + 1, t0],
            )
            wave_t = wave0 + h * wave_step
            next_t = wave_t + wave_step
            for name in outcome_names:
                if name not in pred:
                    continue
                rows.append(
                    {
                        "person_id": pid,
                        "origin_step": t0,
                        "scenario": scenario,
                        "scenario_label": scenario_label,
                        "switch_horizon": switch_horizon,
                        "horizon": horizon,
                        "wave": wave_t,
                        "next_wave": next_t,
                        "start_wave": wave0,
                        "start_next_wave": next0,
                        "event": name,
                        "label": EVENT_LABELS.get(name, name),
                        "risk_prob": float(pred[name][0].detach().cpu()),
                    }
                )
            z = feature
    return pd.DataFrame(rows)


@torch.inference_mode()
def collect_one_step_person_trajectories(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    outcome: str | Sequence[str] = "adl_worsening",
    max_rows: int = 320_000,
) -> pd.DataFrame:
    """Real-wave one-step trajectories for Fig 3D (not open-loop horizons).

    For each valid transition ``t`` of a person::

        observed = next-wave Bernoulli reward (0/1)
        jepa     = ClinicalHeads P(event | Predictor(z_t, a_t, Δt))

    ``outcome`` accepts one name or several; all requested binary rewards
    are collected in a single loader pass and stacked long-form.
    """
    _ = preprocessing  # binary rewards are not inverse-standardized
    world = agent.world
    requested = [outcome] if isinstance(outcome, str) else list(outcome)
    lookup = {n: i for i, n in enumerate(agent.spec.reward_binary)}
    targets: list[dict[str, Any]] = []
    for name in requested:
        if name not in lookup:
            continue
        targets.append({"name": name, "ridx": lookup[name]})
    if not targets:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        _, steps = batch.valid.shape
        z_seq = encode_batch_trajectory(world, batch)
        static_embed = encode_batch_static(world, batch, t=0)
        for t in range(steps):
            valid = batch.valid[:, t] > 0.5
            if not valid.any():
                continue
            feature = world.predict_next(
                z_seq[:, t],
                batch.action_cont[:, t],
                batch.action_cont_mask[:, t],
                batch.action_cat[:, t],
                batch.delta_t_norm[:, t],
                static_embed=static_embed,
            )
            pred = world.clinical_heads.mean_dict(
                feature,
                state_cont=batch.state_cont[:, t],
                state_cont_mask=batch.state_cont_mask[:, t],
            )
            pids = np.asarray(batch.person_id, dtype=object)
            waves = batch.wave[:, t].detach().cpu().numpy().astype(int)
            next_waves = batch.next_wave[:, t].detach().cpu().numpy().astype(int)
            for target in targets:
                name = str(target["name"])
                ridx = int(target["ridx"])
                if name not in pred:
                    continue
                mask = (batch.reward_mask[:, t, ridx] > 0.5) & valid
                if not mask.any():
                    continue
                active = mask.detach().cpu().numpy()
                obs = batch.reward[:, t, ridx].detach().cpu().numpy()
                jepa = pred[name].detach().cpu().numpy()
                for i in np.where(active)[0]:
                    rows.append(
                        {
                            "person_id": str(pids[i]),
                            "wave": int(waves[i]),
                            "next_wave": int(next_waves[i]),
                            "step": t + 1,
                            "outcome": name,
                            "label": EVENT_LABELS.get(name, name),
                            "observed_raw": float(obs[i]),
                            "jepa_raw": float(jepa[i]),
                            "persistence_raw": float("nan"),
                        }
                    )
            # Truncate on whole timesteps so outcomes stay aligned per person.
            if len(rows) >= max_rows:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def pick_example_trajectories(
    trajectories: pd.DataFrame,
    outcome: str = "adl_worsening",
    max_persons: int = 4,
    seed: int = 2026,
    min_steps: int = 3,
    keep_all_outcomes: bool = False,
) -> pd.DataFrame:
    """Select a few people with enough steps and non-trivial ADL variation.

    ``outcome`` is the reference used for eligibility and variation ranking.
    With ``keep_all_outcomes`` the returned frame keeps every outcome for the
    chosen people, so one person is one colour across all Fig 3D panels.
    """
    if trajectories.empty:
        return pd.DataFrame()
    use = trajectories[trajectories["outcome"] == outcome].copy()
    if use.empty:
        use = trajectories.copy()
    if use.empty:
        return pd.DataFrame()
    time_col = "wave" if "wave" in use.columns else "horizon"
    counts = use.groupby("person_id")[time_col].nunique()
    eligible = counts[counts >= min(min_steps, int(counts.max()))].index.tolist()
    if not eligible:
        eligible = use["person_id"].unique().tolist()
    if keep_all_outcomes:
        # Prefer people observed on every outcome so no panel is left blank.
        n_outcomes = trajectories["outcome"].nunique()
        covered = trajectories.groupby("person_id")["outcome"].nunique()
        full = [pid for pid in eligible if int(covered.get(pid, 0)) >= n_outcomes]
        if full:
            eligible = full
    # Prefer people whose observed level actually changes (more interpretable).
    scored: list[tuple[float, str]] = []
    for pid in eligible:
        sub = use[use["person_id"] == pid]
        obs = pd.to_numeric(sub["observed_raw"], errors="coerce").to_numpy(float)
        span = float(np.nanmax(obs) - np.nanmin(obs)) if len(obs) else 0.0
        scored.append((span, str(pid)))
    scored.sort(key=lambda x: (-x[0], x[1]))
    varied = [pid for span, pid in scored if span > 1e-3]
    pool = varied if varied else [pid for _, pid in scored]
    rng = np.random.default_rng(seed)
    # Keep top varying candidates, then sample for diversity.
    top = pool[: max(max_persons * 4, max_persons)]
    chosen = list(rng.choice(top, size=min(max_persons, len(top)), replace=False))
    source = trajectories if keep_all_outcomes else use
    out = source[source["person_id"].isin(chosen)].copy()
    sort_cols = [
        c for c in ("outcome", "person_id", "wave", "step", "horizon") if c in out.columns
    ]
    return out.sort_values(sort_cols, kind="stable")


# ---------------------------------------------------------------------------
# Fig 4 collectors
# ---------------------------------------------------------------------------


def clinical_compare_table(
    events_jepa: pd.DataFrame,
    events_mlp: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """AUROC / AUPRC / Brier / thresholded rates for Prevalence vs Logistic vs MLP vs JEPA."""
    if events_jepa.empty:
        return pd.DataFrame()
    keys = ["person_id", "wave", "next_wave", "event"]
    frame = events_jepa.rename(columns={"predicted_prob": "jepa_prob"}).copy()
    if not events_mlp.empty:
        frame = frame.merge(
            events_mlp[keys + ["mlp_prob"]],
            on=keys,
            how="left",
        )
    else:
        frame["mlp_prob"] = np.nan
    # Attach logistic baseline
    chunks = []
    for name, group in frame.groupby("event", sort=False):
        col = f"baseline_prob__{name}"
        use = baseline[["person_id", "wave", "next_wave"] + ([col] if col in baseline.columns else [])].copy()
        use["person_id"] = use["person_id"].astype(str)
        if col not in use:
            use[col] = float(group["observed"].mean())
        merged = group.merge(use, on=["person_id", "wave", "next_wave"], how="left")
        merged = merged.rename(columns={col: "logistic_prob"})
        chunks.append(merged)
    frame = pd.concat(chunks, ignore_index=True) if chunks else frame

    rows = []
    for name, group in frame.groupby("event", sort=False):
        y = group["observed"].to_numpy(int)
        prev = float(y.mean()) if len(y) else 0.0
        # Constant prevalence predictor: AUPRC ≈ prevalence, AUROC = 0.5.
        p_prev = np.full(len(y), prev, dtype=np.float64)
        model_cols = [
            ("Prevalence", None),
            ("Logistic", "logistic_prob"),
            ("MLP", "mlp_prob"),
            ("JEPA", "jepa_prob"),
        ]
        for model, col in model_cols:
            if model == "Prevalence":
                p = p_prev
            else:
                if col not in group or group[col].isna().all():
                    continue
                p = group[col].to_numpy(float)
            rates = binary_rates_at_threshold(y, p, threshold=0.5)
            rows.append(
                {
                    "event": name,
                    "label": EVENT_LABELS.get(str(name), str(name)),
                    "model": model,
                    "auroc": 0.5 if model == "Prevalence" else safe_auroc(p, y),
                    "auprc": safe_auprc(y, p),
                    "brier": brier(y, p),
                    "balanced_accuracy": float(rates["balanced_accuracy"]),
                    "sensitivity": float(rates["sensitivity"]),
                    "specificity": float(rates["specificity"]),
                    "n": len(y),
                    "events": int(y.sum()),
                    "prevalence": prev,
                }
            )
    return pd.DataFrame(rows)


def calibration_curve_frame(
    events: pd.DataFrame,
    prob_col: str = "predicted_prob",
    bins: int = 10,
) -> pd.DataFrame:
    if events.empty or prob_col not in events.columns:
        return pd.DataFrame()
    rows = []
    for name, group in events.groupby("event", sort=False):
        use = group.dropna(subset=[prob_col, "observed"]).copy()
        if len(use) < 50:
            continue
        use["bin"] = pd.qcut(use[prob_col], q=bins, labels=False, duplicates="drop")
        agg = use.groupby("bin", as_index=False).agg(
            mean_pred=(prob_col, "mean"),
            mean_obs=("observed", "mean"),
            n=("observed", "size"),
        )
        agg["event"] = name
        agg["label"] = EVENT_LABELS.get(str(name), str(name))
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def risk_stratification_frame(events: pd.DataFrame, event: str = "death_event") -> pd.DataFrame:
    use = events[events["event"] == event].copy() if not events.empty else pd.DataFrame()
    if use.empty or "predicted_prob" not in use.columns:
        return pd.DataFrame()
    use["quintile"] = pd.qcut(
        use["predicted_prob"].rank(method="first"),
        q=5,
        labels=False,
        duplicates="drop",
    ) + 1
    return use.groupby("quintile", as_index=False).agg(
        event_rate=("observed", "mean"),
        n=("observed", "size"),
        mean_risk=("predicted_prob", "mean"),
    )


def kaplan_meier_by_risk(
    events: pd.DataFrame,
    event: str = "death_event",
) -> pd.DataFrame:
    """Discrete-time KM-style survival by risk tertile (wave as time)."""
    use = events[events["event"] == event].copy() if not events.empty else pd.DataFrame()
    if use.empty:
        return pd.DataFrame()
    # Person-level: first observed transition risk + whether event ever occurs.
    person = (
        use.sort_values(["person_id", "wave"])
        .groupby("person_id", as_index=False)
        .agg(
            risk=("predicted_prob", "first"),
            event=("observed", "max"),
            time=("next_wave", "max"),
        )
    )
    if len(person) < 30:
        return pd.DataFrame()
    try:
        person["stratum"] = pd.qcut(person["risk"], q=3, labels=["low", "mid", "high"])
    except ValueError:
        return pd.DataFrame()
    rows = []
    for stratum, group in person.groupby("stratum", observed=True):
        g = group.sort_values("time")
        n = len(g)
        alive = n
        cum_surv = 1.0
        for t, chunk in g.groupby("time", sort=True):
            d = int(chunk["event"].sum())
            if alive <= 0:
                break
            cum_surv *= 1.0 - d / alive
            rows.append(
                {
                    "stratum": str(stratum),
                    "time": int(t),
                    "survival": cum_surv,
                    "at_risk": alive,
                    "events": d,
                }
            )
            alive -= len(chunk)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fig 5 collectors
# ---------------------------------------------------------------------------

# Active Fig 4 interventions (computed + plotted).
FIG5_SIMULATION_ACTIONS: tuple[str, ...] = FIG4_ACTION_ORDER

# Full Fig 5/6 intervention order when all actions are enabled. Continuous: ±1
# standardized units. Categorical: min / max class (activity 0=never … 4=most
# frequent; meds 0=off / 1=on). Only overwrite steps where mask is 1.
FIG5_ACTION_ORDER = [
    "cigarettes_per_day",
    "drinks_per_drinking_day",
    "alcohol_days_per_week",
    "vigorous_activity_frequency",
    "moderate_activity_frequency",
    "light_activity_frequency",
    "hypertension_treatment",
    "diabetes_oral_medication",
]
# Shown in Fig 5/6 neither as rows nor in simulation cache manifest.
FIG5_EXCLUDED_ACTIONS: tuple[str, ...] = ()


def filter_fig5_actions(actions: Sequence[str]) -> list[str]:
    skip = set(FIG5_EXCLUDED_ACTIONS)
    return [a for a in actions if a not in skip]


def resolve_fig5_simulation_actions(available_actions: Sequence[str]) -> list[str]:
    """Fig 4 actions to simulate and plot (subset of ``FIG4_ACTION_ORDER``)."""
    known = set(available_actions)
    return filter_fig5_actions([a for a in FIG4_ACTION_ORDER if a in known])


def _resolve_sim_action(
    spec: ModelSpec,
    action_name: str | None,
) -> tuple[str, str, int, int, int] | None:
    """Return ``(name, kind, index, cat_low_embed, cat_high_embed)`` or None.

    ``kind`` is ``cont`` or ``cat``. Embedding index 0 is reserved for missing;
    valid classes are 1..K in ``feature.values`` order.
    """
    cont = [a.name for a in spec.action_continuous]
    cat_feats = list(spec.action_categorical)
    cat = [a.name for a in cat_feats]
    known = set(cont) | set(cat)
    name = action_name if action_name in known else None
    if name is None:
        name = next((a for a in FIG5_ACTION_ORDER if a in known), None)
        if name is None:
            name = (cont + cat)[0] if (cont or cat) else None
    if name is None:
        return None
    if name in cont:
        return name, "cont", cont.index(name), 0, 0
    feat = cat_feats[cat.index(name)]
    vals = [float(v) for v in feat.values]
    low_code, high_code = min(vals), max(vals)
    return (
        name,
        "cat",
        cat.index(name),
        vals.index(low_code) + 1,
        vals.index(high_code) + 1,
    )


def _apply_sim_action(
    batch: TrajectoryBatch,
    t: int,
    *,
    kind: str,
    index: int,
    scale: float,
    cat_low_idx: int,
    cat_high_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return intervened ``(action_cont, action_cat)`` at step ``t``.

    ``scale`` < 0 → low, 0 → observed, > 0 → high. Categorical interventions
    only overwrite rows whose action mask at ``t`` is observed.
    """
    action_cont = batch.action_cont[:, t].clone()
    action_cat = batch.action_cat[:, t].clone()
    if abs(float(scale)) < 1e-12:
        return action_cont, action_cat
    if kind == "cont":
        mask = batch.action_cont_mask[:, t, index]
        action_cont[:, index] = action_cont[:, index] + float(scale) * mask
        return action_cont, action_cat
    mask = batch.action_cat_mask[:, t, index]
    target = int(cat_low_idx if scale < 0 else cat_high_idx)
    tgt = torch.full(
        action_cat[:, index].shape,
        target,
        dtype=action_cat.dtype,
        device=action_cat.device,
    )
    action_cat[:, index] = torch.where(mask > 0.5, tgt, action_cat[:, index])
    return action_cont, action_cat


def _gather_start_batches(
    loader: DataLoader,
    device: torch.device,
    n_persons: int,
) -> list[TrajectoryBatch]:
    starts: list[TrajectoryBatch] = []
    seen = 0
    for raw in loader:
        starts.append(raw.to(device))
        seen += int(raw.valid[:, 0].sum().item())
        if seen >= n_persons:
            break
    return starts


@torch.inference_mode()
def collect_action_simulation(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    action_name: str | None = None,
    horizons: int = 5,
    n_persons: int = 256,
    outcome_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Perturb one action; mean P(event) / latent trajectories.

    Continuous actions: ±1 in standardized units (mask-gated).
    Categorical actions: set to min / max class (mask-gated).

    ``outcome_names`` defaults to ``adl_worsening``. Returns long-form levels with
    columns ``event`` and ``event_prob`` (plus ``adl_pred`` alias when event is ADL).
    """
    world = agent.world
    resolved = _resolve_sim_action(agent.spec, action_name)
    if resolved is None:
        return pd.DataFrame(), pd.DataFrame(), None
    action_name, kind, a_idx, cat_low, cat_high = resolved
    _ = preprocessing
    if outcome_names is None:
        outcomes = ["adl_worsening"]
    else:
        outcomes = [n for n in outcome_names if n in agent.spec.reward_binary]
    if not outcomes:
        return pd.DataFrame(), pd.DataFrame(), action_name

    starts = _gather_start_batches(loader, device, n_persons)
    if not starts:
        return pd.DataFrame(), pd.DataFrame(), action_name

    scales = (-1.0, 0.0, 1.0)
    level_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []

    for scale in scales:
        label = {-1.0: "low", 0.0: "baseline", 1.0: "high"}[scale]
        for batch in starts:
            steps = batch.valid.shape[1]
            use_steps = min(steps, horizons)
            z_seq = encode_batch_trajectory(world, batch)
            z = z_seq[:, 0]
            static_embed = encode_batch_static(world, batch, t=0)
            for t in range(use_steps):
                action_cont, action_cat = _apply_sim_action(
                    batch,
                    t,
                    kind=kind,
                    index=a_idx,
                    scale=scale,
                    cat_low_idx=cat_low,
                    cat_high_idx=cat_high,
                )
                feature = world.predict_next(
                    z,
                    action_cont,
                    batch.action_cont_mask[:, t],
                    action_cat,
                    batch.delta_t_norm[:, t],
                    static_embed=static_embed,
                )
                pred = world.clinical_heads.mean_dict(
                    feature,
                    state_cont=batch.state_cont[:, t],
                    state_cont_mask=batch.state_cont_mask[:, t],
                )
                valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
                z_np = feature.detach().cpu().numpy()
                for i in np.where(valid)[0]:
                    latent_rows.append(
                        {
                            "action": action_name,
                            "regime": label,
                            "horizon": t + 1,
                            "z0": float(z_np[i, 0]),
                            "z1": float(z_np[i, 1]) if z_np.shape[1] > 1 else 0.0,
                        }
                    )
                    for name in outcomes:
                        if name not in pred:
                            continue
                        p = float(pred[name][i].detach().cpu())
                        row = {
                            "action": action_name,
                            "regime": label,
                            "scale": scale,
                            "horizon": t + 1,
                            "event": name,
                            "event_prob": p,
                        }
                        if name == "adl_worsening":
                            row["adl_pred"] = p
                        level_rows.append(row)
                z = feature

    levels = pd.DataFrame(level_rows)
    latents = pd.DataFrame(latent_rows)
    if not levels.empty:
        agg_cols = ["action", "regime", "scale", "horizon", "event"]
        aggregations: dict[str, tuple[str, str]] = {
            "event_prob": ("event_prob", "mean"),
            "n": ("event_prob", "size"),
        }
        if "adl_pred" in levels.columns:
            aggregations["adl_pred"] = ("adl_pred", "mean")
        levels = levels.groupby(agg_cols, as_index=False).agg(**aggregations)
    if not latents.empty:
        latents = latents.groupby(["action", "regime", "horizon"], as_index=False).agg(
            z0=("z0", "mean"),
            z1=("z1", "mean"),
            n=("z0", "size"),
        )
    return levels, latents, action_name


@torch.inference_mode()
def collect_delayed_action_switch(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    action_name: str | None = "cigarettes_per_day",
    switch_horizon: int = 3,
    horizons: int = 5,
    n_persons: int = 512,
    outcome_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Baseline-start cohort; switch action to low/high at ``switch_horizon``.

    Continuous / multi-level categorical: persons whose observed action at t=0
    falls in the **middle tertile**. Binary categorical (e.g. meds): all persons
    with an observed action at t=0.

    Open-loop uses the observed action until horizon ``switch_horizon - 1``, then
    low / high / stay-observed. ``outcome_names`` defaults to ``adl_worsening``.
    """
    world = agent.world
    resolved = _resolve_sim_action(agent.spec, action_name)
    if resolved is None:
        return pd.DataFrame(), None
    action_name, kind, a_idx, cat_low, cat_high = resolved
    _ = preprocessing
    if outcome_names is None:
        outcomes = ["adl_worsening"]
    else:
        outcomes = [n for n in outcome_names if n in agent.spec.reward_binary]
    if not outcomes:
        return pd.DataFrame(), action_name
    switch_horizon = max(int(switch_horizon), 1)
    horizons = max(int(horizons), switch_horizon)

    starts = _gather_start_batches(loader, device, n_persons)
    if not starts:
        return pd.DataFrame(), action_name

    # Observed action at origin among valid, masked persons.
    cig0_parts: list[pd.DataFrame] = []
    for bi, batch in enumerate(starts):
        valid0 = (batch.valid[:, 0] > 0.5).detach().cpu().numpy()
        if kind == "cont":
            mask0 = (batch.action_cont_mask[:, 0, a_idx] > 0.5).detach().cpu().numpy()
            action0 = batch.action_cont[:, 0, a_idx].detach().cpu().numpy()
        else:
            mask0 = (batch.action_cat_mask[:, 0, a_idx] > 0.5).detach().cpu().numpy()
            action0 = batch.action_cat[:, 0, a_idx].detach().cpu().numpy().astype(np.float64)
        keep = valid0 & mask0
        if not keep.any():
            continue
        cig0_parts.append(
            pd.DataFrame(
                {
                    "batch_i": bi,
                    "local_i": np.where(keep)[0],
                    "person_id": np.asarray(batch.person_id, dtype=object)[keep].astype(str),
                    "action0": action0[keep],
                }
            )
        )
    if not cig0_parts:
        return pd.DataFrame(), action_name
    cig0 = pd.concat(cig0_parts, ignore_index=True).drop_duplicates("person_id")
    use_tertile = len(cig0) >= 9 and not (kind == "cat" and cig0["action0"].nunique() <= 2)
    if use_tertile:
        try:
            cig0["tertile"] = pd.qcut(
                cig0["action0"].rank(method="first"),
                q=3,
                labels=["low", "mid", "high"],
            )
            cohort = cig0[cig0["tertile"].astype(str) == "mid"].copy()
        except ValueError:
            cohort = cig0
    else:
        cohort = cig0
    if cohort.empty:
        cohort = cig0
    cohort_keys = set(zip(cohort["batch_i"].tolist(), cohort["local_i"].tolist()))

    regimes = (
        ("baseline", 0.0),
        ("switch_low", -1.0),
        ("switch_high", 1.0),
    )
    level_rows: list[dict[str, Any]] = []
    for regime, post_scale in regimes:
        for bi, batch in enumerate(starts):
            steps = batch.valid.shape[1]
            use_steps = min(steps, horizons)
            z_seq = encode_batch_trajectory(world, batch)
            z = z_seq[:, 0]
            static_embed = encode_batch_static(world, batch, t=0)
            for t in range(use_steps):
                horizon = t + 1
                scale = post_scale if horizon >= switch_horizon else 0.0
                action_cont, action_cat = _apply_sim_action(
                    batch,
                    t,
                    kind=kind,
                    index=a_idx,
                    scale=scale,
                    cat_low_idx=cat_low,
                    cat_high_idx=cat_high,
                )
                feature = world.predict_next(
                    z,
                    action_cont,
                    batch.action_cont_mask[:, t],
                    action_cat,
                    batch.delta_t_norm[:, t],
                    static_embed=static_embed,
                )
                pred = world.clinical_heads.mean_dict(
                    feature,
                    state_cont=batch.state_cont[:, t],
                    state_cont_mask=batch.state_cont_mask[:, t],
                )
                valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
                for i in np.where(valid)[0]:
                    if (bi, int(i)) not in cohort_keys:
                        continue
                    for name in outcomes:
                        if name not in pred:
                            continue
                        p = float(pred[name][i].detach().cpu())
                        row = {
                            "action": action_name,
                            "regime": regime,
                            "switch_horizon": switch_horizon,
                            "scale_after_switch": post_scale,
                            "horizon": horizon,
                            "event": name,
                            "event_prob": p,
                            "person_id": str(batch.person_id[i]),
                        }
                        if name == "adl_worsening":
                            row["adl_pred"] = p
                        level_rows.append(row)
                z = feature

    levels = pd.DataFrame(level_rows)
    if levels.empty:
        return levels, action_name
    group_cols = [
        "action",
        "regime",
        "switch_horizon",
        "scale_after_switch",
        "horizon",
        "event",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "event_prob": ("event_prob", "mean"),
        "n": ("event_prob", "size"),
    }
    if "adl_pred" in levels.columns:
        aggregations["adl_pred"] = ("adl_pred", "mean")
    summary = levels.groupby(group_cols, as_index=False).agg(**aggregations)
    return summary, action_name


def _action_values_and_masks(
    batch: TrajectoryBatch,
    t: int,
    *,
    kind: str,
    a_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "cont":
        vals = batch.action_cont[:, t, a_idx].detach().cpu().numpy().astype(np.float64)
        masks = (batch.action_cont_mask[:, t, a_idx] > 0.5).detach().cpu().numpy()
    else:
        vals = batch.action_cat[:, t, a_idx].detach().cpu().numpy().astype(np.float64)
        masks = (batch.action_cat_mask[:, t, a_idx] > 0.5).detach().cpu().numpy()
    return vals, masks


def _assign_origin_strata(
    action0: pd.Series,
    *,
    kind: str,
    cat_low_idx: int,
    cat_high_idx: int,
) -> pd.Series:
    if action0.empty:
        return pd.Series(dtype=str)
    if kind == "cat" and action0.nunique(dropna=True) <= 2:
        out = pd.Series(index=action0.index, dtype=object)
        out.loc[action0 == float(cat_low_idx)] = "low"
        out.loc[action0 == float(cat_high_idx)] = "high"
        return out.dropna().astype(str)
    if len(action0) < 3:
        return pd.Series("all", index=action0.index, dtype=object)
    try:
        ranks = action0.rank(method="first")
        return pd.qcut(ranks, q=3, labels=["low", "mid", "high"]).astype(str)
    except ValueError:
        return pd.Series("all", index=action0.index, dtype=object)


def _stratum_at_value(
    value: float,
    *,
    kind: str,
    cat_low_idx: int,
    cat_high_idx: int,
    low_upper: float,
    high_lower: float,
) -> str | None:
    if kind == "cat" and cat_low_idx != cat_high_idx:
        iv = int(value)
        if iv == cat_low_idx:
            return "low"
        if iv == cat_high_idx:
            return "high"
        return "mid"
    if value <= low_upper:
        return "low"
    if value >= high_lower:
        return "high"
    return "mid"


def _classify_natural_switch(
    action0: float,
    action_sw: float,
    *,
    kind: str,
    cat_low_idx: int,
    cat_high_idx: int,
    low_upper: float,
    high_lower: float,
) -> str | None:
    s0 = _stratum_at_value(
        action0,
        kind=kind,
        cat_low_idx=cat_low_idx,
        cat_high_idx=cat_high_idx,
        low_upper=low_upper,
        high_lower=high_lower,
    )
    s1 = _stratum_at_value(
        action_sw,
        kind=kind,
        cat_low_idx=cat_low_idx,
        cat_high_idx=cat_high_idx,
        low_upper=low_upper,
        high_lower=high_lower,
    )
    if s0 is None or s1 is None:
        return None
    if kind == "cat" and cat_low_idx != cat_high_idx:
        if int(action0) == cat_high_idx and int(action_sw) == cat_low_idx:
            return "natural_to_low"
        if int(action0) == cat_low_idx and int(action_sw) == cat_high_idx:
            return "natural_to_high"
        if int(action0) == int(action_sw):
            return "natural_stable"
        return None
    if s1 == "low" and s0 in {"mid", "high"}:
        return "natural_to_low"
    if s1 == "high" and s0 in {"low", "mid"}:
        return "natural_to_high"
    if s0 == "mid" and s1 == "mid":
        return "natural_stable"
    return None


def _assign_age_band(age: float) -> str | None:
    if not math.isfinite(age):
        return None
    if age < 60.0:
        return "<60"
    if age < 70.0:
        return "60-69"
    if age < 80.0:
        return "70-79"
    return "80+"


def _origin_age_map(
    loader: DataLoader,
    device: torch.device,
    spec: ModelSpec,
    preprocessing: Mapping[str, Any],
) -> dict[str, float]:
    """Person-level age in years at trajectory origin (inverse-standardized).

    ``age_years`` is a static continuous channel in current ModelSpec.
    """
    static_names = list(spec.static_continuous)
    state_names = list(spec.state_continuous)
    if "age_years" in static_names:
        kind, idx = "static", static_names.index("age_years")
    elif "age_years" in state_names:
        kind, idx = "state", state_names.index("age_years")
    else:
        return {}
    stats = preprocessing.get("continuous", {}).get("age_years", {})
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std", 1.0))
    if not math.isfinite(std) or std < 1e-8:
        std = 1.0
    ages: dict[str, float] = {}
    for raw in loader:
        batch = raw.to(device)
        if batch.valid.shape[1] < 1:
            continue
        pids = np.asarray(batch.person_id, dtype=object).astype(str)
        valid = (batch.valid[:, 0] > 0.5).detach().cpu().numpy()
        if kind == "static":
            mask_t = batch.static_cont_mask[:, 0, idx]
            val_t = batch.static_cont[:, 0, idx]
        else:
            mask_t = batch.state_cont_mask[:, 0, idx]
            val_t = batch.state_cont[:, 0, idx]
        mask = (mask_t > 0.5).detach().cpu().numpy()
        z = val_t.detach().cpu().numpy()
        for i in np.where(valid & mask)[0]:
            ages[pids[i]] = float(z[i]) * std + mean
    return ages


def _standard_age_weights(ages: pd.Series) -> pd.Series:
    bands = ages.map(_assign_age_band)
    counts = bands.value_counts(dropna=True)
    weights = pd.Series(
        {lab: float(counts.get(lab, 0.0)) for lab in AGE_STD_LABELS},
        dtype=float,
    )
    total = float(weights.sum())
    if total <= 0:
        return weights
    return weights / total


def _age_std_mean(
    frame: pd.DataFrame,
    value_col: str,
    weights: pd.Series,
) -> float:
    rates: list[float] = []
    wts: list[float] = []
    for lab in AGE_STD_LABELS:
        sub = frame[frame["age_band"] == lab]
        if sub.empty:
            continue
        rates.append(float(sub[value_col].mean()))
        wts.append(float(weights.get(lab, 0.0)))
    wsum = float(sum(wts))
    if wsum <= 0.0 or not rates:
        return float("nan")
    return float(sum(r * w for r, w in zip(rates, wts)) / wsum)


def _summarize_age_standardized(
    rows: list[dict[str, Any]],
    *,
    value_col: str,
    out_col: str,
    action_name: str,
) -> pd.DataFrame:
    """Direct age-std of person-level rates; empty age bands are renormalized out."""
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "age" not in frame.columns:
        return pd.DataFrame()
    frame = frame[np.isfinite(pd.to_numeric(frame["age"], errors="coerce"))].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["age_band"] = frame["age"].map(_assign_age_band)
    origin = (
        frame.sort_values("horizon", kind="stable")
        .drop_duplicates("person_id")
    )
    weights = _standard_age_weights(origin["age"])
    n_origin = origin.groupby("stratum")["person_id"].nunique()
    group_cols = ["stratum", "horizon"]
    if "event" in frame.columns:
        group_cols.append("event")
    parts: list[dict[str, Any]] = []
    for keys, sub in frame.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: val for col, val in zip(group_cols, keys)}
        rec["action"] = action_name
        rec[out_col] = _age_std_mean(sub, value_col, weights)
        rec["crude_rate"] = float(sub[value_col].mean())
        rec["n_labeled"] = int(len(sub))
        rec["n"] = int(len(sub))
        rec["n_origin"] = int(n_origin.get(rec["stratum"], 0))
        parts.append(rec)
    out = pd.DataFrame(parts)
    sort_cols = [c for c in ("event", "stratum", "horizon") if c in out.columns]
    return out.sort_values(sort_cols, kind="stable")


def _log_h1_cf_obs_stratum_rates(observed: pd.DataFrame, *, action_name: str) -> None:
    """Print h1 obs rates by stay/switch stratum; warn if they collapse to one value."""
    if observed.empty or "horizon" not in observed.columns or "stratum" not in observed.columns:
        return
    h1 = observed[pd.to_numeric(observed["horizon"], errors="coerce") == 1]
    if h1.empty:
        return
    for event, sub in h1.groupby("event", sort=False):
        rates = (
            sub.groupby("stratum", sort=False)["observed_rate"]
            .first()
            .astype(float)
        )
        if rates.empty:
            continue
        parts = [f"{k}={v:.6f}" for k, v in rates.items()]
        print(
            f"Fig 4 h1 obs ({action_name} / {event}): " + ", ".join(parts),
            flush=True,
        )
        if len(rates) >= 2 and rates.nunique(dropna=True) == 1:
            print(
                f"WARNING [{action_name} / {event}]: h1 obs rates are identical "
                f"across strata ({rates.iloc[0]:.6f}) — likely pooled cache or old code.",
                flush=True,
            )


def pool_h1_cf_observed_rate(
    observed: pd.DataFrame,
    *,
    person_rows: Sequence[Mapping[str, Any]] | pd.DataFrame | None = None,
    action_name: str | None = None,
) -> pd.DataFrame:
    """Use the full h1-low cohort risk at horizon 1 for every CF obs stratum.

    Stay/switch labels are defined from h2 onward, so stratum-specific h1 rates
    are selected subsets. Horizon 1 should share the pooled h1-low risk.
    """
    if observed is None or observed.empty or "horizon" not in observed.columns:
        return observed
    if "observed_rate" not in observed.columns:
        return observed
    out = observed.copy()
    h1 = pd.to_numeric(out["horizon"], errors="coerce") == 1
    if not bool(h1.any()):
        return out

    rows_list: list[dict[str, Any]] | None = None
    if person_rows is not None:
        if isinstance(person_rows, pd.DataFrame):
            if not person_rows.empty:
                rows_list = person_rows.to_dict("records")
        else:
            rows_list = [dict(r) for r in person_rows]
    if rows_list:
        h1_rows: list[dict[str, Any]] = []
        for rec in rows_list:
            try:
                if int(rec.get("horizon", 0)) != 1:
                    continue
            except (TypeError, ValueError):
                continue
            item = dict(rec)
            item["stratum"] = "_all"
            h1_rows.append(item)
        if h1_rows:
            name = action_name
            if not name and "action" in out.columns and out["action"].notna().any():
                name = str(out["action"].iloc[0])
            pooled = _summarize_age_standardized(
                h1_rows,
                value_col="observed",
                out_col="observed_rate",
                action_name=name or "",
            )
            if not pooled.empty:
                for _, prow in pooled.iterrows():
                    mask = h1.copy()
                    if "event" in out.columns and "event" in prow.index:
                        mask &= out["event"].astype(str).eq(str(prow["event"]))
                    if "action" in out.columns and "action" in prow.index:
                        mask &= out["action"].astype(str).eq(str(prow["action"]))
                    out.loc[mask, "observed_rate"] = float(prow["observed_rate"])
                    if "crude_rate" in out.columns and "crude_rate" in prow.index:
                        out.loc[mask, "crude_rate"] = float(prow["crude_rate"])
                return out

    keys = [c for c in ("action", "event") if c in out.columns]
    weight_col = next((c for c in ("n_labeled", "n") if c in out.columns), None)

    def _weighted_mean(series: pd.Series, weights: pd.Series | None) -> float:
        vals = pd.to_numeric(series, errors="coerce")
        if weights is None:
            return float(vals.mean())
        w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
        if float(w.sum()) <= 0 or not np.isfinite(vals).any():
            return float(vals.mean())
        keep = np.isfinite(vals.to_numpy()) & (w.to_numpy() >= 0)
        if not keep.any():
            return float(vals.mean())
        return float(np.average(vals.to_numpy()[keep], weights=w.to_numpy()[keep]))

    work = out.loc[h1]
    grouped = work.groupby(keys, sort=False) if keys else [(None, work)]
    for _, sub in grouped:
        mask = h1.copy()
        for col in keys:
            mask &= out[col].eq(sub[col].iloc[0])
        w = sub[weight_col] if weight_col else None
        out.loc[mask, "observed_rate"] = _weighted_mean(sub["observed_rate"], w)
        if "crude_rate" in out.columns:
            out.loc[mask, "crude_rate"] = _weighted_mean(sub["crude_rate"], w)
    return out


def _summarize_observed_outcome_rows(
    rows: list[dict[str, Any]],
    *,
    action_name: str,
    group_col: str,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    origin_keys = [group_col]
    if "event" in frame.columns:
        origin_keys.append("event")
    origin = frame.groupby(origin_keys, as_index=False).agg(
        n_origin=("person_id", "nunique")
    )
    group_cols = ["action", group_col, "horizon"]
    if "event" in frame.columns:
        group_cols.append("event")
    summary = frame.groupby(group_cols, as_index=False).agg(
        observed_rate=("observed", "mean"),
        n_labeled=("observed", "size"),
    )
    summary = summary.merge(origin, on=origin_keys, how="left")
    summary["action"] = action_name
    sort_cols = [c for c in (group_col, "event", "horizon") if c in summary.columns]
    return summary.sort_values(sort_cols, kind="stable")


def _continuous_preprocessing_stats(
    preprocessing: Mapping[str, Any],
    action_name: str,
) -> tuple[float, float]:
    stats = preprocessing.get("continuous", {}).get(action_name, {})
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std", 1.0))
    if not math.isfinite(std) or std < 1e-8:
        std = 1.0
    return mean, std


def _cigarettes_preprocessing_stats(
    preprocessing: Mapping[str, Any],
) -> tuple[float, float, float]:
    """Return ``(mean, std, zero_z)`` for ``cigarettes_per_day``."""
    mean, std = _continuous_preprocessing_stats(preprocessing, "cigarettes_per_day")
    return mean, std, (0.0 - mean) / std


def _cigs_z_to_raw(z: float, mean: float, std: float) -> float:
    return float(z) * std + mean


def _cigs_raw_to_z(raw_cigs: float, mean: float, std: float) -> float:
    return (float(raw_cigs) - mean) / std


def _classify_cigs_raw(raw_cigs: float) -> str:
    """Map raw cigarettes/day to mid / high tier (non-zero smokers only)."""
    if raw_cigs >= PERSISTENT_SMOKING_HIGH_MIN_CIGS:
        return "high"
    if raw_cigs > PERSISTENT_SMOKING_ZERO_EPS:
        return "mid"
    return "zero"


def get_fig4_action_spec(action_name: str) -> Fig4ActionSpec | None:
    return FIG4_ACTION_SPECS.get(action_name)


def _cat_class_values(spec: ModelSpec, action_name: str) -> list[float]:
    for feat in spec.action_categorical:
        if feat.name == action_name:
            return [float(v) for v in feat.values]
    return []


def _embed_to_class(embed: float, cat_values: Sequence[float]) -> float | None:
    idx = int(embed)
    if idx < 1 or idx > len(cat_values):
        return None
    return float(cat_values[idx - 1])


def _class_to_embed(raw: float, cat_values: Sequence[float]) -> int | None:
    target = float(raw)
    for i, val in enumerate(cat_values):
        if abs(float(val) - target) < 1e-8:
            return i + 1
    return None


def _classify_action_raw(raw: float, action_spec: Fig4ActionSpec) -> str:
    if raw <= action_spec.low_max:
        return "low"
    if raw >= action_spec.high_min:
        return "high"
    return "high" if action_spec.binary else "mid"


def _decode_action_raw(
    model_value: float,
    *,
    action_spec: Fig4ActionSpec,
    mean: float,
    std: float,
    cat_values: Sequence[float],
) -> float | None:
    if action_spec.kind == "cont":
        return float(model_value) * std + mean
    return _embed_to_class(model_value, cat_values)


def _collect_person_action_raws(
    loader: DataLoader,
    device: torch.device,
    *,
    kind: str,
    a_idx: int,
    action_spec: Fig4ActionSpec,
    mean: float,
    std: float,
    cat_values: Sequence[float],
    max_horizon: int,
) -> dict[str, dict[int, float]]:
    person_raws: dict[str, dict[int, float]] = {}
    for raw in loader:
        batch = raw.to(device)
        steps = batch.valid.shape[1]
        use_steps = min(steps, max_horizon)
        pids = np.asarray(batch.person_id, dtype=object).astype(str)
        for t in range(use_steps):
            valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
            vals, masks = _action_values_and_masks(batch, t, kind=kind, a_idx=a_idx)
            keep = valid & masks
            for i in np.where(keep)[0]:
                decoded = _decode_action_raw(
                    float(vals[i]),
                    action_spec=action_spec,
                    mean=mean,
                    std=std,
                    cat_values=cat_values,
                )
                if decoded is None:
                    continue
                person_raws.setdefault(str(pids[i]), {})[t] = decoded
    return person_raws


def build_persistent_action_cohort_map(
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    *,
    spec: ModelSpec,
    action_spec: Fig4ActionSpec,
    a_idx: int,
    kind: str,
    max_horizon: int = PERSISTENT_ACTION_MAX_HORIZON,
) -> dict[str, str]:
    """Assign persistent-action cohorts from the first two waves."""
    mean, std = (0.0, 1.0)
    cat_values: list[float] = []
    if kind == "cont":
        mean, std = _continuous_preprocessing_stats(preprocessing, action_spec.name)
    else:
        cat_values = _cat_class_values(spec, action_spec.name)
    low_h = PERSISTENT_ACTION_LOW_HORIZONS
    mid_high_h = PERSISTENT_ACTION_MID_HIGH_HORIZONS
    collect_h = max(max_horizon, low_h, mid_high_h)
    person_raws = _collect_person_action_raws(
        loader,
        device,
        kind=kind,
        a_idx=a_idx,
        action_spec=action_spec,
        mean=mean,
        std=std,
        cat_values=cat_values,
        max_horizon=collect_h,
    )
    cohort: dict[str, str] = {}
    for pid, by_t in person_raws.items():
        if all(t in by_t for t in range(low_h)) and all(
            _classify_action_raw(by_t[t], action_spec) == "low" for t in range(low_h)
        ):
            cohort[pid] = "low"
            continue
        if not all(t in by_t for t in range(mid_high_h)):
            continue
        tiers = [_classify_action_raw(by_t[t], action_spec) for t in range(mid_high_h)]
        if len(set(tiers)) != 1 or tiers[0] == "low":
            continue
        if action_spec.binary and tiers[0] != "high":
            continue
        cohort[pid] = tiers[0]
    return cohort


def build_persistent_smoking_cohort_map(
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    *,
    a_idx: int,
    max_horizon: int = PERSISTENT_SMOKING_MAX_HORIZON,
) -> dict[str, str]:
    """Backward-compatible smoking wrapper around ``build_persistent_action_cohort_map``."""
    action_spec = FIG4_ACTION_SPECS["cigarettes_per_day"]
    dummy_spec = None  # filled by callers that still pass only a_idx
    _ = dummy_spec
    mean, std, _ = _cigarettes_preprocessing_stats(preprocessing)
    low_h = PERSISTENT_SMOKING_LOW_HORIZONS
    mid_high_h = PERSISTENT_SMOKING_MID_HIGH_HORIZONS
    collect_h = max(max_horizon, low_h, mid_high_h)
    person_raws: dict[str, dict[int, float]] = {}
    for raw in loader:
        batch = raw.to(device)
        steps = batch.valid.shape[1]
        use_steps = min(steps, collect_h)
        pids = np.asarray(batch.person_id, dtype=object).astype(str)
        for t in range(use_steps):
            valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
            vals, masks = _action_values_and_masks(batch, t, kind="cont", a_idx=a_idx)
            keep = valid & masks
            for i in np.where(keep)[0]:
                person_raws.setdefault(str(pids[i]), {})[t] = _cigs_z_to_raw(
                    float(vals[i]), mean, std
                )
    cohort: dict[str, str] = {}
    for pid, by_t in person_raws.items():
        if all(t in by_t for t in range(low_h)) and all(
            by_t[t] <= action_spec.low_max for t in range(low_h)
        ):
            cohort[pid] = "low"
            continue
        if not all(t in by_t for t in range(mid_high_h)):
            continue
        tiers = [_classify_action_raw(by_t[t], action_spec) for t in range(mid_high_h)]
        if len(set(tiers)) == 1 and tiers[0] in ("mid", "high"):
            cohort[pid] = tiers[0]
    return cohort


def _collapse_obs_exposed_strata(
    rows: list[dict[str, Any]],
    *,
    binary: bool,
) -> list[dict[str, Any]]:
    """Merge persistent mid/high into one observed stratum (non-binary actions)."""
    if binary:
        return list(rows)
    collapsed: list[dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        if rec.get("stratum") in ("mid", "high"):
            rec["stratum"] = "mid_high"
        collapsed.append(rec)
    return collapsed


def _collapse_obs_smoking_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _collapse_obs_exposed_strata(rows, binary=False)


@torch.inference_mode()
def collect_persistent_action_validation(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    *,
    action_name: str = FIG4_DEFAULT_ACTION,
    max_horizon: int = PERSISTENT_ACTION_MAX_HORIZON,
    outcome_names: Sequence[str] = FIG5_VALIDATION_OUTCOMES,
    return_person_rows: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persistent-action cohorts: age-std observed rates vs open-loop sim."""
    action_spec = get_fig4_action_spec(action_name)
    if action_spec is None:
        return pd.DataFrame(), pd.DataFrame()
    resolved = _resolve_sim_action(agent.spec, action_name)
    if resolved is None:
        return pd.DataFrame(), pd.DataFrame()
    resolved_name, kind, a_idx, _, _ = resolved
    if kind != action_spec.kind:
        return pd.DataFrame(), pd.DataFrame()

    outcomes = [n for n in outcome_names if n in agent.spec.reward_binary]
    if not outcomes:
        return pd.DataFrame(), pd.DataFrame()

    cohort_map = build_persistent_action_cohort_map(
        loader,
        device,
        preprocessing,
        spec=agent.spec,
        action_spec=action_spec,
        a_idx=a_idx,
        kind=kind,
        max_horizon=max_horizon,
    )
    if not cohort_map:
        return pd.DataFrame(), pd.DataFrame()

    age_map = _origin_age_map(loader, device, agent.spec, preprocessing)
    world = agent.world
    sim_labels = ("low", "high") if action_spec.binary else ("low", "mid", "high")

    obs_rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        steps = batch.valid.shape[1]
        use_steps = min(steps, max_horizon)
        pids = np.asarray(batch.person_id, dtype=object).astype(str)
        for outcome_name in outcomes:
            outcome_idx = list(agent.spec.reward_binary).index(outcome_name)
            for t in range(use_steps):
                valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
                rmask = (
                    batch.reward_mask[:, t, outcome_idx].detach().cpu().numpy() > 0.5
                )
                y = batch.reward[:, t, outcome_idx].detach().cpu().numpy()
                for i in np.where(valid & rmask)[0]:
                    pid = pids[i]
                    if pid not in cohort_map or pid not in age_map:
                        continue
                    obs_rows.append(
                        {
                            "person_id": pid,
                            "action": resolved_name,
                            "stratum": cohort_map[pid],
                            "horizon": t + 1,
                            "event": outcome_name,
                            "observed": float(y[i] > 0.5),
                            "age": age_map[pid],
                        }
                    )

    sim_rows: list[dict[str, Any]] = []
    cohort_sets = {
        label: {pid for pid, s in cohort_map.items() if s == label}
        for label in sim_labels
    }
    for stratum, cohort_pids in cohort_sets.items():
        if not cohort_pids:
            continue
        for raw in loader:
            batch = raw.to(device)
            steps = batch.valid.shape[1]
            use_steps = min(steps, max_horizon)
            z_seq = encode_batch_trajectory(world, batch)
            z = z_seq[:, 0]
            static_embed = encode_batch_static(world, batch, t=0)
            pids = np.asarray(batch.person_id, dtype=object).astype(str)
            for t in range(use_steps):
                action_cont = batch.action_cont[:, t]
                action_cat = batch.action_cat[:, t]
                feature = world.predict_next(
                    z,
                    action_cont,
                    batch.action_cont_mask[:, t],
                    action_cat,
                    batch.delta_t_norm[:, t],
                    static_embed=static_embed,
                )
                pred = world.clinical_heads.mean_dict(
                    feature,
                    state_cont=batch.state_cont[:, t],
                    state_cont_mask=batch.state_cont_mask[:, t],
                )
                valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
                for outcome_name in outcomes:
                    if outcome_name not in pred:
                        continue
                    probs = pred[outcome_name].detach().cpu().numpy()
                    for i in np.where(valid)[0]:
                        pid = pids[i]
                        if pid not in cohort_pids or pid not in age_map:
                            continue
                        sim_rows.append(
                            {
                                "person_id": pid,
                                "action": resolved_name,
                                "stratum": stratum,
                                "horizon": t + 1,
                                "event": outcome_name,
                                "event_prob": float(probs[i]),
                                "age": age_map[pid],
                            }
                        )
                z = feature

    if return_person_rows:
        return pd.DataFrame(obs_rows), pd.DataFrame(sim_rows)
    observed = _summarize_age_standardized(
        obs_rows,
        value_col="observed",
        out_col="observed_rate",
        action_name=resolved_name,
    )
    sim = _summarize_age_standardized(
        sim_rows,
        value_col="event_prob",
        out_col="event_prob",
        action_name=resolved_name,
    )
    return observed, sim


def collect_persistent_smoking_validation(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    *,
    max_horizon: int = PERSISTENT_SMOKING_MAX_HORIZON,
    outcome_names: Sequence[str] = FIG5_VALIDATION_OUTCOMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return collect_persistent_action_validation(
        agent,
        loader,
        device,
        preprocessing,
        action_name="cigarettes_per_day",
        max_horizon=max_horizon,
        outcome_names=outcome_names,
    )


def build_h1_low_action_obs_map(
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    *,
    spec: ModelSpec,
    action_spec: Fig4ActionSpec,
    a_idx: int,
    kind: str,
    max_horizon: int = PERSISTENT_ACTION_MAX_HORIZON,
) -> dict[str, str]:
    """Classify h1-low persons as stay_low / switch_mid / switch_high (h2 onward)."""
    mean, std = (0.0, 1.0)
    cat_values: list[float] = []
    if kind == "cont":
        mean, std = _continuous_preprocessing_stats(preprocessing, action_spec.name)
    else:
        cat_values = _cat_class_values(spec, action_spec.name)
    person_raws = _collect_person_action_raws(
        loader,
        device,
        kind=kind,
        a_idx=a_idx,
        action_spec=action_spec,
        mean=mean,
        std=std,
        cat_values=cat_values,
        max_horizon=max_horizon,
    )
    obs_map: dict[str, str] = {}
    for pid, by_t in person_raws.items():
        if 0 not in by_t or _classify_action_raw(by_t[0], action_spec) != "low":
            continue
        dest = None
        for t in range(1, max_horizon):
            if t not in by_t:
                continue
            tier = _classify_action_raw(by_t[t], action_spec)
            if tier == "low":
                continue
            dest = (
                "switch_high"
                if action_spec.binary or tier == "high"
                else "switch_mid"
            )
            break
        obs_map[pid] = dest or "stay_low"
    return obs_map


def build_h1_low_smoking_obs_map(
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    *,
    a_idx: int,
    max_horizon: int = PERSISTENT_SMOKING_MAX_HORIZON,
) -> dict[str, str]:
    """Backward-compatible smoking wrapper."""
    action_spec = FIG4_ACTION_SPECS["cigarettes_per_day"]
    mean, std, _ = _cigarettes_preprocessing_stats(preprocessing)
    person_raws: dict[str, dict[int, float]] = {}
    for raw in loader:
        batch = raw.to(device)
        steps = batch.valid.shape[1]
        use_steps = min(steps, max_horizon)
        pids = np.asarray(batch.person_id, dtype=object).astype(str)
        for t in range(use_steps):
            valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
            vals, masks = _action_values_and_masks(batch, t, kind="cont", a_idx=a_idx)
            keep = valid & masks
            for i in np.where(keep)[0]:
                person_raws.setdefault(str(pids[i]), {})[t] = _cigs_z_to_raw(
                    float(vals[i]), mean, std
                )
    obs_map: dict[str, str] = {}
    for pid, by_t in person_raws.items():
        if _classify_action_raw(by_t.get(0, np.inf), action_spec) != "low":
            continue
        dest = None
        for t in range(1, max_horizon):
            if t not in by_t:
                continue
            tier = _classify_action_raw(by_t[t], action_spec)
            if tier == "low":
                continue
            dest = "switch_high" if tier == "high" else "switch_mid"
            break
        obs_map[pid] = dest or "stay_low"
    return obs_map


def _cf_model_value(
    raw_level: float,
    *,
    action_spec: Fig4ActionSpec,
    mean: float,
    std: float,
    cat_values: Sequence[float],
) -> float | None:
    if action_spec.kind == "cont":
        return _cigs_raw_to_z(raw_level, mean, std)
    embed = _class_to_embed(raw_level, cat_values)
    return None if embed is None else float(embed)


@torch.inference_mode()
def collect_h1_low_counterfactual_validation(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    *,
    action_name: str = FIG4_DEFAULT_ACTION,
    max_horizon: int = PERSISTENT_ACTION_MAX_HORIZON,
    outcome_names: Sequence[str] = FIG5_VALIDATION_OUTCOMES,
    switch_horizon: int = FIG5_H1_CF_SWITCH_HORIZON,
    return_person_rows: bool = False,
    matched_obs_groups: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """h1-low cohort: obs stay vs switch.

    ``matched_obs_groups=True`` (Fig 4 right column): simulate each person with
    observed actions and summarize in the same stay/switch groups as obs.
    ``matched_obs_groups=False`` (Fig 5 KM): counterfactual levels from h2 on
    the full h1-low sample.
    """
    action_spec = get_fig4_action_spec(action_name)
    if action_spec is None:
        return pd.DataFrame(), pd.DataFrame()
    resolved = _resolve_sim_action(agent.spec, action_name)
    if resolved is None:
        return pd.DataFrame(), pd.DataFrame()
    resolved_name, kind, a_idx, _, _ = resolved
    if kind != action_spec.kind:
        return pd.DataFrame(), pd.DataFrame()

    outcomes = [n for n in outcome_names if n in agent.spec.reward_binary]
    if not outcomes:
        return pd.DataFrame(), pd.DataFrame()

    obs_map = build_h1_low_action_obs_map(
        loader,
        device,
        preprocessing,
        spec=agent.spec,
        action_spec=action_spec,
        a_idx=a_idx,
        kind=kind,
        max_horizon=max_horizon,
    )
    if not obs_map:
        return pd.DataFrame(), pd.DataFrame()

    h1_low_pids = set(obs_map.keys())
    age_map = _origin_age_map(loader, device, agent.spec, preprocessing)
    world = agent.world
    mean, std = (0.0, 1.0)
    cat_values: list[float] = []
    if kind == "cont":
        mean, std = _continuous_preprocessing_stats(preprocessing, resolved_name)
    else:
        cat_values = _cat_class_values(agent.spec, resolved_name)
    switch_t = max(int(switch_horizon) - 1, 0)
    cf_regimes: list[tuple[str, float]] = []
    if not matched_obs_groups:
        cf_raw_levels: list[tuple[str, float]] = [("low", float(action_spec.cf_low_raw))]
        if action_spec.cf_mid_raw is not None and not action_spec.binary:
            cf_raw_levels.append(("mid", float(action_spec.cf_mid_raw)))
        cf_raw_levels.append(("high", float(action_spec.cf_high_raw)))
        for label, raw_level in cf_raw_levels:
            model_val = _cf_model_value(
                raw_level,
                action_spec=action_spec,
                mean=mean,
                std=std,
                cat_values=cat_values,
            )
            if model_val is None:
                continue
            cf_regimes.append((label, float(model_val)))
        if not cf_regimes:
            return pd.DataFrame(), pd.DataFrame()

    obs_rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        steps = batch.valid.shape[1]
        use_steps = min(steps, max_horizon)
        pids = np.asarray(batch.person_id, dtype=object).astype(str)
        for outcome_name in outcomes:
            outcome_idx = list(agent.spec.reward_binary).index(outcome_name)
            for t in range(use_steps):
                valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
                rmask = (
                    batch.reward_mask[:, t, outcome_idx].detach().cpu().numpy() > 0.5
                )
                y = batch.reward[:, t, outcome_idx].detach().cpu().numpy()
                for i in np.where(valid & rmask)[0]:
                    pid = pids[i]
                    if pid not in obs_map or pid not in age_map:
                        continue
                    obs_rows.append(
                        {
                            "person_id": pid,
                            "action": resolved_name,
                            "stratum": obs_map[pid],
                            "horizon": t + 1,
                            "event": outcome_name,
                            "observed": float(y[i] > 0.5),
                            "age": age_map[pid],
                        }
                    )

    sim_rows: list[dict[str, Any]] = []
    if matched_obs_groups:
        for raw in loader:
            batch = raw.to(device)
            steps = batch.valid.shape[1]
            use_steps = min(steps, max_horizon)
            z_seq = encode_batch_trajectory(world, batch)
            z = z_seq[:, 0]
            static_embed = encode_batch_static(world, batch, t=0)
            pids = np.asarray(batch.person_id, dtype=object).astype(str)
            for t in range(use_steps):
                feature = world.predict_next(
                    z,
                    batch.action_cont[:, t],
                    batch.action_cont_mask[:, t],
                    batch.action_cat[:, t],
                    batch.delta_t_norm[:, t],
                    static_embed=static_embed,
                )
                pred = world.clinical_heads.mean_dict(
                    feature,
                    state_cont=batch.state_cont[:, t],
                    state_cont_mask=batch.state_cont_mask[:, t],
                )
                valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
                for outcome_name in outcomes:
                    if outcome_name not in pred:
                        continue
                    probs = pred[outcome_name].detach().cpu().numpy()
                    for i in np.where(valid)[0]:
                        pid = pids[i]
                        if pid not in obs_map or pid not in age_map:
                            continue
                        sim_rows.append(
                            {
                                "person_id": pid,
                                "action": resolved_name,
                                "stratum": obs_map[pid],
                                "horizon": t + 1,
                                "event": outcome_name,
                                "event_prob": float(probs[i]),
                                "age": age_map[pid],
                            }
                        )
                z = feature
    else:
        for stratum, target_val in cf_regimes:
            for raw in loader:
                batch = raw.to(device)
                steps = batch.valid.shape[1]
                use_steps = min(steps, max_horizon)
                z_seq = encode_batch_trajectory(world, batch)
                z = z_seq[:, 0]
                static_embed = encode_batch_static(world, batch, t=0)
                pids = np.asarray(batch.person_id, dtype=object).astype(str)
                for t in range(use_steps):
                    action_cont = batch.action_cont[:, t]
                    action_cat = batch.action_cat[:, t]
                    if t >= switch_t:
                        if kind == "cont":
                            action_cont = batch.action_cont[:, t].clone()
                            mask = batch.action_cont_mask[:, t, a_idx] > 0.5
                            action_cont[:, a_idx] = torch.where(
                                mask,
                                torch.full_like(action_cont[:, a_idx], target_val),
                                action_cont[:, a_idx],
                            )
                        else:
                            action_cat = batch.action_cat[:, t].clone()
                            mask = batch.action_cat_mask[:, t, a_idx] > 0.5
                            tgt = torch.full(
                                action_cat[:, a_idx].shape,
                                int(target_val),
                                dtype=action_cat.dtype,
                                device=action_cat.device,
                            )
                            action_cat[:, a_idx] = torch.where(
                                mask, tgt, action_cat[:, a_idx]
                            )
                    feature = world.predict_next(
                        z,
                        action_cont,
                        batch.action_cont_mask[:, t],
                        action_cat,
                        batch.delta_t_norm[:, t],
                        static_embed=static_embed,
                    )
                    pred = world.clinical_heads.mean_dict(
                        feature,
                        state_cont=batch.state_cont[:, t],
                        state_cont_mask=batch.state_cont_mask[:, t],
                    )
                    valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
                    for outcome_name in outcomes:
                        if outcome_name not in pred:
                            continue
                        probs = pred[outcome_name].detach().cpu().numpy()
                        for i in np.where(valid)[0]:
                            pid = pids[i]
                            if pid not in h1_low_pids or pid not in age_map:
                                continue
                            sim_rows.append(
                                {
                                    "person_id": pid,
                                    "action": resolved_name,
                                    "stratum": stratum,
                                    "horizon": t + 1,
                                    "event": outcome_name,
                                    "event_prob": float(probs[i]),
                                    "age": age_map[pid],
                                }
                            )
                    z = feature

    if return_person_rows:
        return pd.DataFrame(obs_rows), pd.DataFrame(sim_rows)
    observed = _summarize_age_standardized(
        obs_rows,
        value_col="observed",
        out_col="observed_rate",
        action_name=resolved_name,
    )
    _log_h1_cf_obs_stratum_rates(observed, action_name=resolved_name)
    sim = _summarize_age_standardized(
        sim_rows,
        value_col="event_prob",
        out_col="event_prob",
        action_name=resolved_name,
    )
    return observed, sim


def _obs_interval_to_person_times(df: pd.DataFrame) -> pd.DataFrame:
    """Convert interval event flags to one (time, event) record per person."""
    if df.empty:
        return pd.DataFrame(columns=["person_id", "stratum", "time", "event"])
    rows: list[dict[str, Any]] = []
    for (pid, stratum), group in df.groupby(["person_id", "stratum"], sort=False):
        ordered = group.sort_values("horizon")
        hits = ordered.loc[ordered["observed"] > 0.5, "horizon"]
        if not hits.empty:
            time = int(hits.min())
            event = 1
        else:
            time = int(ordered["horizon"].max())
            event = 0
        rows.append(
            {
                "person_id": str(pid),
                "stratum": str(stratum),
                "time": time,
                "event": event,
            }
        )
    return pd.DataFrame(rows)


def discrete_kaplan_meier(person_times: pd.DataFrame) -> pd.DataFrame:
    """Discrete-time KM with steps at observed event/censor times."""
    if person_times.empty:
        return pd.DataFrame(
            columns=["stratum", "horizon", "survival", "n_persons", "n_events"]
        )
    parts: list[pd.DataFrame] = []
    for stratum, group in person_times.groupby("stratum", sort=False):
        times = group["time"].to_numpy(dtype=int)
        events = group["event"].to_numpy(dtype=int)
        max_t = int(times.max()) if len(times) else 0
        n0 = int(len(group))
        survival = 1.0
        rows = [
            {
                "stratum": str(stratum),
                "horizon": 0,
                "survival": 1.0,
                "n_persons": n0,
                "n_events": 0,
            }
        ]
        for t in range(1, max_t + 1):
            at_risk = int(np.sum(times >= t))
            n_events = int(np.sum((times == t) & (events == 1)))
            if at_risk > 0 and n_events > 0:
                survival *= 1.0 - (n_events / at_risk)
            rows.append(
                {
                    "stratum": str(stratum),
                    "horizon": t,
                    "survival": float(survival),
                    "n_persons": n0,
                    "n_events": n_events,
                }
            )
        parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def mean_sim_survival(df: pd.DataFrame) -> pd.DataFrame:
    """Mean open-loop survival: S_i(h) = prod_{t<=h} (1 - p_{i,t})."""
    if df.empty:
        return pd.DataFrame(
            columns=["stratum", "horizon", "survival", "n_persons", "n_events"]
        )
    person_rows: list[dict[str, Any]] = []
    for (pid, stratum), group in df.groupby(["person_id", "stratum"], sort=False):
        ordered = group.sort_values("horizon")
        survival = 1.0
        for _, row in ordered.iterrows():
            survival *= 1.0 - float(np.clip(row["event_prob"], 0.0, 1.0))
            person_rows.append(
                {
                    "person_id": str(pid),
                    "stratum": str(stratum),
                    "horizon": int(row["horizon"]),
                    "survival": float(survival),
                }
            )
    if not person_rows:
        return pd.DataFrame(
            columns=["stratum", "horizon", "survival", "n_persons", "n_events"]
        )
    person_df = pd.DataFrame(person_rows)
    out = (
        person_df.groupby(["stratum", "horizon"], as_index=False)
        .agg(survival=("survival", "mean"), n_persons=("person_id", "nunique"))
    )
    out["n_events"] = 0
    zeros = (
        out.groupby("stratum", as_index=False)
        .agg(n_persons=("n_persons", "max"))
        .assign(horizon=0, survival=1.0, n_events=0)
    )
    return pd.concat([zeros, out], ignore_index=True).sort_values(
        ["stratum", "horizon"]
    )


def _km_table(
    df: pd.DataFrame,
    *,
    source: str,
    panel: str,
    kind: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if kind == "obs":
        km = discrete_kaplan_meier(_obs_interval_to_person_times(df))
    else:
        km = mean_sim_survival(df)
    if km.empty:
        return km
    km["source"] = source
    km["panel"] = panel
    km["kind"] = kind
    return km


FIG5_KM_ACTION = "cigarettes_per_day"
FIG5_KM_EVENT = "death_event"
FIG5_KM_CACHE_VERSION = "2"


def _fig5_km_manifest(
    *,
    test_data_path: str | Path,
    seed: int,
    max_horizon: int,
    action: str = FIG5_KM_ACTION,
    preprocessing_path: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "cache_version": FIG5_KM_CACHE_VERSION,
        "test_data": _file_fingerprint(test_data_path),
        "seed": int(seed),
        "action": str(action),
        "event": FIG5_KM_EVENT,
        "max_horizon": int(max_horizon),
        "low_horizons": int(PERSISTENT_ACTION_LOW_HORIZONS),
        "mid_high_horizons": int(PERSISTENT_ACTION_MID_HIGH_HORIZONS),
        "cf_switch_horizon": int(FIG5_H1_CF_SWITCH_HORIZON),
        "preprocessing": (
            None
            if preprocessing_path is None
            else _file_fingerprint(preprocessing_path)
        ),
    }


def _fig5_km_manifest_matches(
    cached: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    keys = [
        "cache_version",
        "test_data",
        "seed",
        "action",
        "event",
        "max_horizon",
        "low_horizons",
        "mid_high_horizons",
        "cf_switch_horizon",
        "preprocessing",
    ]
    return all(cached.get(key) == expected.get(key) for key in keys)


def _try_load_fig5_km_dir(
    cache_dir: Path,
    expected: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    persistent_path = cache_dir / "fig5_km_persistent.parquet"
    cf_path = cache_dir / "fig5_km_counterfactual.parquet"
    meta_path = cache_dir / "manifest.json"
    if not (persistent_path.exists() and cf_path.exists() and meta_path.exists()):
        return None
    try:
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not _fig5_km_manifest_matches(cached, expected):
        return None
    try:
        persistent = pd.read_parquet(persistent_path)
        counterfactual = pd.read_parquet(cf_path)
    except (OSError, ValueError):
        return None
    if persistent.empty and counterfactual.empty:
        return None
    return persistent, counterfactual


def load_fig5_km_cache(
    cache_dir: str | Path,
    *,
    test_data_path: str | Path,
    seed: int,
    max_horizon: int,
    action: str = FIG5_KM_ACTION,
    preprocessing_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    cache_dir = Path(cache_dir)
    expected = _fig5_km_manifest(
        test_data_path=test_data_path,
        seed=seed,
        max_horizon=max_horizon,
        action=action,
        preprocessing_path=preprocessing_path,
    )
    candidates = [cache_dir / action]
    if action == FIG5_KM_ACTION:
        candidates.append(cache_dir)
    for directory in candidates:
        loaded = _try_load_fig5_km_dir(directory, expected)
        if loaded is not None:
            return loaded
    return None


def save_fig5_km_cache(
    cache_dir: str | Path,
    persistent: pd.DataFrame,
    counterfactual: pd.DataFrame,
    *,
    test_data_path: str | Path,
    seed: int,
    max_horizon: int,
    action: str = FIG5_KM_ACTION,
    preprocessing_path: str | Path | None = None,
) -> None:
    cache_dir = Path(cache_dir) / action
    cache_dir.mkdir(parents=True, exist_ok=True)
    persistent.to_parquet(cache_dir / "fig5_km_persistent.parquet", index=False)
    counterfactual.to_parquet(
        cache_dir / "fig5_km_counterfactual.parquet", index=False
    )
    manifest = _fig5_km_manifest(
        test_data_path=test_data_path,
        seed=seed,
        max_horizon=max_horizon,
        action=action,
        preprocessing_path=preprocessing_path,
    )
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


@torch.inference_mode()
def collect_fig5_action_death_km(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    *,
    action_name: str,
    max_horizon: int = PERSISTENT_ACTION_MAX_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Death KM by Fig 4 action groups: persistent (A) and h1-low CF (B)."""
    action_spec = get_fig4_action_spec(action_name)
    if action_spec is None:
        return pd.DataFrame(), pd.DataFrame()
    if FIG5_KM_EVENT not in agent.spec.reward_binary:
        return pd.DataFrame(), pd.DataFrame()

    obs_p, sim_p = collect_persistent_action_validation(
        agent,
        loader,
        device,
        preprocessing,
        action_name=action_name,
        max_horizon=max_horizon,
        outcome_names=[FIG5_KM_EVENT],
        return_person_rows=True,
    )
    obs_c, sim_c = collect_h1_low_counterfactual_validation(
        agent,
        loader,
        device,
        preprocessing,
        action_name=action_name,
        max_horizon=max_horizon,
        outcome_names=[FIG5_KM_EVENT],
        return_person_rows=True,
    )

    if not obs_p.empty:
        obs_p = obs_p.loc[obs_p["event"] == FIG5_KM_EVENT].copy()
    if not sim_p.empty:
        sim_p = sim_p.loc[sim_p["event"] == FIG5_KM_EVENT].copy()
    if not obs_c.empty:
        obs_c = obs_c.loc[obs_c["event"] == FIG5_KM_EVENT].copy()
    if not sim_c.empty:
        sim_c = sim_c.loc[sim_c["event"] == FIG5_KM_EVENT].copy()

    persistent_parts = [
        df
        for df in (
            _km_table(obs_p, source="obs", panel="A", kind="obs"),
            _km_table(sim_p, source="sim", panel="A", kind="sim"),
        )
        if not df.empty
    ]
    counterfactual_parts = [
        df
        for df in (
            _km_table(obs_c, source="obs", panel="B", kind="obs"),
            _km_table(sim_c, source="sim", panel="B", kind="sim"),
        )
        if not df.empty
    ]
    persistent = (
        pd.concat(persistent_parts, ignore_index=True)
        if persistent_parts
        else pd.DataFrame()
    )
    counterfactual = (
        pd.concat(counterfactual_parts, ignore_index=True)
        if counterfactual_parts
        else pd.DataFrame()
    )
    if not persistent.empty:
        persistent["action"] = action_name
        persistent["event"] = FIG5_KM_EVENT
    if not counterfactual.empty:
        counterfactual["action"] = action_name
        counterfactual["event"] = FIG5_KM_EVENT
    return persistent, counterfactual


@torch.inference_mode()
def collect_fig5_smoking_death_km(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    *,
    max_horizon: int = PERSISTENT_ACTION_MAX_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible smoking wrapper."""
    return collect_fig5_action_death_km(
        agent,
        loader,
        device,
        preprocessing,
        action_name=FIG5_KM_ACTION,
        max_horizon=max_horizon,
    )


@torch.inference_mode()
def collect_observed_action_strata_outcomes(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    action_name: str,
    *,
    max_horizon: int = 5,
    outcome_name: str = "adl_worsening",
) -> pd.DataFrame:
    """Plan B: observed ADL rate by origin action stratum vs horizon.

    Persons are assigned low / mid / high from **observed** action at t=0
    (tertile for continuous / multi-level categorical; low/high only for binary).
    """
    resolved = _resolve_sim_action(agent.spec, action_name)
    if resolved is None:
        return pd.DataFrame()
    action_name, kind, a_idx, cat_low, cat_high = resolved
    if outcome_name not in agent.spec.reward_binary:
        return pd.DataFrame()
    outcome_idx = list(agent.spec.reward_binary).index(outcome_name)

    origin_parts: list[pd.DataFrame] = []
    for raw in loader:
        batch = raw.to(device)
        valid0 = (batch.valid[:, 0] > 0.5).detach().cpu().numpy()
        vals0, mask0 = _action_values_and_masks(batch, 0, kind=kind, a_idx=a_idx)
        keep = valid0 & mask0
        if not keep.any():
            continue
        pids = np.asarray(batch.person_id, dtype=object)[keep].astype(str)
        origin_parts.append(
            pd.DataFrame({"person_id": pids, "action0": vals0[keep]})
        )
    if not origin_parts:
        return pd.DataFrame()
    origin = pd.concat(origin_parts, ignore_index=True).drop_duplicates("person_id")
    origin["stratum"] = _assign_origin_strata(
        origin["action0"],
        kind=kind,
        cat_low_idx=cat_low,
        cat_high_idx=cat_high,
    )
    origin = origin.dropna(subset=["stratum"])
    if origin.empty:
        return pd.DataFrame()
    stratum_map = origin.set_index("person_id")["stratum"].to_dict()
    low_upper, high_lower = (
        float("-inf"),
        float("inf"),
    )
    if kind == "cont" or (kind == "cat" and origin["action0"].nunique() > 2):
        vals = origin["action0"].to_numpy(dtype=np.float64)
        if len(vals) >= 3:
            low_upper, high_lower = np.quantile(vals, [1.0 / 3.0, 2.0 / 3.0])
            low_upper = float(low_upper)
            high_lower = float(high_lower)

    rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        steps = batch.valid.shape[1]
        use_steps = min(steps, max_horizon)
        pids = np.asarray(batch.person_id, dtype=object).astype(str)
        for t in range(use_steps):
            valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
            rmask = (
                batch.reward_mask[:, t, outcome_idx].detach().cpu().numpy() > 0.5
            )
            y = batch.reward[:, t, outcome_idx].detach().cpu().numpy()
            for i in np.where(valid & rmask)[0]:
                pid = pids[i]
                if pid not in stratum_map:
                    continue
                rows.append(
                    {
                        "person_id": pid,
                        "action": action_name,
                        "stratum": stratum_map[pid],
                        "horizon": t + 1,
                        "observed": float(y[i] > 0.5),
                    }
                )
    _ = (low_upper, high_lower)
    return _summarize_observed_outcome_rows(
        rows, action_name=action_name, group_col="stratum"
    )


@torch.inference_mode()
def collect_natural_action_switch_outcomes(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    action_name: str,
    *,
    switch_horizon: int = FIG5_SWITCH_HORIZON,
    max_horizon: int = 5,
    outcome_name: str = "adl_worsening",
) -> pd.DataFrame:
    """Plan C: observed ADL rate for persons with natural action switches.

    ``natural_to_low`` / ``natural_to_high`` / ``natural_stable`` are defined
    from observed action at t=0 vs t=switch_horizon-1 (wave-level switch at h).
    """
    resolved = _resolve_sim_action(agent.spec, action_name)
    if resolved is None:
        return pd.DataFrame()
    action_name, kind, a_idx, cat_low, cat_high = resolved
    if outcome_name not in agent.spec.reward_binary:
        return pd.DataFrame()
    outcome_idx = list(agent.spec.reward_binary).index(outcome_name)
    switch_t = max(int(switch_horizon) - 1, 0)

    origin_parts: list[pd.DataFrame] = []
    for raw in loader:
        batch = raw.to(device)
        steps = batch.valid.shape[1]
        if steps <= switch_t:
            continue
        valid0 = (batch.valid[:, 0] > 0.5).detach().cpu().numpy()
        valid_sw = (batch.valid[:, switch_t] > 0.5).detach().cpu().numpy()
        vals0, mask0 = _action_values_and_masks(batch, 0, kind=kind, a_idx=a_idx)
        vals_sw, mask_sw = _action_values_and_masks(
            batch, switch_t, kind=kind, a_idx=a_idx
        )
        n_steps = (
            (batch.valid > 0.5).sum(dim=1).detach().cpu().numpy().astype(np.int64)
        )
        keep = valid0 & mask0 & valid_sw & mask_sw & (n_steps >= switch_horizon)
        if not keep.any():
            continue
        pids = np.asarray(batch.person_id, dtype=object)[keep].astype(str)
        origin_parts.append(
            pd.DataFrame(
                {
                    "person_id": pids,
                    "action0": vals0[keep],
                    "action_sw": vals_sw[keep],
                }
            )
        )
    if not origin_parts:
        return pd.DataFrame()
    origin = pd.concat(origin_parts, ignore_index=True).drop_duplicates("person_id")
    low_upper, high_lower = float("-inf"), float("inf")
    if kind == "cont" or (kind == "cat" and origin["action0"].nunique() > 2):
        vals = origin["action0"].to_numpy(dtype=np.float64)
        if len(vals) >= 3:
            low_upper, high_lower = np.quantile(vals, [1.0 / 3.0, 2.0 / 3.0])
            low_upper = float(low_upper)
            high_lower = float(high_lower)
    regimes: list[str | None] = []
    for _, row in origin.iterrows():
        regimes.append(
            _classify_natural_switch(
                float(row["action0"]),
                float(row["action_sw"]),
                kind=kind,
                cat_low_idx=cat_low,
                cat_high_idx=cat_high,
                low_upper=low_upper,
                high_lower=high_lower,
            )
        )
    origin["regime"] = regimes
    origin = origin.dropna(subset=["regime"])
    if origin.empty:
        return pd.DataFrame()
    regime_map = origin.set_index("person_id")["regime"].to_dict()

    rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        steps = batch.valid.shape[1]
        use_steps = min(steps, max_horizon)
        pids = np.asarray(batch.person_id, dtype=object).astype(str)
        for t in range(use_steps):
            valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
            rmask = (
                batch.reward_mask[:, t, outcome_idx].detach().cpu().numpy() > 0.5
            )
            y = batch.reward[:, t, outcome_idx].detach().cpu().numpy()
            for i in np.where(valid & rmask)[0]:
                pid = pids[i]
                if pid not in regime_map:
                    continue
                rows.append(
                    {
                        "person_id": pid,
                        "action": action_name,
                        "regime": regime_map[pid],
                        "horizon": t + 1,
                        "observed": float(y[i] > 0.5),
                    }
                )
    summary = _summarize_observed_outcome_rows(
        rows, action_name=action_name, group_col="regime"
    )
    return summary


def summarize_openloop_relative_persistence(rollout: pd.DataFrame) -> pd.DataFrame:
    """Per-outcome open-loop MAE for JEPA / Persistence and JEPA÷Persistence."""
    if rollout.empty:
        return pd.DataFrame()
    use = rollout.copy()
    use["jepa_err"] = (use["jepa_raw"] - use["observed_raw"]).abs()
    use["pers_err"] = (use["persistence_raw"] - use["observed_raw"]).abs()
    agg = (
        use.groupby(["outcome", "horizon"], as_index=False)
        .agg(
            jepa_mae=("jepa_err", "mean"),
            persistence_mae=("pers_err", "mean"),
            n=("jepa_err", "size"),
        )
        .sort_values(["outcome", "horizon"], kind="stable")
    )
    agg["label"] = agg["outcome"].map(lambda o: LEVEL_OUTCOMES.get(str(o), str(o)))
    agg["relative_mae"] = agg["jepa_mae"] / agg["persistence_mae"].clip(lower=1e-8)
    agg["mae_gap"] = agg["jepa_mae"] - agg["persistence_mae"]
    return agg


def _frozen_prob_maps(
    table: pd.DataFrame | None,
    names: Sequence[str],
    *,
    column_for: Callable[[str], str],
) -> dict[str, dict[tuple[str, int, int], float]]:
    """(person_id, wave, next_wave) -> frozen h=1 probability per event."""
    maps: dict[str, dict[tuple[str, int, int], float]] = {n: {} for n in names}
    if table is None or table.empty:
        return maps
    b = table.copy()
    b["person_id"] = b["person_id"].astype(str)
    if "wave" not in b.columns or "next_wave" not in b.columns:
        return maps
    for name in names:
        col = column_for(name)
        if col not in b.columns:
            continue
        use = b.dropna(subset=[col, "wave", "next_wave"])
        for _, row in use[["person_id", "wave", "next_wave", col]].iterrows():
            key = (str(row["person_id"]), int(row["wave"]), int(row["next_wave"]))
            maps[name][key] = float(row[col])
    return maps


def enrich_event_rollout_with_mlp(
    rollout: pd.DataFrame,
    mlp_tabular: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach frozen MLP h=1 probabilities to cached open-loop rollout rows."""
    if rollout.empty or mlp_tabular is None or mlp_tabular.empty:
        return rollout
    if "mlp_prob" in rollout.columns:
        return rollout
    events = [str(e) for e in rollout["event"].unique()]
    maps = _frozen_prob_maps(
        mlp_tabular,
        events,
        column_for=lambda name: f"mlp_prob__{name}",
    )
    out = rollout.copy()
    probs: list[float] = []
    for _, row in out.iterrows():
        key = (str(row["person_id"]), int(row["start_wave"]), int(row["start_next_wave"]))
        ev = str(row["event"])
        probs.append(float(maps.get(ev, {}).get(key, np.nan)))
    out["mlp_prob"] = probs
    return out


@torch.inference_mode()
def collect_jepa_event_rollouts(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    max_horizon: int = 5,
    event_names: Sequence[str] | None = None,
    baseline: pd.DataFrame | None = None,
    mlp_tabular: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Open-loop binary-event probabilities vs horizon.

    Persistence analog = freeze each person's **Logistic** one-step probability
    at the trajectory's first transition (horizon=1), then score later-wave
    labels with that fixed risk. Columns::

        jepa_prob           open-loop JEPA ClinicalHeads at horizon h
        persistence_prob    Logistic ``baseline_prob__*`` from (person, wave0, next_wave0)
        mlp_prob            Tabular MLP ``mlp_prob__*`` from the same h=1 transition
    """
    world = agent.world
    names = list(event_names) if event_names is not None else list(PROBE_BINARY_REWARDS)
    names = [n for n in names if n in agent.spec.reward_binary]
    if not names:
        return pd.DataFrame()
    lookup = {n: i for i, n in enumerate(agent.spec.reward_binary)}

    logistic_maps = _frozen_prob_maps(
        baseline,
        names,
        column_for=lambda name: f"baseline_prob__{name}",
    )
    mlp_maps = _frozen_prob_maps(
        mlp_tabular,
        names,
        column_for=lambda name: f"mlp_prob__{name}",
    )

    rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        batch_size, steps = batch.valid.shape
        use_steps = min(steps, max_horizon)
        z_seq = encode_batch_trajectory(world, batch)
        z = z_seq[:, 0]
        static_embed = encode_batch_static(world, batch, t=0)
        pids = np.asarray(batch.person_id, dtype=object)
        wave0 = batch.wave[:, 0].detach().cpu().numpy().astype(int)
        next0 = batch.next_wave[:, 0].detach().cpu().numpy().astype(int)
        # Freeze Logistic / MLP scores from the h=1 transition for each person/event.
        p_h1 = {n: np.full(batch_size, np.nan, dtype=np.float64) for n in names}
        p_mlp_h1 = {n: np.full(batch_size, np.nan, dtype=np.float64) for n in names}
        for name in names:
            for i in range(batch_size):
                key = (str(pids[i]), int(wave0[i]), int(next0[i]))
                if key in logistic_maps.get(name, {}):
                    p_h1[name][i] = logistic_maps[name][key]
                if key in mlp_maps.get(name, {}):
                    p_mlp_h1[name][i] = mlp_maps[name][key]

        for t in range(use_steps):
            valid = batch.valid[:, t] > 0.5
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
            active = valid.detach().cpu().numpy()
            for name in names:
                if name not in reward_pred:
                    continue
                idx = lookup[name]
                m = active & (batch.reward_mask[:, t, idx].detach().cpu().numpy() > 0.5)
                if not m.any():
                    continue
                p = reward_pred[name].detach().cpu().numpy()
                y = batch.reward[:, t, idx].detach().cpu().numpy()
                for i in np.where(m)[0]:
                    rows.append(
                        {
                            "person_id": str(pids[i]),
                            "horizon": t + 1,
                            "event": name,
                            "label": EVENT_LABELS.get(name, name),
                            "observed": float(y[i]),
                            "jepa_prob": float(p[i]),
                            "persistence_prob": float(p_h1[name][i]),
                            "mlp_prob": float(p_mlp_h1[name][i]),
                            "start_wave": int(wave0[i]),
                            "start_next_wave": int(next0[i]),
                        }
                    )
            z = feature
    return pd.DataFrame(rows)


def summarize_event_openloop_dynamics(rollout: pd.DataFrame) -> pd.DataFrame:
    """Per-event open-loop metrics for JEPA vs frozen Logistic/MLP h=1.

    Absolute panels use Logistic (``persistence_*``). Relative panels use
    JEPA / frozen MLP h=1 (``relative_*`` vs ``mlp_h1_*``).
    """
    if rollout.empty:
        return pd.DataFrame()
    has_mlp = "mlp_prob" in rollout.columns
    rows: list[dict[str, Any]] = []
    for (event, horizon), group in rollout.groupby(["event", "horizon"], sort=False):
        y = (pd.to_numeric(group["observed"], errors="coerce").to_numpy(float) > 0.5).astype(int)
        p_j = pd.to_numeric(group["jepa_prob"], errors="coerce").to_numpy(float)
        p_p = pd.to_numeric(group["persistence_prob"], errors="coerce").to_numpy(float)
        p_m = (
            pd.to_numeric(group["mlp_prob"], errors="coerce").to_numpy(float)
            if has_mlp
            else np.full(len(group), np.nan, dtype=np.float64)
        )
        m_log = np.isfinite(p_j) & np.isfinite(p_p)
        m_mlp = np.isfinite(p_j) & np.isfinite(p_m) if has_mlp else np.zeros(len(y), dtype=bool)
        if m_log.sum() < 20 or len(np.unique(y[m_log])) < 2:
            continue
        auroc_j = safe_auroc(p_j[m_log], y[m_log])
        auroc_p = safe_auroc(p_p[m_log], y[m_log])
        auprc_j = safe_auprc(y[m_log], p_j[m_log])
        auprc_p = safe_auprc(y[m_log], p_p[m_log])
        rates_j = binary_rates_at_threshold(y[m_log], p_j[m_log])
        rates_p = binary_rates_at_threshold(y[m_log], p_p[m_log])
        sen_at_spec_j = sensitivity_at_min_specificity(
            y[m_log], p_j[m_log], min_specificity=0.8
        )
        sen_at_spec_p = sensitivity_at_min_specificity(
            y[m_log], p_p[m_log], min_specificity=0.8
        )
        if m_mlp.sum() >= 20 and len(np.unique(y[m_mlp])) >= 2:
            auroc_j_mlp = safe_auroc(p_j[m_mlp], y[m_mlp])
            auroc_m = safe_auroc(p_m[m_mlp], y[m_mlp])
            auprc_j_mlp = safe_auprc(y[m_mlp], p_j[m_mlp])
            auprc_m = safe_auprc(y[m_mlp], p_m[m_mlp])
            rates_j_mlp = binary_rates_at_threshold(y[m_mlp], p_j[m_mlp])
            rates_m = binary_rates_at_threshold(y[m_mlp], p_m[m_mlp])
            sen_at_spec_j_mlp = sensitivity_at_min_specificity(
                y[m_mlp], p_j[m_mlp], min_specificity=0.8
            )
            sen_at_spec_m = sensitivity_at_min_specificity(
                y[m_mlp], p_m[m_mlp], min_specificity=0.8
            )
        else:
            auroc_j_mlp = float("nan")
            auroc_m = float("nan")
            auprc_j_mlp = float("nan")
            auprc_m = float("nan")
            rates_j_mlp = {k: float("nan") for k in rates_j}
            rates_m = {k: float("nan") for k in rates_j}
            sen_at_spec_j_mlp = {"sensitivity": float("nan"), "threshold": float("nan"), "specificity": float("nan")}
            sen_at_spec_m = {"sensitivity": float("nan"), "threshold": float("nan"), "specificity": float("nan")}
        prev = float(y[m_log].mean())
        row: dict[str, Any] = {
            "event": event,
            "label": EVENT_LABELS.get(str(event), str(event)),
            "horizon": int(horizon),
            "jepa_auroc": auroc_j,
            "persistence_auroc": auroc_p,
            "logistic_h1_auroc": auroc_p,
            "mlp_h1_auroc": auroc_m,
            "jepa_auprc": auprc_j,
            "persistence_auprc": auprc_p,
            "logistic_h1_auprc": auprc_p,
            "mlp_h1_auprc": auprc_m,
            "relative_auroc": (
                float(auroc_j_mlp / auroc_m)
                if np.isfinite(auroc_j_mlp) and np.isfinite(auroc_m) and auroc_m > 1e-8
                else float("nan")
            ),
            "relative_auprc": (
                float(auprc_j_mlp / auprc_m)
                if np.isfinite(auprc_j_mlp) and np.isfinite(auprc_m) and auprc_m > 1e-8
                else float("nan")
            ),
            "n": int(m_log.sum()),
            "events": int(y[m_log].sum()),
            "prevalence": prev,
        }
        for metric in ("accuracy", "balanced_accuracy", "sensitivity", "specificity"):
            j_val = float(rates_j[metric])
            p_val = float(rates_p[metric])
            j_rel = float(rates_j_mlp[metric])
            m_val = float(rates_m[metric])
            row[f"jepa_{metric}"] = j_val
            row[f"persistence_{metric}"] = p_val
            row[f"logistic_h1_{metric}"] = p_val
            row[f"mlp_h1_{metric}"] = m_val
            row[f"relative_{metric}"] = (
                float(j_rel / m_val)
                if np.isfinite(j_rel) and np.isfinite(m_val) and m_val > 1e-8
                else float("nan")
            )
        metric_np = "sensitivity_at_spec80"
        j_np = float(sen_at_spec_j["sensitivity"])
        p_np = float(sen_at_spec_p["sensitivity"])
        j_np_rel = float(sen_at_spec_j_mlp["sensitivity"])
        m_np = float(sen_at_spec_m["sensitivity"])
        row[f"jepa_{metric_np}"] = j_np
        row[f"persistence_{metric_np}"] = p_np
        row[f"logistic_h1_{metric_np}"] = p_np
        row[f"mlp_h1_{metric_np}"] = m_np
        row[f"relative_{metric_np}"] = (
            float(j_np_rel / m_np)
            if np.isfinite(j_np_rel) and np.isfinite(m_np) and m_np > 1e-8
            else float("nan")
        )
        row["jepa_threshold_at_spec80"] = float(sen_at_spec_j["threshold"])
        row["persistence_threshold_at_spec80"] = float(sen_at_spec_p["threshold"])
        row["jepa_specificity_at_spec80"] = float(sen_at_spec_j["specificity"])
        row["persistence_specificity_at_spec80"] = float(
            sen_at_spec_p["specificity"]
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["event", "horizon"], kind="stable")


def summarize_event_openloop_dynamics_hstar(rollout: pd.DataFrame) -> pd.DataFrame:
    """Open-loop metrics vs horizon-specific tabular MLP-h* baseline."""
    if rollout.empty or "mlp_hstar_prob" not in rollout.columns:
        return pd.DataFrame()
    tmp = rollout.copy()
    tmp["mlp_prob"] = tmp["mlp_hstar_prob"]
    base = summarize_event_openloop_dynamics(tmp)
    if base.empty:
        return base
    rename: dict[str, str] = {}
    for col in base.columns:
        if col.startswith("mlp_h1_"):
            rename[col] = col.replace("mlp_h1_", "mlp_hstar_", 1)
        elif col.startswith("relative_"):
            rename[col] = col.replace("relative_", "relative_hstar_", 1)
    return base.rename(columns=rename)


@torch.inference_mode()
def collect_intervention_effects(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    preprocessing: Mapping[str, Any],
    action_name: str | None = None,
    horizons: int = 5,
    n_persons: int = 256,
) -> tuple[pd.DataFrame, str | None]:
    """Open-loop high−low action contrast vs horizon (ADL level + death prob).

    Effect at horizon h::

        ΔY(h) = mean(Y | a←high) − mean(Y | a←low)

    computed on the same starting states (paired by person index in batch).
    """
    world = agent.world
    cont_actions = [a.name for a in agent.spec.action_continuous]
    if not cont_actions:
        return pd.DataFrame(), None
    if action_name is None or action_name not in cont_actions:
        preferred = [
            "vigorous_activity_frequency",
            "moderate_activity_frequency",
            "cigarettes_per_day",
            "drinks_per_drinking_day",
        ]
        action_name = next((a for a in preferred if a in cont_actions), cont_actions[0])
    a_idx = cont_actions.index(action_name)
    _ = preprocessing
    adl_name = "adl_worsening"
    has_adl = adl_name in agent.spec.reward_binary
    has_death = "death_event" in agent.spec.reward_binary
    if not has_adl and not has_death:
        return pd.DataFrame(), action_name

    starts: list[TrajectoryBatch] = []
    seen = 0
    for raw in loader:
        starts.append(raw.to(device))
        seen += int(raw.valid[:, 0].sum().item())
        if seen >= n_persons:
            break
    if not starts:
        return pd.DataFrame(), action_name

    # person_id, horizon -> {low/high adl, death}
    rows: list[dict[str, Any]] = []
    for scale, regime in [(-1.0, "low"), (1.0, "high")]:
        for batch in starts:
            batch_size, steps = batch.valid.shape
            use_steps = min(steps, horizons)
            z_seq = encode_batch_trajectory(world, batch)
            z = z_seq[:, 0]
            static_embed = encode_batch_static(world, batch, t=0)
            pids = np.asarray(batch.person_id, dtype=object)
            for t in range(use_steps):
                action_cont = batch.action_cont[:, t].clone()
                mask = batch.action_cont_mask[:, t, a_idx]
                action_cont[:, a_idx] = action_cont[:, a_idx] + scale * mask
                feature = world.predict_next(
                    z,
                    action_cont,
                    batch.action_cont_mask[:, t],
                    batch.action_cat[:, t],
                    batch.delta_t_norm[:, t],
                    static_embed=static_embed,
                )
                pred = world.clinical_heads.mean_dict(
                    feature,
                    state_cont=batch.state_cont[:, t],
                    state_cont_mask=batch.state_cont_mask[:, t],
                )
                valid = (batch.valid[:, t] > 0.5).detach().cpu().numpy()
                adl_raw = None
                if has_adl and adl_name in pred:
                    adl_raw = pred[adl_name].detach().cpu().numpy()
                death_p = (
                    pred["death_event"].detach().cpu().numpy() if has_death else None
                )
                for i in np.where(valid)[0]:
                    rows.append(
                        {
                            "person_id": str(pids[i]),
                            "action": action_name,
                            "regime": regime,
                            "horizon": t + 1,
                            "adl_pred": float(adl_raw[i]) if adl_raw is not None else float("nan"),
                            "death_prob": float(death_p[i]) if death_p is not None else float("nan"),
                        }
                    )
                z = feature

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, action_name
    low = frame[frame["regime"] == "low"].rename(
        columns={"adl_pred": "adl_low", "death_prob": "death_low"}
    )
    high = frame[frame["regime"] == "high"].rename(
        columns={"adl_pred": "adl_high", "death_prob": "death_high"}
    )
    keys = ["person_id", "action", "horizon"]
    merged = low[keys + ["adl_low", "death_low"]].merge(
        high[keys + ["adl_high", "death_high"]], on=keys, how="inner"
    )
    if merged.empty:
        return pd.DataFrame(), action_name
    merged["adl_effect"] = merged["adl_high"] - merged["adl_low"]
    merged["death_effect"] = merged["death_high"] - merged["death_low"]
    out = (
        merged.groupby(["action", "horizon"], as_index=False)
        .agg(
            adl_effect=("adl_effect", "mean"),
            death_effect=("death_effect", "mean"),
            adl_effect_std=("adl_effect", "std"),
            death_effect_std=("death_effect", "std"),
            n=("adl_effect", "size"),
        )
        .sort_values("horizon", kind="stable")
    )
    return out, action_name


def _carry_last_valid(step_valid: torch.Tensor, current: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    """Keep ``last`` where the step is padding; otherwise take ``current``."""
    view = step_valid.reshape((step_valid.shape[0],) + (1,) * (current.ndim - 1))
    return torch.where(view, current, last)


@torch.inference_mode()
def collect_openloop_event_risk(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    event_names: Sequence[str],
    max_horizon: int = 5,
) -> pd.DataFrame:
    """Open-loop event probability vs horizon for one or more binary rewards.

    Every person with a valid origin is unrolled up to
    ``min(batch_steps, max_horizon)`` — the same cap as
    ``collect_jepa_event_rollouts`` (Fig 3A–C). No carry-forward beyond the
    padded trajectory width. Observed events and ``followup_wave`` use the full
    *observed* trajectory.
    """
    names = [n for n in event_names if n in agent.spec.reward_binary]
    if not names:
        return pd.DataFrame()
    world = agent.world
    lookup = {n: list(agent.spec.reward_binary).index(n) for n in names}
    rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = raw.to(device)
        batch_size, steps = batch.valid.shape
        z_seq = encode_batch_trajectory(world, batch)
        z = z_seq[:, 0]
        static_embed = encode_batch_static(world, batch, t=0)
        pids = np.asarray(batch.person_id, dtype=object)
        origin = (batch.valid[:, 0] > 0.5).detach().cpu().numpy()
        n_valid_steps = (
            (batch.valid > 0.5).sum(dim=1).detach().cpu().numpy().astype(np.int64)
        )
        obs_event = {n: np.zeros(batch_size, dtype=bool) for n in names}
        followup_wave = {n: np.full(batch_size, -1, dtype=np.int64) for n in names}
        n_event_steps = {n: np.zeros(batch_size, dtype=np.int64) for n in names}
        for t in range(steps):
            valid_t = batch.valid[:, t] > 0.5
            nw = batch.next_wave[:, t].detach().cpu().numpy().astype(np.int64)
            for name, idx in lookup.items():
                m = valid_t & (batch.reward_mask[:, t, idx] > 0.5)
                y = batch.reward[:, t, idx] > 0.5
                m_np = m.detach().cpu().numpy()
                obs_event[name] |= (m & y).detach().cpu().numpy()
                n_event_steps[name] += m_np.astype(np.int64)
                followup_wave[name] = np.where(m_np, nw, followup_wave[name])
        action_cont = batch.action_cont[:, 0]
        action_cont_mask = batch.action_cont_mask[:, 0]
        action_cat = batch.action_cat[:, 0]
        delta_t_norm = batch.delta_t_norm[:, 0]
        state_cont = batch.state_cont[:, 0]
        state_cont_mask = batch.state_cont_mask[:, 0]
        p_h1 = {n: np.full(batch_size, np.nan, dtype=np.float64) for n in names}
        use_steps = min(steps, max_horizon)
        for t in range(use_steps):
            step_valid = batch.valid[:, t] > 0.5
            action_cont = _carry_last_valid(step_valid, batch.action_cont[:, t], action_cont)
            action_cont_mask = _carry_last_valid(
                step_valid, batch.action_cont_mask[:, t], action_cont_mask
            )
            action_cat = _carry_last_valid(step_valid, batch.action_cat[:, t], action_cat)
            delta_t_norm = _carry_last_valid(
                step_valid, batch.delta_t_norm[:, t], delta_t_norm
            )
            state_cont = _carry_last_valid(step_valid, batch.state_cont[:, t], state_cont)
            state_cont_mask = _carry_last_valid(
                step_valid, batch.state_cont_mask[:, t], state_cont_mask
            )
            feature = world.predict_next(
                z,
                action_cont,
                action_cont_mask,
                action_cat,
                delta_t_norm,
                static_embed=static_embed,
            )
            pred = world.clinical_heads.mean_dict(
                feature,
                state_cont=state_cont,
                state_cont_mask=state_cont_mask,
            )
            if t == 0:
                for name in names:
                    if name in pred:
                        p_h1[name][origin] = pred[name].detach().cpu().numpy()[origin]
            for i in np.where(origin)[0]:
                for name in names:
                    if name not in pred:
                        continue
                    rows.append(
                        {
                            "event": name,
                            "person_id": str(pids[i]),
                            "horizon": t + 1,
                            "event_prob": float(pred[name][i].detach().cpu()),
                            "event_prob_h1": float(p_h1[name][i]),
                            "observed_event": bool(obs_event[name][i]),
                            "followup_wave": int(followup_wave[name][i]),
                            "n_valid_steps": int(n_valid_steps[i]),
                            "n_event_steps": int(n_event_steps[name][i]),
                        }
                    )
            z = feature
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    tertile_parts = []
    for name, sub in frame[frame["horizon"] == 1].groupby("event", sort=False):
        h1 = (
            sub[["person_id", "event_prob_h1"]]
            .drop_duplicates("person_id")
            .dropna(subset=["event_prob_h1"])
        )
        if len(h1) >= 9:
            try:
                h1["risk_tertile"] = pd.qcut(
                    h1["event_prob_h1"].rank(method="first"),
                    q=3,
                    labels=["low", "mid", "high"],
                )
            except ValueError:
                h1["risk_tertile"] = "all"
        else:
            h1["risk_tertile"] = "all"
        h1 = h1[["person_id", "risk_tertile"]].copy()
        h1["event"] = name
        tertile_parts.append(h1)
    if tertile_parts:
        tertiles = pd.concat(tertile_parts, ignore_index=True)
        frame = frame.merge(tertiles, on=["event", "person_id"], how="left")
    else:
        frame["risk_tertile"] = "all"
    frame["risk_tertile"] = frame["risk_tertile"].astype(str).fillna("all")
    frame["obs_stratum"] = np.where(
        frame["observed_event"], "observed event", "no observed event"
    )
    return frame


@torch.inference_mode()
def collect_openloop_death_risk(
    agent: JEPAAgent,
    loader: DataLoader,
    device: torch.device,
    max_horizon: int = 5,
) -> pd.DataFrame:
    """Death-only open-loop rollout (Fig 5D); same unroll as event-risk KM."""
    frame = collect_openloop_event_risk(
        agent,
        loader,
        device,
        event_names=("death_event",),
        max_horizon=max_horizon,
    )
    if frame.empty:
        return frame
    out = frame.rename(
        columns={
            "event_prob": "death_prob",
            "event_prob_h1": "death_prob_h1",
            "observed_event": "observed_death",
        }
    )
    out["obs_stratum"] = np.where(
        out["observed_death"], "observed death", "no observed death"
    )
    return out


def kaplan_meier_openloop_event_horizons(
    event_roll: pd.DataFrame,
    event: str,
    horizons: Sequence[int] = (1, 2, 3),
    *,
    prob_col: str = "event_prob",
    obs_col: str = "observed_event",
) -> pd.DataFrame:
    """KM-style survival by open-loop risk tertile at each horizon for one event.

    Cohort at horizon ``h`` is people with at least ``h`` observed at-risk
    steps (``n_event_steps``, else ``n_valid_steps``) — no padded short
    trajectories. Event = ever-observed label on the observed trajectory;
    time = last ``followup_wave``. Tertiles are cut within that h-specific set.
    """
    if event_roll.empty:
        return pd.DataFrame()
    use_all = event_roll
    if "event" in event_roll.columns:
        use_all = event_roll[event_roll["event"] == event]
    need = {"person_id", "horizon", prob_col, obs_col, "followup_wave"}
    if use_all.empty or not need.issubset(use_all.columns):
        return pd.DataFrame()
    step_col = next(
        (c for c in ("n_event_steps", "n_valid_steps") if c in use_all.columns),
        None,
    )
    rows: list[dict[str, Any]] = []
    for h in horizons:
        use = use_all[use_all["horizon"] == int(h)].copy()
        if step_col is not None:
            use = use[use[step_col] >= int(h)]
        if use.empty:
            continue
        person = (
            use.groupby("person_id", as_index=False)
            .agg(
                risk=(prob_col, "first"),
                event_flag=(obs_col, "max"),
                time=("followup_wave", "max"),
            )
            .dropna(subset=["risk", "time"])
        )
        person = person[person["time"] >= 0]
        if len(person) < 30:
            continue
        try:
            person["stratum"] = pd.qcut(
                person["risk"].rank(method="first"),
                q=3,
                labels=["low", "mid", "high"],
            )
        except ValueError:
            continue
        for stratum, group in person.groupby("stratum", observed=True):
            g = group.sort_values("time")
            n = len(g)
            alive = n
            cum_surv = 1.0
            for t, chunk in g.groupby("time", sort=True):
                d = int(chunk["event_flag"].sum())
                if alive <= 0:
                    break
                cum_surv *= 1.0 - d / alive
                rows.append(
                    {
                        "event": event,
                        "horizon": int(h),
                        "stratum": str(stratum),
                        "time": int(t),
                        "survival": cum_surv,
                        "at_risk": alive,
                        "events": d,
                        "n_persons": n,
                    }
                )
                alive -= len(chunk)
    return pd.DataFrame(rows)


def kaplan_meier_openloop_death_horizons(
    death_roll: pd.DataFrame,
    horizons: Sequence[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Death KM; accepts either death-specific or generic event-risk columns."""
    if death_roll.empty:
        return pd.DataFrame()
    if "death_prob" in death_roll.columns:
        return kaplan_meier_openloop_event_horizons(
            death_roll,
            event="death_event",
            horizons=horizons,
            prob_col="death_prob",
            obs_col="observed_death",
        )
    return kaplan_meier_openloop_event_horizons(
        death_roll, event="death_event", horizons=horizons
    )


def summarize_event_risk_rollout(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean open-loop P(event) by event × stratum × horizon (Fig 3/4/5 style).

    Expects columns from ``collect_openloop_event_risk``: ``event``, ``horizon``,
    ``event_prob``, ``risk_tertile``, ``obs_stratum``.
    """
    if frame.empty:
        return pd.DataFrame()
    if "event_prob" not in frame.columns:
        return pd.DataFrame()
    chunks = []
    group_base = ["event"] if "event" in frame.columns else []
    for col, kind in [("risk_tertile", "risk_tertile"), ("obs_stratum", "obs_stratum")]:
        if col not in frame.columns:
            continue
        agg = (
            frame.groupby(group_base + [col, "horizon"], as_index=False)
            .agg(
                event_prob=("event_prob", "mean"),
                event_prob_std=("event_prob", "std"),
                n=("event_prob", "size"),
            )
            .rename(columns={col: "stratum"})
        )
        agg["stratum_kind"] = kind
        chunks.append(agg)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def summarize_death_risk_rollout(frame: pd.DataFrame) -> pd.DataFrame:
    """Death-only summary for Fig 5D (``death_prob`` column)."""
    if frame.empty:
        return pd.DataFrame()
    if "death_prob" not in frame.columns and "event_prob" in frame.columns:
        frame = frame.rename(columns={"event_prob": "death_prob"})
    if "death_prob" not in frame.columns:
        return pd.DataFrame()
    chunks = []
    for col, name in [("risk_tertile", "risk_tertile"), ("obs_stratum", "obs_stratum")]:
        if col not in frame.columns:
            continue
        agg = (
            frame.groupby([col, "horizon"], as_index=False)
            .agg(
                death_prob=("death_prob", "mean"),
                death_prob_std=("death_prob", "std"),
                n=("death_prob", "size"),
            )
            .rename(columns={col: "stratum"})
        )
        agg["stratum_kind"] = name
        chunks.append(agg)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def summarize_latent_by_horizon(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean latent error per horizon, plus the same-space persistence reference.

    ``relative_mse`` < 1 means the predictor beats "assume the latent does not
    move". Raw ``latent_mse`` is only meaningful within one embedding space.
    """
    if frame.empty:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {
        "cosine_sim": ("cosine_sim", "mean"),
        "latent_mse": ("latent_mse", "mean"),
    }
    for column in ("persistence_cosine", "persistence_mse"):
        if column in frame.columns:
            aggregations[column] = (column, "mean")
    aggregations["n"] = ("cosine_sim", "size")
    summary = (
        frame.groupby(["model", "horizon"], as_index=False)
        .agg(**aggregations)
        .sort_values(["model", "horizon"])
    )
    if "persistence_mse" in summary.columns:
        denominator = summary["persistence_mse"].replace(0.0, np.nan)
        summary["relative_mse"] = summary["latent_mse"] / denominator
    return summary


def level_compare_metrics(
    jepa_levels: pd.DataFrame,
    persistence: pd.DataFrame,
    linear: pd.DataFrame,
    mlp_levels: pd.DataFrame,
) -> pd.DataFrame:
    if jepa_levels.empty:
        return pd.DataFrame()
    keys = ["person_id", "wave", "next_wave"]
    rows = []
    for name in PRIMARY_LEVELS:
        group = jepa_levels[jepa_levels["outcome"] == name].copy()
        if group.empty:
            continue
        group["person_id"] = group["person_id"].astype(str)
        pcol = f"persistence_raw__{name}"
        lcol = f"linear_raw__{name}"
        if pcol in persistence.columns:
            group = group.merge(
                persistence[keys + [pcol]].assign(person_id=lambda d: d["person_id"].astype(str)),
                on=keys,
                how="left",
            )
            group = group.rename(columns={pcol: "persistence_raw"})
        if lcol in linear.columns:
            group = group.merge(
                linear[keys + [lcol]].assign(person_id=lambda d: d["person_id"].astype(str)),
                on=keys,
                how="left",
            )
            group = group.rename(columns={lcol: "linear_raw"})
        mcol = f"mlp_raw__{name}"
        if not mlp_levels.empty and mcol in mlp_levels.columns:
            m = mlp_levels[keys + [mcol]].copy()
            m["person_id"] = m["person_id"].astype(str)
            group = group.merge(m, on=keys, how="left")
            group = group.rename(columns={mcol: "mlp_raw"})
        elif not mlp_levels.empty and "mlp_raw" in mlp_levels.columns and "outcome" in mlp_levels.columns:
            # Legacy long format from world-model MLP clinical head.
            m = mlp_levels[mlp_levels["outcome"] == name][keys + ["mlp_raw"]].copy()
            m["person_id"] = m["person_id"].astype(str)
            group = group.merge(m, on=keys, how="left")
        obs = group["observed_raw"].to_numpy(float)
        finite = np.isfinite(obs)
        for model, col in [
            ("JEPA", "predicted_raw"),
            ("Persistence", "persistence_raw"),
            ("Linear", "linear_raw"),
            ("MLP", "mlp_raw"),
        ]:
            if col not in group:
                continue
            pred = pd.to_numeric(group[col], errors="coerce").to_numpy(float)
            msk = finite & np.isfinite(pred)
            if msk.sum() < 30:
                continue
            mae = float(np.mean(np.abs(pred[msk] - obs[msk])))
            mse = float(np.mean((pred[msk] - obs[msk]) ** 2))
            rmse = float(math.sqrt(mse))
            rows.append(
                {
                    "outcome": name,
                    "label": LEVEL_OUTCOMES.get(name, name),
                    "model": model,
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse,
                    "n": int(msk.sum()),
                }
            )
    return pd.DataFrame(rows)


def prepare_linear_and_persistence(
    *,
    train_data: Path,
    test_data: Path,
    preprocessing_path: Path,
    preprocessing: Mapping[str, Any],
    baseline_dir: Path,
    seed: int,
    skip_linear_baseline: bool,
    refit_baselines: bool,
    event_names: Sequence[str],
    output_dir: Path,
    skip_mlp: bool = False,
    level_names: Sequence[str] | None = None,
    binary_pos_weight_max: float = DEFAULT_BINARY_POS_WEIGHT_MAX,
    allowed_feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Logistic/mean cache + persistence + Ridge levels + tabular MLP levels/events."""
    pos_max = float(binary_pos_weight_max)
    baseline, manifest = load_or_prepare_baselines(
        train_data=train_data,
        test_data=test_data,
        preprocessing_path=Path(preprocessing_path),
        baseline_dir=baseline_dir,
        output_dir=output_dir,
        seed=seed,
        skip_linear_baseline=skip_linear_baseline,
        refit_baselines=refit_baselines,
        event_names=event_names,
        binary_pos_weight_max=pos_max,
        allowed_feature_names=allowed_feature_names,
    )
    test = read_table(test_data)
    train = read_table(train_data)
    persistence = _persistence_raw_from_table(
        test, preprocessing, level_names=level_names
    )
    linear = fit_linear_level_baselines(
        train,
        test,
        preprocessing,
        seed,
        level_names=level_names,
        allowed_feature_names=allowed_feature_names,
    )
    mlp_tabular = load_or_fit_mlp_tabular(
        train=train,
        test=test,
        preprocessing=preprocessing,
        baseline_dir=baseline_dir,
        seed=seed,
        event_names=event_names,
        refit=refit_baselines,
        skip=skip_mlp,
        level_names=level_names,
        binary_pos_weight_max=pos_max,
        allowed_feature_names=allowed_feature_names,
    )
    return baseline, persistence, linear, mlp_tabular, manifest
