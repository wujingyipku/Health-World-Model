"""Demo: load the checkpoint and evaluate the de-identified excerpt.

Runs an open-loop rollout (default H=3) from each person's first transition and
writes predicted death / ADL / IADL risks. Also rolls one single-lever
counterfactual that sets light_activity_frequency to the high code used in
Figure 5 from horizon 2 onward. Other actions stay at the origin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from code.data import TrajectoryDataset, collate_trajectories
from code.nn import JEPAAgent
from code.specs import ModelSpec


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=here / "data" / "cleaned" / "demo_transitions.parquet",
    )
    parser.add_argument("--spec", type=Path, default=here / "weights" / "model_spec.json")
    parser.add_argument("--config", type=Path, default=here / "weights" / "resolved_config.json")
    parser.add_argument("--checkpoint", type=Path, default=here / "weights" / "best_final.pt")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=here / "results" / "demo_predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_name = "cuda" if args.device == "gpu" else args.device
    device = torch.device(device_name)
    spec = ModelSpec.load(args.spec)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    agent = JEPAAgent(spec, cfg["model"])
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint["agent"])
    agent.to(device)
    agent.eval()
    world = agent.world

    if not args.data.exists():
        raise FileNotFoundError(
            f"Example table not found at {args.data}. "
            "Obtain the de-identified excerpt and place it there; see data/README.md."
        )
    dataset = TrajectoryDataset(args.data, spec)
    batch = collate_trajectories([dataset[i] for i in range(len(dataset))]).to(device)
    horizon = max(int(args.horizon), 1)
    # Figure 5 switches light activity at horizon 2. Raw codes are 0–4;
    # embedding index 0 is reserved for missing, so raw 4 maps to index 5.
    light_idx = next(
        i
        for i, feat in enumerate(spec.action_categorical)
        if feat.name == "light_activity_frequency"
    )
    light_high_raw = 4.0
    light_feature = spec.action_categorical[light_idx]
    light_high = {float(value): idx + 1 for idx, value in enumerate(light_feature.values)}[
        light_high_raw
    ]
    switch_horizon = 2

    with torch.no_grad():
        z_seq = world.encode_trajectory_waves(
            batch.state_cont,
            batch.state_cont_mask,
            batch.state_cat,
            batch.state_cat_mask,
            batch.valid,
        )
        static = world.encode_static(
            batch.static_cont[:, 0],
            batch.static_cont_mask[:, 0],
            batch.static_cat[:, 0],
            batch.static_cat_mask[:, 0],
        )

        rows: list[dict[str, object]] = []
        for scenario, override in (("observed", False), ("light_activity_high", True)):
            z = z_seq[:, 0]
            for step in range(horizon):
                action_cont = batch.action_cont[:, 0].clone()
                action_cat = batch.action_cat[:, 0].clone()
                action_cont_mask = batch.action_cont_mask[:, 0]
                action_cat_mask = batch.action_cat_mask[:, 0]
                if override and step + 1 >= switch_horizon:
                    action_cat[:, light_idx] = light_high
                    action_cat_mask = action_cat_mask.clone()
                    action_cat_mask[:, light_idx] = 1.0
                feature = world.predict_next(
                    z,
                    action_cont,
                    action_cont_mask,
                    action_cat,
                    batch.delta_t_norm[:, 0],
                    static_embed=static,
                )
                pred = world.clinical_heads.mean_dict(feature)
                for i, person_id in enumerate(batch.person_id):
                    if float(batch.valid[i, 0]) < 0.5:
                        continue
                    row = {
                        "person_id": person_id,
                        "scenario": scenario,
                        "horizon": step + 1,
                    }
                    for name in spec.reward_binary:
                        row[name] = float(pred[name][i].detach().cpu())
                    rows.append(row)
                z = feature

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    death = frame[frame["horizon"] == horizon].pivot(
        index="person_id", columns="scenario", values="death_event"
    )
    death["delta_light_activity_high"] = death["light_activity_high"] - death["observed"]
    print(f"persons={len(dataset)} horizon={horizon} device={device}")
    print(death.round(4).to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
