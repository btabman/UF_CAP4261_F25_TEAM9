"""
Module for play-level feature processing and dataset creation.
Handles formation embeddings and team-frame data preparation.
"""
#%%
from __future__ import annotations

import polars as pl
import numpy as np
from typing import Optional, Tuple, List, Callable
import torch
from torch.utils.data import Dataset

from src.features.features import (
    normalize_rightward,   # (x,y,dir,o,play_dir, FIELD_LENGTH, FIELD_WIDTH) -> (x',y',dir',o')
    angle_sin_cos_deg,     # scalar helper; we’ll use vectorized path below
    FIELD_LENGTH,
    FIELD_WIDTH,
)


# ---------------------------------------------------------------------
# Formation features (per frame, per side), then join back to rows
#    Adds: formation_width, formation_depth, formation_x_mean, formation_y_mean,
#          distance_to_formation_center, relative_formation_depth, relative_formation_width
# Needs: game_id, play_id, frame_id, player_side, x, y
#%%
def add_formation_features(
    df: pl.DataFrame,
    game="game_id",
    play="play_id",
    frame="frame_id",
    side="player_side",
    x="x",
    y="y",
) -> pl.DataFrame:
    keys = [game, play, frame, side]

    agg = (
        df.select([pl.col(c) for c in keys] + [pl.col(x), pl.col(y)])
          .group_by(keys)
          .agg([
              pl.col(x).max().alias("_x_max"),
              pl.col(x).min().alias("_x_min"),
              pl.col(y).max().alias("_y_max"),
              pl.col(y).min().alias("_y_min"),
              pl.col(x).mean().alias("_x_mean"),
              pl.col(y).mean().alias("_y_mean"),
          ])
          .with_columns([
              (pl.col("_y_max") - pl.col("_y_min")).alias("formation_width"),
              (pl.col("_x_max") - pl.col("_x_min")).alias("formation_depth"),
              pl.col("_x_mean").alias("formation_x_mean"),
              pl.col("_y_mean").alias("formation_y_mean"),
          ])
          .select([game, play, frame, side, "formation_width", "formation_depth",
                   "formation_x_mean", "formation_y_mean"])
    )

    out = df.join(agg, on=[game, play, frame, side], how="left")
    out = out.with_columns([
        ((pl.col(x) - pl.col("formation_x_mean"))**2 +
         (pl.col(y) - pl.col("formation_y_mean"))**2).sqrt().alias("distance_to_formation_center"),
        (pl.col(x) - pl.col("formation_x_mean")).alias("relative_formation_depth"),
        (pl.col(y) - pl.col("formation_y_mean")).alias("relative_formation_width"),
    ])
    return out


#%%
def compute_formation_targets(df: pl.DataFrame) -> np.ndarray:
    """
    Compute formation-level target features for embedding supervision.
    Uses normalized coordinates (x_norm, y_norm).
    Returns [width_x, width_y, center_x, center_y, mean_dist_to_center].
    """
    if df.height == 0:
        return np.array([0, 0, 0, 0, 0], dtype=np.float32)

    xs = df["x_norm"].to_numpy()
    ys = df["y_norm"].to_numpy()

    # Guard against NaNs
    mask = np.isfinite(xs) & np.isfinite(ys)
    if not mask.any():
        return np.array([0, 0, 0, 0, 0], dtype=np.float32)

    xs = xs[mask]
    ys = ys[mask]

    width_x = float(xs.max() - xs.min())
    width_y = float(ys.max() - ys.min())
    cx = float(xs.mean())
    cy = float(ys.mean())
    dist_center = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).mean())
    return np.array([width_x, width_y, cx, cy, dist_center], dtype=np.float32)

#%%
class FormationDataset(Dataset):
    """
    Produces one sample per (game_id, play_id, frame_id, player_side).
    Each sample: (X[n_max, D], mask[n_max], targets[5], meta).
      - X columns: [x_norm, y_norm, s, a, dir_sin, dir_cos, o_sin, o_cos]
    """
    def __init__(
        self,
        parquet_path: str,
        frame_selector: Optional[Callable[[pl.DataFrame], pl.DataFrame]] = None,
        n_max: int = 11,
    ):
        df = pl.read_parquet(parquet_path)

        if frame_selector is not None:
            df = frame_selector(df)

        cols = set(df.columns)

        # If normalized columns already exist, use them as-is.
        has_pre_norm = {"x_norm", "y_norm", "dir_norm", "o_norm"}.issubset(cols)

        if not has_pre_norm:
            # Fall back to normalizing from raw columns.
            needed_raw = {"x", "y", "dir", "o", "play_direction"}
            if not needed_raw.issubset(cols):
                missing = needed_raw - cols
                raise ValueError(
                    "FormationDataset needs either normalized cols "
                    "(x_norm,y_norm,dir_norm,o_norm) OR raw cols "
                    "(x,y,dir,o,play_direction). Missing: "
                    + ", ".join(sorted(missing))
                )

            def _safe_norm(s: dict[str, float | str]) -> tuple[float, float, float, float]:
                xn, yn, dn, on = normalize_rightward(
                    s["x"], s["y"], s["dir"], s["o"], s["play_direction"], FIELD_LENGTH, FIELD_WIDTH
                )
                return float(xn), float(yn), float(dn), float(on)

            df = (
                df.with_columns([
                    pl.struct(["x", "y", "dir", "o", "play_direction"])
                      .map_elements(_safe_norm, return_dtype=pl.Array(pl.Float64, 4))
                      .alias("_norm")
                ])
                .with_columns([
                    pl.col("_norm").arr.get(0).alias("x_norm"),
                    pl.col("_norm").arr.get(1).alias("y_norm"),
                    pl.col("_norm").arr.get(2).alias("dir_norm"),
                    pl.col("_norm").arr.get(3).alias("o_norm"),
                ])
                .drop("_norm")
            )

        # Stable sort (ordering won’t matter to DeepSets, but helps packing determinism)
        self.keys = ["game_id", "play_id", "frame_id", "player_side"]
        df = df.sort(self.keys + ["x_norm", "y_norm"])

        # Partition into per-(game,play,frame,side) groups
        self.groups: List[pl.DataFrame] = df.partition_by(self.keys, as_dict=False, maintain_order=True)
        self.n_max = int(n_max)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", tuple]:
        g: pl.DataFrame = self.groups[idx]

        dir_rad = np.deg2rad(g["dir_norm"].to_numpy())
        o_rad   = np.deg2rad(g["o_norm"].to_numpy())
        dir_sin, dir_cos = np.sin(dir_rad), np.cos(dir_rad)
        o_sin,   o_cos   = np.sin(o_rad),   np.cos(o_rad)

        features = np.stack([
            g["x_norm"].to_numpy(),
            g["y_norm"].to_numpy(),
            g["s"].to_numpy(),
            g["a"].to_numpy(),
            dir_sin, dir_cos,
            o_sin, o_cos,
        ], axis=1).astype(np.float32)  # [N, 8]

        N, D = features.shape
        N_out = min(N, self.n_max)
        X = np.zeros((self.n_max, D), dtype=np.float32)
        mask = np.zeros((self.n_max,), dtype=np.float32)
        if N_out > 0:
            X[:N_out] = features[:N_out]
            mask[:N_out] = 1.0

        targets = compute_formation_targets(g)
        meta = tuple(g[k][0] for k in self.keys)

        if torch is not None:
            return (
                torch.from_numpy(X),
                torch.from_numpy(mask),
                torch.from_numpy(targets),
                meta,
            )
        else:
            return X, mask, targets, meta

# %%
