# src/models/deepsets_formation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import polars as pl
import numpy as np

# Reuse your dataset utilities from earlier code
# If you placed FormationDataset elsewhere, adjust the import accordingly.
from src.data.features import (
    normalize_rightward, FIELD_LENGTH, FIELD_WIDTH, angle_sin_cos_deg
)

# ---------- Formation dataset (play-side-frame → set of players) ----------

def compute_formation_targets(df: pl.DataFrame) -> np.ndarray:
    """
    Compute 5 formation targets using normalized coordinates:
    [width, height, center_x, center_y, mean_dist_to_center]
    """
    xs = df["x_norm"].to_numpy()
    ys = df["y_norm"].to_numpy()
    width  = xs.max() - xs.min()
    height = ys.max() - ys.min()
    cx = xs.mean()
    cy = ys.mean()
    mdc = np.sqrt((xs - cx)**2 + (ys - cy)**2).mean()
    return np.array([width, height, cx, cy, mdc], dtype=np.float32)

class FormationDataset(torch.utils.data.Dataset):
    """
    Produces samples by (game_id, play_id, frame_id, player_side).
    Each sample:
      X   : [n_max, D] padded player features
      mask: [n_max] 1.0 for valid players, 0.0 for padding
      y   : [5] formation targets
      meta: (game_id, play_id, frame_id, player_side)
    """
    def __init__(
        self,
        parquet_path: str,
        frame_selector: Optional[Callable[[pl.DataFrame], pl.DataFrame]] = None,
        n_max: int = 11,
    ):
        self.n_max = n_max
        df = pl.read_parquet(parquet_path)

        # normalize to rightward once; create x_norm/y_norm/dir_norm/o_norm
        df = (
            df.with_columns(
                pl.struct(["x", "y", "dir", "o", "play_direction"]).map_elements(
                    lambda r: normalize_rightward(
                        r["x"], r["y"], r["dir"], r["o"], r["play_direction"],
                        FIELD_LENGTH, FIELD_WIDTH
                    )
                ).alias("_norm")
            )
            .with_columns([
                pl.col("_norm").arr.get(0).alias("x_norm"),
                pl.col("_norm").arr.get(1).alias("y_norm"),
                pl.col("_norm").arr.get(2).alias("dir_norm"),
                pl.col("_norm").arr.get(3).alias("o_norm"),
            ])
            .drop("_norm")
        )

        if frame_selector is not None:
            df = frame_selector(df)

        # consistent order
        self.keys = ["game_id","play_id","frame_id","player_side"]
        df = df.sort(self.keys + ["x_norm", "y_norm"])

        # group into team formations per frame
        # Collect group DataFrames in a Python list for easy __getitem__
        groups = []
        for *key_vals, in df.group_by(self.keys):
            # pl 1.0+ yields group iterator: use .get_group / .agg trick
            # Simpler: rebuild by filter each unique key; for efficiency we materialize once
            pass
        # Efficient materialization:
        # Use group_by.apply to get per-group DataFrames:
        grouped = df.group_by(self.keys).apply(lambda g: g)
        # 'grouped' is a Series of DataFrames; convert to list
        self.groups = [g for g in grouped.to_list()]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        g: pl.DataFrame = self.groups[idx]

        # trig features
        dir_sc = np.array([angle_sin_cos_deg(d) for d in g["dir_norm"]], dtype=np.float32)
        o_sc   = np.array([angle_sin_cos_deg(d) for d in g["o_norm"]],   dtype=np.float32)
        dir_sin, dir_cos = dir_sc[:,0], dir_sc[:,1]
        o_sin,   o_cos   = o_sc[:,0],  o_sc[:,1]

        X = np.stack([
            g["x_norm"].to_numpy(),
            g["y_norm"].to_numpy(),
            g["s"].to_numpy(),      # speed
            g["a"].to_numpy(),      # accel
            dir_sin, dir_cos,
            o_sin,   o_cos,
        ], axis=1).astype(np.float32)  # [N, 8]

        # pad/truncate
        N, D = X.shape
        out  = np.zeros((self.n_max, D), dtype=np.float32)
        mask = np.zeros((self.n_max,), dtype=np.float32)
        n_out = min(N, self.n_max)
        out[:n_out]  = X[:n_out]
        mask[:n_out] = 1.0

        y = compute_formation_targets(g)  # [5]

        meta = tuple(g[k][0] for k in self.keys)  # (game_id, play_id, frame_id, player_side)

        return (
            torch.from_numpy(out),
            torch.from_numpy(mask),
            torch.from_numpy(y),
            meta,
        )

# --------------------------- DeepSets model ---------------------------

@dataclass
class DeepSetsConfig:
    in_dim: int = 8        # features per player
    hidden: int = 128      # phi hidden
    embed: int = 64        # phi output (set element embedding)
    pool: str = "mean"     # "mean" | "sum" | "max"
    rho_hidden: int = 128  # rho hidden
    out_dim: int = 5       # regression to formation targets

class DeepSets(nn.Module):
    def __init__(self, cfg: DeepSetsConfig):
        super().__init__()
        self.cfg = cfg

        self.phi = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.embed),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(cfg.embed, cfg.rho_hidden),
            nn.ReLU(),
            nn.Linear(cfg.rho_hidden, cfg.embed),
            nn.ReLU(),
        )

        self.head = nn.Linear(cfg.embed, cfg.out_dim)

    def pool_op(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        z:   [B, N, E]
        mask:[B, N]
        returns [B, E]
        """
        mask = mask.unsqueeze(-1)  # [B,N,1]
        z = z * mask
        if self.cfg.pool == "sum":
            pooled = z.sum(dim=1)
        elif self.cfg.pool == "max":
            # for masked max, set invalid to large negative
            z_masked = z + (1.0 - mask) * (-1e9)
            pooled, _ = z_masked.max(dim=1)
        else:
            summed = z.sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom
        return pooled

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        X:    [B, N, D]
        mask: [B, N]
        returns (embedding [B,E], preds [B,5])
        """
        B, N, D = X.shape
        z = self.phi(X)              # [B,N,E]
        s = self.pool_op(z, mask)    # [B,E]
        h = self.rho(s)              # [B,E]
        y = self.head(h)             # [B,5]
        return h, y
