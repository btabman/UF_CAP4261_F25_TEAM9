# src/models/player_dataset.py

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

# ----------------- Teamframe feature list -----------------

TEAMFRAME_COLS = [
    "formation_width",
    "formation_depth",
    "formation_x_mean",
    "formation_y_mean",
    "team_spread_mean",
    "depth_std",
    "width_std",
    "nn_teammate_mean",
    "nn_opponent_mean",
    "coverage_density_mean",
    "opp_within_3yds_sum",
    "opp_within_5yds_sum",
    "opp_within_7yds_sum",
    "tm_within_3yds_sum",
    "tm_within_5yds_sum",
    "tm_within_7yds_sum",
]


# ----------------- Basic loading helpers -----------------


def load_processed_data(base_dir: str = "data/processed"):
    """
    Load the processed parquet files from the given base directory.
    """
    players_train = pl.read_parquet(f"{base_dir}/players_train.parquet")
    players_test = pl.read_parquet(f"{base_dir}/players_test.parquet")
    teamframe_train = pl.read_parquet(f"{base_dir}/teamframe_train.parquet")
    teamframe_test = pl.read_parquet(f"{base_dir}/teamframe_test.parquet")
    return players_train, players_test, teamframe_train, teamframe_test


def join_teamframe(players: pl.DataFrame, teamframe: pl.DataFrame) -> pl.DataFrame:
    """
    Join teamframe features onto each player frame.
    Keys: (game_id, play_id, frame_id, player_side)
    """
    df = players.join(
        teamframe.select(["game_id", "play_id", "frame_id", "player_side"] + TEAMFRAME_COLS),
        on=["game_id", "play_id", "frame_id", "player_side"],
        how="left",
    )

    # Fill nulls in teamframe columns with 0.0 as a simple baseline
    df = df.with_columns(
        [pl.col(c).fill_null(0.0).alias(c) for c in TEAMFRAME_COLS]
    )

    return df


def build_player_id_map(players_train: pl.DataFrame) -> Dict[int, int]:
    """
    Map nfl_id → dense integer [0, num_players).
    """
    unique_players = players_train.select("nfl_id").unique().to_series().to_list()
    return {int(pid): i for i, pid in enumerate(unique_players)}


# ----------------- Per-play dataset -----------------


class PlayerSequenceDataset(Dataset):
    """
    One sequence per (game_id, play_id, nfl_id).

    If target_cols is not None:
        returns (game_id, play_id, nfl_id, x, player_idx, y)
    Else:
        returns (game_id, play_id, nfl_id, x, player_idx)
    """

    def __init__(
        self,
        df: pl.DataFrame,
        player_id_map: Dict[int, int],
        feature_cols: List[str],
        target_cols: Optional[List[str]] = None,
        max_seq_len: int = 32,
    ):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.max_seq_len = max_seq_len
        self.player_id_map = player_id_map

        needed_cols = ["game_id", "play_id", "nfl_id", "frame_id"] + feature_cols
        if target_cols is not None:
            needed_cols = needed_cols + target_cols

        df_small = df.select(needed_cols)

        pdf = df_small.to_pandas()
        grouped = pdf.groupby(["game_id", "play_id", "nfl_id"], sort=False)

        self.items: List[Tuple[int, int, int, np.ndarray, int, Optional[np.ndarray]]] = []

        for (game_id, play_id, nfl_id), g in grouped:
            g = g.sort_values("frame_id")

            # features (T, F)
            x = g[feature_cols].to_numpy(dtype=np.float32)

            # keep last max_seq_len frames
            if self.max_seq_len is not None:
                x = x[-self.max_seq_len :]
            T, F = x.shape
            if T < self.max_seq_len:
                pad = np.zeros((self.max_seq_len - T, F), dtype=np.float32)
                x = np.concatenate([pad, x], axis=0)

            # sanitize features
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            pid_int = int(nfl_id)
            if pid_int not in player_id_map:
                continue
            player_idx = player_id_map[pid_int]

            if target_cols is not None:
                y = g[target_cols].iloc[-1].to_numpy(dtype=np.float32)
                # skip sequences with NaN/Inf targets
                if np.isnan(y).any() or np.isinf(y).any():
                    continue
            else:
                y = None

            self.items.append((int(game_id), int(play_id), pid_int, x, player_idx, y))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        game_id, play_id, nfl_id, x, player_idx, y = self.items[idx]
        x_tensor = torch.from_numpy(x)  # (T, F)
        pid_tensor = torch.tensor(player_idx, dtype=torch.long)

        if y is not None:
            y_tensor = torch.from_numpy(y)
            return game_id, play_id, nfl_id, x_tensor, pid_tensor, y_tensor
        else:
            return game_id, play_id, nfl_id, x_tensor, pid_tensor


# ----------------- Frame-level datasets (new) -----------------


class FrameSeqDataset(Dataset):
    """
    Per (game_id, play_id, nfl_id) sequence, using last max_seq_len frames.

    X: (T, F)  features
    Y: (T, 2)  targets: (x_norm, y_norm) per frame
    """

    def __init__(
        self,
        df: pl.DataFrame,
        player_id_map: Dict[int, int],
        feature_cols: List[str],
        max_seq_len: int = 32,
    ):
        self.feature_cols = feature_cols
        self.max_seq_len = max_seq_len
        self.player_id_map = player_id_map

        base_cols = ["game_id", "play_id", "nfl_id", "frame_id", "x_norm", "y_norm"]

        # Deduplicate, in order: base_cols first, then any extra features
        needed_cols = list(dict.fromkeys(base_cols + feature_cols))
        df_small = df.select(needed_cols)

        pdf = df_small.to_pandas()
        grouped = pdf.groupby(["game_id", "play_id", "nfl_id"], sort=False)

        self.items: List[Tuple[int, int, int, np.ndarray, np.ndarray, int]] = []

        for (game_id, play_id, nfl_id), g in grouped:
            g = g.sort_values("frame_id")

            x = g[feature_cols].to_numpy(dtype=np.float32)
            y = g[["x_norm", "y_norm"]].to_numpy(dtype=np.float32)

            if len(g) < max_seq_len:
                # skip very short sequences for simplicity
                continue

            x = x[-max_seq_len:]
            y = y[-max_seq_len:]

            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            pid_int = int(nfl_id)
            if pid_int not in player_id_map:
                continue
            player_idx = player_id_map[pid_int]

            self.items.append((int(game_id), int(play_id), pid_int, x, y, player_idx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        game_id, play_id, nfl_id, x, y, player_idx = self.items[idx]
        x_tensor = torch.from_numpy(x)  # (T, F)
        y_tensor = torch.from_numpy(y)  # (T, 2)
        pid_tensor = torch.tensor(player_idx, dtype=torch.long)
        return game_id, play_id, nfl_id, x_tensor, y_tensor, pid_tensor


class FrameSeqTestDataset(Dataset):
    """
    Like FrameSeqDataset but without targets; keeps frame_ids so they can be attached to predictions.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        player_id_map: Dict[int, int],
        feature_cols: List[str],
        max_seq_len: int = 32,
    ):
        self.feature_cols = feature_cols
        self.max_seq_len = max_seq_len
        self.player_id_map = player_id_map

        # Base columns plus features, de-duplicated
        base_cols = ["game_id", "play_id", "nfl_id", "frame_id"]
        needed_cols = list(dict.fromkeys(base_cols + feature_cols))

        df_small = df.select(needed_cols)

        pdf = df_small.to_pandas()
        grouped = pdf.groupby(["game_id", "play_id", "nfl_id"], sort=False)

        self.items: List[Tuple[int, int, int, np.ndarray, np.ndarray, int]] = []

        for (game_id, play_id, nfl_id), g in grouped:
            g = g.sort_values("frame_id")

            if len(g) < max_seq_len:
                continue

            g_tail = g.iloc[-max_seq_len:]
            x = g_tail[feature_cols].to_numpy(dtype=np.float32)
            frame_ids = g_tail["frame_id"].to_numpy(dtype=np.int32)

            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            pid_int = int(nfl_id)
            if pid_int not in player_id_map:
                continue
            player_idx = player_id_map[pid_int]

            self.items.append((int(game_id), int(play_id), pid_int, frame_ids, x, player_idx))

    def __len__(self) -> int:
        # 🔑 this is what DataLoader needs
        return len(self.items)

    def __getitem__(self, idx):
        game_id, play_id, nfl_id, frame_ids, x, player_idx = self.items[idx]
        x_tensor = torch.from_numpy(x)          # (T, F)
        frame_ids_tensor = torch.from_numpy(frame_ids)  # (T,)
        pid_tensor = torch.tensor(player_idx, dtype=torch.long)
        return game_id, play_id, nfl_id, frame_ids_tensor, x_tensor, pid_tensor

