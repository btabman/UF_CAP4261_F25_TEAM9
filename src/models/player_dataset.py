# src/models/player_dataset.py

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

# ----------------- teamframe feature list -----------------

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


# ----------------- basic loading helpers -----------------

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

    # Fill nulls in teamframe columns with 0 as a simple baseline
    df = df.with_columns([
        pl.col(c).fill_null(0.0).alias(c) for c in TEAMFRAME_COLS
    ])

    return df


def build_player_id_map(players_train: pl.DataFrame) -> Dict[int, int]:
    """
    Map nfl_id → dense integer [0, num_players).
    """
    unique_players = players_train.select("nfl_id").unique().to_series().to_list()
    return {int(pid): i for i, pid in enumerate(unique_players)}


# ----------------- dataset -----------------

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

        pdf = df.to_pandas()
        grouped = pdf.groupby(["game_id", "play_id", "nfl_id"], sort=False)

        # items: (game_id, play_id, nfl_id, x_seq, player_idx, y)
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

            # sanitize features: replace NaN/inf with 0
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            pid_int = int(nfl_id)
            if pid_int not in player_id_map:
                # for train we expect all known; for test we may skip unknowns
                continue
            player_idx = player_id_map[pid_int]

            if target_cols is not None:
                y = g[target_cols].iloc[-1].to_numpy(dtype=np.float32)
                if np.isnan(y).any() or np.isinf(y).any():
                    continue  # skip invalid targets
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
