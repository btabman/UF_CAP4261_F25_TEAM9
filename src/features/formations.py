"""
formations.py

Utilities for building simple formation embeddings and a KNN model
from player-level features.

We:
- take the FIRST frame of each (game_id, play_id, player_side),
- use compute_formation_targets(x_norm, y_norm) to summarize the formation,
- fit a KNN model over those formation vectors,
- provide helpers to save/load and query similar formations.

Intended to be called from notebooks, not via CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import polars as pl
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import hdbscan

from src.features.play_features import compute_formation_targets


# -------------------------------------------------------------------
# Core: build formation feature matrix from players parquet
# -------------------------------------------------------------------

def build_formation_matrix(
    parquet_path: str | Path,
    side: Optional[str] = None,
) -> Tuple[np.ndarray, pl.DataFrame]:
    """
    Build a formation feature matrix from a *player-level* parquet file.

    Each row in the output corresponds to a unique
    (game_id, play_id, player_side) formation, using ONLY the
    FIRST frame of that play for that side.

    Args
    ----
    parquet_path : path to players parquet
        e.g. "data/processed/players_test.parquet" or "data/processed/players_train.parquet"
    side : "offense", "defense", or None
        If provided, filter to that side before forming formations.
        If None, include both and you can filter later.

    Returns
    -------
    X : np.ndarray of shape [N_formations, D]
        Formation feature vectors; here D = 5:
        [width_x, width_y, center_x, center_y, mean_dist_to_center]
    meta : pl.DataFrame of shape [N_formations, 3]
        Columns: ["game_id", "play_id", "player_side"]
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Players parquet not found: {parquet_path}")

    df = pl.read_parquet(parquet_path)

    required_cols = {"game_id", "play_id", "frame_id", "player_side", "x_norm", "y_norm"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"build_formation_matrix requires columns {sorted(required_cols)}, "
            f"missing: {sorted(missing)}"
        )

    if side is not None:
        df = df.filter(pl.col("player_side") == side)

    if df.height == 0:
        raise ValueError("No rows left after applying side filter; check 'side' argument.")

    # ----------------------------------------------------------------
    # 1) Get FIRST frame per (game, play, side)
    # ----------------------------------------------------------------
    first_frames = (
        df.group_by(["game_id", "play_id", "player_side"])
          .agg(pl.col("frame_id").min().alias("first_frame_id"))
    )

    df_first = (
        df.join(
            first_frames,
            on=["game_id", "play_id", "player_side"],
            how="inner",
        )
        .filter(pl.col("frame_id") == pl.col("first_frame_id"))
        .drop("first_frame_id")
    )

    # ----------------------------------------------------------------
    # 2) Partition by (game_id, play_id, player_side) and compute
    #    formation feature vector per group using compute_formation_targets
    # ----------------------------------------------------------------
    groups = df_first.partition_by(
        ["game_id", "play_id", "player_side"],
        as_dict=False,
        maintain_order=True,
    )

    feat_list: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    for g in groups:
        # compute_formation_targets expects x_norm, y_norm columns
        # (they should already exist in players parquet)
        fvec = compute_formation_targets(g)
        feat_list.append(fvec)

        meta_rows.append(
            {
                "game_id": g["game_id"][0],
                "play_id": g["play_id"][0],
                "player_side": g["player_side"][0],
            }
        )

    X = np.stack(feat_list, axis=0)  # [N_formations, 5]
    meta = pl.DataFrame(meta_rows)

    return X, meta


# -------------------------------------------------------------------
# KNN fitting / saving / loading
# -------------------------------------------------------------------

def fit_formation_knn(
    X: np.ndarray,
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> NearestNeighbors:
    """
    Fit a NearestNeighbors KNN model on the formation feature matrix.

    Args
    ----
    X : [N_formations, D]
    n_neighbors : how many neighbors to precompute for queries
    metric : distance metric for KNN

    Returns
    -------
    model : sklearn.neighbors.NearestNeighbors
    """
    if X.ndim != 2:
        raise ValueError(f"X should be 2D, got shape {X.shape}")

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
    )
    model.fit(X)
    return model


def save_knn_model(
    model: NearestNeighbors,
    meta: pl.DataFrame,
    X: np.ndarray,
    path: str | Path,
) -> None:
    """
    Save a KNN model + metadata + features to disk via pickle.

    This makes it easy to query later without rebuilding everything.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "meta": meta.to_dicts(),  # list of {game_id, play_id, player_side}
        "features": X,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_knn_model(path: str | Path) -> Tuple[NearestNeighbors, pl.DataFrame, np.ndarray]:
    """
    Load a previously saved KNN model + meta + features.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No KNN file found at: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    model: NearestNeighbors = payload["model"]
    meta = pl.DataFrame(payload["meta"])
    X = np.asarray(payload["features"], dtype=np.float32)

    return model, meta, X


def train_formation_knn(
    players_parquet: str | Path,
    side: Optional[str] = None,
    n_neighbors: int = 10,
    metric: str = "euclidean",
    save_path: Optional[str | Path] = None,
) -> Tuple[NearestNeighbors, pl.DataFrame, np.ndarray]:
    """
    High-level helper: build formation matrix, fit KNN, optionally save.

    Args
    ----
    players_parquet : players-level feature parquet
    side : "Offense", "Defense", or None
        If None, includes both sides; if you want separate models,
        call this twice with side="Offense" and side="Defense".
    n_neighbors : neighbors for NearestNeighbors
    metric : distance metric
    save_path : if provided, saves a pickle with model/meta/features

    Returns
    -------
    model, meta, X
    """
    X, meta = build_formation_matrix(players_parquet, side=side)
    model = fit_formation_knn(X, n_neighbors=n_neighbors, metric=metric)

    if save_path is not None:
        save_knn_model(model, meta, X, save_path)

    return model, meta, X


# -------------------------------------------------------------------
# Query helpers
# -------------------------------------------------------------------

def knn_neighbors_by_index(
    model: NearestNeighbors,
    meta: pl.DataFrame,
    X: np.ndarray,
    idx: int,
    k: int = 5,
) -> pl.DataFrame:
    """
    Retrieve the K nearest neighbors for X[idx].
    """
    dists, inds = model.kneighbors(X[idx : idx+1], n_neighbors=k)

    inds_list = inds[0].tolist()
    dists_list = dists[0].tolist()

    # Polars: select rows by index array
    neigh_meta = meta[inds_list].with_columns(
        pl.Series("distance", dists_list)
    )

    return neigh_meta


def knn_neighbors_by_key(
    model: NearestNeighbors,
    meta: pl.DataFrame,
    X: np.ndarray,
    game_id: int,
    play_id: int,
    player_side: str,
    k: int = 5,
) -> pl.DataFrame:
    """
    Convenience wrapper: query neighbors by (game_id, play_id, player_side).

    Looks up the formation index for that key, then calls knn_neighbors_by_index.
    """
    # Find index where meta matches the requested key
    mask = (
        (meta["game_id"] == game_id)
        & (meta["play_id"] == play_id)
        & (meta["player_side"] == player_side)
    )

    matches = meta.filter(mask)
    if matches.height == 0:
        raise ValueError(
            f"No formation found for (game_id={game_id}, play_id={play_id}, side={player_side})"
        )

    # Use the *first* match index
    idx = int(matches.row(0, named=True)["index"]) if "index" in meta.columns else None
    if idx is None:
        # If meta has no explicit index column, we'll reconstruct it
        # by joining with row numbers.
        meta_with_idx = meta.with_row_count("idx")
        m2 = meta_with_idx.filter(
            (pl.col("game_id") == game_id)
            & (pl.col("play_id") == play_id)
            & (pl.col("player_side") == player_side)
        )
        idx = int(m2["idx"][0])

    return knn_neighbors_by_index(model, meta, X, idx, k=k)

def build_formation_vectors(
    players_parquet: str | Path,
    side: str | None = "Offense",
    *,
    game_col: str = "game_id",
    play_col: str = "play_id",
    frame_col: str = "frame_id",
    side_col: str = "player_side",
    x_col: str = "x_norm",
    y_col: str = "y_norm",
    max_players: int = 11,
    frame_policy: str = "first",  # "first" or "last"
) -> Tuple[pl.DataFrame, np.ndarray]:
    """
    Build fixed-length formation vectors per (game_id, play_id, side).

    Each formation vector is:
        [x1, y1, x2, y2, ..., xN, yN]  (padded to max_players players)

    Args:
        players_parquet: Path to processed players parquet (e.g. players_test.parquet)
        side: Filter to a specific side ("Offense", "Defense") or None for both.
        game_col, play_col, frame_col, side_col: column names for IDs.
        x_col, y_col: normalized coordinate columns to use.
        max_players: max players per side (NFL = 11).
        frame_policy: "first" or "last" frame per (game, play, side).

    Returns:
        meta: polars DataFrame, one row per formation:
              [game_id, play_id, player_side, frame_id, n_players]
        X:    numpy array of shape [n_formations, 2 * max_players]
    """
    path = Path(players_parquet)
    df = pl.read_parquet(path)

    # Optional side filter
    if side is not None:
        df = df.filter(pl.col(side_col) == side)

    if df.is_empty():
        raise ValueError(f"No rows found for side={side!r} in {players_parquet}")

    keys = [game_col, play_col, side_col]

    # Choose which frame to represent the formation
    if frame_policy == "first":
        frames = (
            df.group_by(keys)
              .agg(pl.col(frame_col).min().alias(frame_col))
        )
    elif frame_policy == "last":
        frames = (
            df.group_by(keys)
              .agg(pl.col(frame_col).max().alias(frame_col))
        )
    else:
        raise ValueError(f"Unsupported frame_policy={frame_policy!r}, use 'first' or 'last'.")

    # Join back to get only the chosen frame for each (game, play, side)
    df_sel = df.join(frames, on=keys + [frame_col], how="inner")

    # Group by team/formation
    gb = df_sel.group_by(keys, maintain_order=True)

    meta_rows: list[dict] = []
    vectors: list[np.ndarray] = []

    # Polars version difference: iteration over group_by yields (keys, df) or just df
    for group in gb:
        if isinstance(group, tuple):
            # Old-style Polars: (group_key, group_df)
            _, g = group
        else:
            # New-style: just the group DataFrame
            g = group

        # Sort players in a consistent way (across-field then down-field)
        g_sorted = g.sort([y_col, x_col])

        n_players = g_sorted.height
        g_used = g_sorted.head(max_players)

        xs = g_used[x_col].to_numpy()
        ys = g_used[y_col].to_numpy()

        # Pad if fewer than max_players
        if n_players < max_players:
            pad = max_players - n_players
            xs = np.pad(xs, (0, pad), constant_values=0.0)
            ys = np.pad(ys, (0, pad), constant_values=0.0)

        # Build [x1,y1,x2,y2,...]
        vec = np.empty(2 * max_players, dtype=np.float32)
        vec[0::2] = xs
        vec[1::2] = ys
        vectors.append(vec)

        meta_rows.append(
            {
                game_col: g_sorted[game_col][0],
                play_col: g_sorted[play_col][0],
                side_col: g_sorted[side_col][0],
                frame_col: g_sorted[frame_col][0],
                "n_players": int(n_players),
            }
        )

    if not vectors:
        raise ValueError("No formations found after grouping.")

    meta = pl.from_dicts(meta_rows)
    X = np.vstack(vectors).astype(np.float32)

    return meta, X


def hdbscan_formations(
    players_parquet: str | Path,
    side: str | None = "Offense",
    *,
    max_players: int = 11,
    frame_policy: str = "first",
    min_cluster_size: int = 30,
    min_samples: int | None = None,
    metric: str = "euclidean",
    cluster_selection_epsilon: float = 0.0,
    save_model_path: str | Path | None = None,
) -> tuple["hdbscan.HDBSCAN", pl.DataFrame]:
    """
    Run HDBSCAN clustering on formation vectors.

    Args:
        players_parquet: processed players parquet (e.g. players_test.parquet)
        side: "Offense", "Defense", or None for both.
        max_players: number of players per side in vector (pads/truncates).
        frame_policy: "first" or "last" frame per (game, play, side).
        min_cluster_size: HDBSCAN min_cluster_size param.
        min_samples: HDBSCAN min_samples param (None -> defaults to min_cluster_size).
        metric: distance metric (usually "euclidean").
        cluster_selection_epsilon: HDBSCAN epsilon (0.0 is standard).
        save_model_path: optional path to save fitted model with joblib.

    Returns:
        model: fitted hdbscan.HDBSCAN instance
        meta_with_clusters: polars DataFrame = meta + ["cluster", "cluster_prob"]
            - cluster == -1 means 'noise' (unassigned)
    """
    meta, X = build_formation_vectors(
        players_parquet=players_parquet,
        side=side,
        max_players=max_players,
        frame_policy=frame_policy,
    )

    if X.shape[0] == 0:
        raise ValueError("No formation vectors to cluster.")

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True,
    )
    model.fit(X)

    labels = model.labels_
    probs = getattr(model, "probabilities_", None)

    cols = [meta]
    cols.append(pl.Series("cluster", labels))

    if probs is not None:
        cols.append(pl.Series("cluster_prob", probs))
    else:
        cols.append(pl.lit(None).alias("cluster_prob"))

    meta_with_clusters = meta.with_columns(cols[1:])  # add new columns

    # Optionally save model
    if save_model_path is not None:
        save_model_path = Path(save_model_path)
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required to save the model. Install with `pip install joblib`.")
        joblib.dump(model, save_model_path)

    return model, meta_with_clusters

def get_cluster_examples(
    meta_with_clusters: pl.DataFrame,
    cluster: int,
    n_examples: int = 10,
) -> pl.DataFrame:
    """
    Return up to n_examples plays from a given cluster.

    Args:
        meta_with_clusters: output DataFrame from hdbscan_formations
        cluster: cluster label to filter on (note: -1 = noise)
        n_examples: number of rows to return

    Returns:
        polars DataFrame with example (game_id, play_id, side, frame_id, ...)
    """
    return (
        meta_with_clusters
        .filter(pl.col("cluster") == cluster)
        .head(n_examples)
    )