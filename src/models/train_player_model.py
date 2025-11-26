# src/models/train_player_model.py

import os
import random
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, random_split, Subset

from .physics_transformer import (
    PhysicsTransformer,
    TransformerConfig,
    PhysicsTransformerFrames,
    TransformerFramesConfig,
)
from .player_dataset import (
    load_processed_data,
    join_teamframe,
    build_player_id_map,
    PlayerSequenceDataset,
    FrameSeqDataset,
    FrameSeqTestDataset,
    TEAMFRAME_COLS,
)

# ----------------- Global config -----------------

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LEN = 32              # per-play model
MAX_SEQ_LEN_FRAMES = 32       # frame-level model

BATCH_SIZE = 256
SEARCH_EPOCHS = 2
FINAL_EPOCHS = 8
VAL_FRACTION = 0.2

# Base per-frame features from players table
BASE_FEATURE_COLS = [
    "x_norm",
    "y_norm",
    "dir_norm",
    "o_norm",
    "s",
    "a",
    "vx",
    "vy",
    "ax",
    "ay",
    "dir_sin",
    "dir_cos",
    "o_sin",
    "o_cos",
    "absolute_yardline_number",
]

FEATURE_COLS = BASE_FEATURE_COLS + TEAMFRAME_COLS
TARGET_COLS = ["ball_land_x", "ball_land_y"]


# ----------------- Utils -----------------


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_heads_compatible(cfg: TransformerFramesConfig) -> TransformerFramesConfig:
    """
    Ensure cfg.num_heads divides cfg.model_dim, as required by MultiheadAttention.

    If not, adjust num_heads to a valid value (1, 2, 4, 8 that divides model_dim).
    """
    model_dim = cfg.model_dim
    num_heads = cfg.num_heads

    # Candidates we allow
    candidate_heads = [1, 2, 4, 8]
    valid_heads = [h for h in candidate_heads if model_dim % h == 0]
    if not valid_heads:
        valid_heads = [1]

    if num_heads in valid_heads:
        return cfg  # already fine

    # Pick the valid head count closest to the requested one
    new_heads = min(valid_heads, key=lambda h: abs(h - num_heads))
    print(
        f"[ensure_heads_compatible] Adjusting num_heads from {num_heads} "
        f"to {new_heads} for model_dim={model_dim}"
    )

    return TransformerFramesConfig(
        input_dim=cfg.input_dim,
        model_dim=cfg.model_dim,
        num_layers=cfg.num_layers,
        num_heads=new_heads,
        num_players=cfg.num_players,
        dropout=cfg.dropout,
    )

# ----------------- Per-play training pipeline -----------------


def build_dataloaders(
    players_joined: pl.DataFrame,
    player_id_map: Dict[int, int],
):
    dataset = PlayerSequenceDataset(
        df=players_joined,
        player_id_map=player_id_map,
        feature_cols=FEATURE_COLS,
        target_cols=TARGET_COLS,
        max_seq_len=MAX_SEQ_LEN,
    )

    val_size = int(len(dataset) * VAL_FRACTION)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # important for Windows + notebooks
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for _, _, _, x, pid, y in loader:
        x = x.to(DEVICE)
        pid = pid.to(DEVICE)
        y = y.to(DEVICE)

        if x.ndim == 2:
            x = x.unsqueeze(1)

        optimizer.zero_grad()
        preds = model(x, pid)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    n_batches = 0

    for _, _, _, x, pid, y in loader:
        x = x.to(DEVICE)
        pid = pid.to(DEVICE)
        y = y.to(DEVICE)

        if x.ndim == 2:
            x = x.unsqueeze(1)

        preds = model(x, pid)
        loss = criterion(preds, y)
        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


def sample_config(input_dim: int, num_players: int) -> TransformerConfig:
    model_dim = random.choice([64, 96, 128])
    num_layers = random.choice([1, 2, 3])
    num_heads = random.choice([2, 4])
    dropout = random.choice([0.0, 0.1])

    return TransformerConfig(
        input_dim=input_dim,
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_players=num_players,
        dropout=dropout,
    )


def hyperparam_search(
    players_joined: pl.DataFrame,
    player_id_map: Dict[int, int],
    num_trials: int = 4,
) -> TransformerConfig:
    """
    Original per-play hyperparam search (not frame-based).
    """
    print(f"Hyperparam search over {num_trials} trials")

    train_loader, val_loader = build_dataloaders(players_joined, player_id_map)

    best_cfg = None
    best_val = float("inf")

    for t in range(num_trials):
        cfg = sample_config(len(FEATURE_COLS), len(player_id_map))
        print(f"Trial {t+1}: {asdict(cfg)}")

        model = PhysicsTransformer(cfg).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()

        for epoch in range(SEARCH_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss = eval_one_epoch(model, val_loader, criterion)
            print(f"  Epoch {epoch+1}/{SEARCH_EPOCHS}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_cfg = cfg
            print(f"  👉 new best val {best_val:.4f}")

    print("Best cfg:", asdict(best_cfg))
    return best_cfg


def train_final_model(
    players_joined: pl.DataFrame,
    player_id_map: Dict[int, int],
    cfg: TransformerConfig,
) -> PhysicsTransformer:
    train_loader, val_loader = build_dataloaders(players_joined, player_id_map)

    model = PhysicsTransformer(cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    for epoch in range(FINAL_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = eval_one_epoch(model, val_loader, criterion)
        print(f"[FINAL] Epoch {epoch+1}/{FINAL_EPOCHS} train={train_loss:.4f}, val={val_loss:.4f}")

    return model


@torch.no_grad()
def predict_on_test(
    model: PhysicsTransformer,
    players_test_joined: pl.DataFrame,
    player_id_map: Dict[int, int],
    out_path: str,
):
    """
    Per-play predictions: one (x,y) per (game_id, play_id, nfl_id).
    """
    dataset = PlayerSequenceDataset(
        df=players_test_joined,
        player_id_map=player_id_map,
        feature_cols=FEATURE_COLS,
        target_cols=None,
        max_seq_len=MAX_SEQ_LEN,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model.eval()
    preds = []

    for game_id, play_id, nfl_id, x, pid in loader:
        x = x.to(DEVICE)
        pid = pid.to(DEVICE)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        y_hat = model(x, pid).cpu().numpy()
        for g, p, n, (px, py) in zip(game_id, play_id, nfl_id, y_hat):
            preds.append(
                dict(
                    game_id=int(g),
                    play_id=int(p),
                    nfl_id=int(n),
                    pred_x=float(px),
                    pred_y=float(py),
                )
            )

    import pandas as pd

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(preds).to_csv(out_path, index=False)
    print(f"Saved {len(preds)} per-play predictions to {out_path}")


# ----------------- Frame-level agentic search & training (new) -----------------


def build_small_frame_dataset(
    players_joined: pl.DataFrame,
    player_id_map: Dict[int, int],
    feature_cols,
    n_sequences: int = 300,
    seed: int = RANDOM_SEED,
) -> FrameSeqDataset:
    """
    Sample a small subset of sequences for frame-level agentic search.
    """
    uniq_keys = (
        players_joined
        .select(["game_id", "play_id", "nfl_id"])
        .unique()
        .sample(n=min(n_sequences, players_joined.height), seed=seed)
    )

    df_small = players_joined.join(
        uniq_keys,
        on=["game_id", "play_id", "nfl_id"],
        how="inner",
    )

    ds = FrameSeqDataset(
        df=df_small,
        player_id_map=player_id_map,
        feature_cols=feature_cols,
        max_seq_len=MAX_SEQ_LEN_FRAMES,
    )
    return ds


def eval_frame_transformer_cv(
    dataset: FrameSeqDataset,
    cfg: TransformerFramesConfig,
    k: int = 3,
    epochs: int = 3,
    batch_size: int = 64,
) -> float:
    """
    K-fold CV RMSE for a frame-level transformer config.
    """

    cfg = ensure_heads_compatible(cfg)
    indices = np.arange(len(dataset))
    from sklearn.model_selection import KFold  # local import to avoid hard dependency at top

    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    rmses = []

    for train_idx, val_idx in kf.split(indices):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        model = PhysicsTransformerFrames(cfg).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        crit = torch.nn.MSELoss()

        for _ in range(epochs):
            model.train()
            for _, _, _, x, y, pid in train_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pid = pid.to(DEVICE)

                opt.zero_grad()
                preds = model(x, pid)        # (B,T,2)
                loss = crit(preds, y)
                loss.backward()
                opt.step()

        # eval RMSE on validation
        model.eval()
        all_true = []
        all_pred = []
        with torch.no_grad():
            for _, _, _, x, y, pid in val_loader:
                x = x.to(DEVICE)
                pid = pid.to(DEVICE)
                preds = model(x, pid).cpu().numpy()
                all_pred.append(preds)
                all_true.append(y.numpy())

        all_true = np.concatenate(all_true, axis=0)
        all_pred = np.concatenate(all_pred, axis=0)
        rmse = float(np.sqrt(((all_true - all_pred) ** 2).mean()))
        rmses.append(rmse)

    return float(np.mean(rmses))




def agentic_search_frames(
    players_joined: pl.DataFrame,
    player_id_map: Dict[int, int],
    feature_cols=FEATURE_COLS,
    n_sequences: int = 300,
    trials: int = 4,
) -> dict:
    """
    Tiny frame-level agent that:
      - samples a small frame dataset,
      - tries a few transformer configs,
      - uses K-fold CV RMSE to pick the best,
      - does local random search around the current best.

    Returns:
      dict with keys:
        - 'cfg': best TransformerFramesConfig
        - 'score': best CV RMSE
        - 'history': list of per-trial results for plotting/debug
    """
    ds = build_small_frame_dataset(
        players_joined=players_joined,
        player_id_map=player_id_map,
        feature_cols=feature_cols,
        n_sequences=n_sequences,
    )
    input_dim = len(feature_cols)
    num_players = len(player_id_map)

    print("Frame dataset size (sequences):", len(ds))

    best = None
    history = []

    for t in range(trials):
        # ---- choose model_dim first ----
        if best is None:
            model_dim = random.choice([64, 96, 128])
            num_layers = random.choice([1, 2, 3])
            dropout = random.choice([0.0, 0.1])
        else:
            model_dim = max(32, best["cfg"].model_dim + random.choice([-32, 0, 32]))
            num_layers = max(1, best["cfg"].num_layers + random.choice([-1, 0, 1]))
            dropout = best["cfg"].dropout

        # ---- choose num_heads so it ALWAYS divides model_dim ----
        # allowed heads
        candidate_heads = [1, 2, 4, 8]
        valid_heads = [h for h in candidate_heads if model_dim % h == 0]
        if not valid_heads:
            valid_heads = [1]

        if best is None:
            num_heads = random.choice(valid_heads)
        else:
            # bias toward something near previous num_heads
            target = max(1, best["cfg"].num_heads + random.choice([-2, 0, 2]))
            num_heads = min(valid_heads, key=lambda h: abs(h - target))

        cfg = TransformerFramesConfig(
            input_dim=input_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_players=num_players,
            dropout=dropout,
        )

        print(f"\n[Frames Agentic] Trial {t+1}/{trials} with cfg: {cfg}")
        cv_rmse = eval_frame_transformer_cv(ds, cfg, k=3, epochs=1, batch_size=64)
        print(f"  CV RMSE (per-frame): {cv_rmse:.4f}")

        # record this trial for plotting
        history.append(
            {
                "trial": t + 1,
                "cv_rmse": float(cv_rmse),
                "model_dim": model_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "dropout": float(dropout),
            }
        )

        if best is None or cv_rmse < best["score"]:
            best = dict(cfg=cfg, score=cv_rmse)
            print("  👉 New best frame config:", best)

    best["history"] = history
    return best




def train_final_frame_model(
    players_joined: pl.DataFrame,
    player_id_map: Dict[int, int],
    cfg: TransformerFramesConfig,
    n_sequences: Optional[int] = None,
) -> PhysicsTransformerFrames:
    """
    Train a final frame-level model using the given config.

    If n_sequences is provided, trains on a sampled subset; otherwise, uses all.
    """

    cfg = ensure_heads_compatible(cfg)

    if n_sequences is not None:
        ds = build_small_frame_dataset(
            players_joined=players_joined,
            player_id_map=player_id_map,
            feature_cols=FEATURE_COLS,
            n_sequences=n_sequences,
        )
    else:
        ds = FrameSeqDataset(
            df=players_joined,
            player_id_map=player_id_map,
            feature_cols=FEATURE_COLS,
            max_seq_len=MAX_SEQ_LEN_FRAMES,
        )

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model = PhysicsTransformerFrames(cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = torch.nn.MSELoss()

    FINAL_EPOCHS_FRAMES = 5

    for epoch in range(FINAL_EPOCHS_FRAMES):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for _, _, _, x, y, pid in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pid = pid.to(DEVICE)

            opt.zero_grad()
            preds = model(x, pid)
            loss = crit(preds, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"[Frames FINAL] Epoch {epoch+1}/{FINAL_EPOCHS_FRAMES}: "
              f"train MSE={total_loss / max(n_batches, 1):.4f}")

    return model


@torch.no_grad()
def predict_frames_on_test(
    model: PhysicsTransformerFrames,
    players_test_joined: pl.DataFrame,
    player_id_map: Dict[int, int],
    out_path: str,
):
    """
    Frame-level predictions: one (pred_x, pred_y) per (game_id, play_id, nfl_id, frame_id)
    for the last MAX_SEQ_LEN_FRAMES frames of each sequence.
    """
    dataset = FrameSeqTestDataset(
        df=players_test_joined,
        player_id_map=player_id_map,
        feature_cols=FEATURE_COLS,
        max_seq_len=MAX_SEQ_LEN_FRAMES,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model.eval()
    preds = []

    for game_id, play_id, nfl_id, frame_ids, x, pid in loader:
        x = x.to(DEVICE)
        pid = pid.to(DEVICE)

        y_hat = model(x, pid).cpu().numpy()          # (B, T, 2)
        frame_ids_np = frame_ids.numpy()             # (B, T)

        for g, p, n, fr_seq, pred_seq in zip(game_id, play_id, nfl_id, frame_ids_np, y_hat):
            for fr_id, (px, py) in zip(fr_seq, pred_seq):
                preds.append(
                    dict(
                        game_id=int(g),
                        play_id=int(p),
                        nfl_id=int(n),
                        frame_id=int(fr_id),
                        pred_x=float(px),
                        pred_y=float(py),
                    )
                )

    import pandas as pd

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(preds).to_csv(out_path, index=False)
    print(f"Saved {len(preds)} frame-level predictions to {out_path}")


# ----------------- Optional CLI main (per-play) -----------------


def main():
    set_seed(RANDOM_SEED)

    players_train, players_test, teamframe_train, teamframe_test = load_processed_data()
    players_train_joined = join_teamframe(players_train, teamframe_train)
    players_test_joined = join_teamframe(players_test, teamframe_test)

    player_id_map = build_player_id_map(players_train_joined)

    best_cfg = hyperparam_search(players_train_joined, player_id_map, num_trials=4)
    model = train_final_model(players_train_joined, player_id_map, best_cfg)

    os.makedirs("models", exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "cfg": asdict(best_cfg), "player_id_map": player_id_map},
        "models/physics_transformer.pt",
    )

    predict_on_test(model, players_test_joined, player_id_map, "models/player_level_predictions.csv")


if __name__ == "__main__":
    main()
