# train_player_model.py

import os
import random
from dataclasses import asdict
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .physics_transformer import PhysicsTransformer, TransformerConfig
from .player_dataset import (
    load_processed_data,
    join_teamframe,
    build_player_id_map,
    PlayerSequenceDataset,
    TEAMFRAME_COLS,
)


# ----------------- config -----------------

RANDOM_SEED = 42
MAX_SEQ_LEN = 32
BATCH_SIZE = 256
SEARCH_EPOCHS = 2      # short search
FINAL_EPOCHS = 8       # longer training for best config
VAL_FRACTION = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose features
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

# 2D regression target; swap to other cols if desired
TARGET_COLS = ["ball_land_x", "ball_land_y"]


# ----------------- utilities -----------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    players_joined,
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
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
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
            x = x.unsqueeze(1)  # (B,1,F)

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


# ----------------- tiny hyperparameter "agent" -----------------

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
    players_joined,
    player_id_map: Dict[int, int],
    num_trials: int = 4,
) -> TransformerConfig:
    print(f"Starting tiny hyperparam search over {num_trials} trials.")

    best_cfg = None
    best_val = float("inf")

    train_loader, val_loader = build_dataloaders(players_joined, player_id_map)

    for t in range(num_trials):
        cfg = sample_config(input_dim=len(FEATURE_COLS), num_players=len(player_id_map))
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
            print(f"  🔁 New best config with val {best_val:.4f}")

    print("Best config:", asdict(best_cfg), "val_loss:", best_val)
    return best_cfg


# ----------------- final training & test preds -----------------

def train_final_model(players_joined, player_id_map: Dict[int, int], cfg: TransformerConfig):
    train_loader, val_loader = build_dataloaders(players_joined, player_id_map)

    model = PhysicsTransformer(cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    for epoch in range(FINAL_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = eval_one_epoch(model, val_loader, criterion)
        print(f"[FINAL] Epoch {epoch+1}/{FINAL_EPOCHS}: train={train_loss:.4f}, val={val_loss:.4f}")

    return model


@torch.no_grad()
def predict_on_test(model, players_test_joined, player_id_map: Dict[int, int], out_path: str):
    # Build test dataset (no targets)
    # Use relative import to ensure it resolves within package
    from .player_dataset import PlayerSequenceDataset

    # Build UNK map (append one index for unseen players)
    player_id_map_ext = dict(player_id_map)
    unk_index = len(player_id_map_ext)
    # we won't train UNK embedding, but it's at least defined
    # WARNING: we didn't expand model.num_players here, so unseen players will be dropped.
    # Simpler option for now: skip unseen players.
    dataset = PlayerSequenceDataset(
        df=players_test_joined,
        player_id_map=player_id_map_ext,
        feature_cols=FEATURE_COLS,
        target_cols=None,
        max_seq_len=MAX_SEQ_LEN,
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    preds = []

    for game_id, play_id, nfl_id, x, pid in loader:
        x = x.to(DEVICE)
        pid = pid.to(DEVICE)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        y_hat = model(x, pid).cpu().numpy()  # (B, 2)
        for g, p, n, (px, py) in zip(game_id, play_id, nfl_id, y_hat):
            preds.append(
                {
                    "game_id": int(g),
                    "play_id": int(p),
                    "nfl_id": int(n),
                    "pred_x": float(px),
                    "pred_y": float(py),
                }
            )

    import pandas as pd

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(preds).to_csv(out_path, index=False)
    print(f"Saved {len(preds)} predictions to {out_path}")


def main():
    set_seed(RANDOM_SEED)

    players_train, players_test, teamframe_train, teamframe_test = load_processed_data()

    players_train_joined = join_teamframe(players_train, teamframe_train)
    players_test_joined = join_teamframe(players_test, teamframe_test)

    player_id_map = build_player_id_map(players_train_joined)

    # 1) tiny search
    best_cfg = hyperparam_search(players_train_joined, player_id_map, num_trials=4)

    # 2) final training
    model = train_final_model(players_train_joined, player_id_map, best_cfg)

    # 3) save model
    os.makedirs("models", exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "cfg": asdict(best_cfg), "player_id_map": player_id_map},
        "models/physics_transformer.pt",
    )
    print("Saved model to models/physics_transformer.pt")

    # 4) predict on test
    predict_on_test(model, players_test_joined, player_id_map, out_path="models/player_level_predictions.csv")


if __name__ == "__main__":
    main()
